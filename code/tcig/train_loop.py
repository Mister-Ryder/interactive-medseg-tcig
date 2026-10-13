"""Prediction-driven multi-round interaction training for TCIG."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor
import torch.nn.functional as F

from tcig.interaction import InteractionUnit


def run_multiround_training_step(
    *,
    model,
    batch: Dict[str, object],
    simulator,
    optimizer,
    device: torch.device,
    max_clicks: int,
    loss_rounds: str = "all",
    dice_threshold: float = 0.5,
) -> Dict[str, object]:
    """Run one prediction-driven interactive training step.

    The loop starts from an empty history, predicts, asks the simulator for the
    next largest-error click, appends it, and repeats. Loss can be applied to
    all predictions, final prediction only, or a random intermediate round.
    """
    if max_clicks < 1:
        raise ValueError("max_clicks must be at least 1")
    if loss_rounds not in ("all", "final", "random"):
        raise ValueError("loss_rounds must be all, final, or random")

    image = batch["image"].to(device=device, dtype=torch.float32)  # type: ignore[index]
    label = batch["label"].to(device=device, dtype=torch.float32)  # type: ignore[index]
    target_cpu = label[0, 0].detach().cpu()

    history: List[InteractionUnit] = []
    round_losses: List[Tensor] = []
    round_dice: List[float] = []
    final_output = None

    with torch.no_grad():
        prediction_cpu = torch.sigmoid(model(image, history)["logits"][0, 0]).detach().cpu()

    for click_index in range(max_clicks):
        click = simulator.next_click(
            prediction_cpu,
            target_cpu,
            round_index=click_index,
            history=history,
        )
        if click is None:
            break
        history.append(click)
        output = model(image, history)
        logits = output["logits"]
        loss = _segmentation_loss(logits, label)
        round_losses.append(loss)
        round_dice.append(float(_soft_dice_score(logits, label).detach().cpu()))
        final_output = output
        prediction_cpu = torch.sigmoid(logits[0, 0]).detach().cpu()

    if not round_losses:
        output = model(image, history)
        logits = output["logits"]
        round_losses.append(_segmentation_loss(logits, label))
        round_dice.append(float(_soft_dice_score(logits, label).detach().cpu()))
        final_output = output

    if loss_rounds == "all":
        total_loss = torch.stack(round_losses).mean()
    elif loss_rounds == "final":
        total_loss = round_losses[-1]
    else:
        index = int(torch.randint(low=0, high=len(round_losses), size=(1,)).item())
        total_loss = round_losses[index]

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    trust_stats = _masked_stats(
        final_output["trust"],  # type: ignore[index,union-attr]
        final_output["node_mask"],  # type: ignore[index,union-attr]
        prefix="trust",
    )
    uncertainty_stats = _masked_stats(
        final_output["uncertainty"],  # type: ignore[index,union-attr]
        final_output["node_mask"],  # type: ignore[index,union-attr]
        prefix="uncertainty",
    )

    return {
        "loss": float(total_loss.detach().cpu()),
        "rounds": len(round_losses),
        "num_clicks": len(history),
        "dice_initial": round_dice[0],
        "dice_final": round_dice[-1],
        "dice_curve": round_dice,
        "history": [unit.to_dict() for unit in history],
        "scatter_gain": float(final_output["scatter_gain"].detach().cpu()),  # type: ignore[index,union-attr]
        "spatial_context_abs_mean": float(final_output["spatial_context"].abs().mean().detach().cpu()),  # type: ignore[index,union-attr]
        **trust_stats,
        **uncertainty_stats,
    }


def _segmentation_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + _soft_dice_loss(logits, target)


def _soft_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    return 1.0 - _soft_dice_score(logits, target)


def _soft_dice_score(logits: Tensor, target: Tensor) -> Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()


def _masked_stats(values: Tensor, mask: Tensor, *, prefix: str) -> Dict[str, float]:
    if values.numel() == 0 or mask.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_std": 0.0,
        }
    valid = values[mask.to(device=values.device, dtype=torch.bool)]
    if valid.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_std": 0.0,
        }
    return {
        f"{prefix}_mean": float(valid.mean().detach().cpu()),
        f"{prefix}_min": float(valid.min().detach().cpu()),
        f"{prefix}_max": float(valid.max().detach().cpu()),
        f"{prefix}_std": float(valid.std(unbiased=False).detach().cpu()),
    }
