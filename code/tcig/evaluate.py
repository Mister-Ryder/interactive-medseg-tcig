"""TopCoW interactive evaluation for TCIG."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch

from tcig.metrics import (
    attribution_faithfulness,
    interactive_noc_episode,
    summarise_noc_episodes,
)
from tcig.clicks import ClickSimulatorConfig, LargestErrorClickSimulator


class TCIGPredictor:
    """Adapter from `TCIGSegmenter` to the NoC evaluator protocol."""

    def __init__(self, model, *, device: torch.device) -> None:
        self.model = model
        self.device = device

    def __call__(self, image: torch.Tensor, history: Sequence[object]) -> torch.Tensor:
        if image.ndim == 4:
            image_b = image.unsqueeze(0)
        elif image.ndim == 5:
            image_b = image
        else:
            raise ValueError(f"image must be [C,D,H,W] or [B,C,D,H,W], got {tuple(image.shape)}")
        image_b = image_b.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            prob = self.model.predict(image_b, history)
        return prob[0, 0].detach().cpu()


def evaluate_topcow_noc(
    *,
    model,
    dataset,
    device: torch.device,
    thresholds: Iterable[float] = (0.85, 0.9),
    max_clicks: int = 20,
    max_cases: int = 0,
    include_faithfulness: bool = True,
    exclude_history_radius: int = 2,
) -> Dict[str, object]:
    model.eval()
    predictor = TCIGPredictor(model, device=device)
    simulator = LargestErrorClickSimulator(
        ClickSimulatorConfig(exclude_history_radius=exclude_history_radius)
    )
    episodes = []
    cases: List[Dict[str, object]] = []
    faithfulness_values: List[float] = []
    limit = len(dataset) if max_cases <= 0 else min(max_cases, len(dataset))

    for index in range(limit):
        item = dataset[index]
        image = item["image"]
        target = item["label"][0]
        episode = interactive_noc_episode(
            predictor=predictor,
            simulator=simulator,
            image=image,
            target=target,
            thresholds=thresholds,
            max_clicks=max_clicks,
        )
        episodes.append(episode)
        case_row = {
            "case_id": item["case_id"],
            "final_dice": episode.final_dice,
            "num_clicks": episode.num_clicks,
            "threshold_noc": {str(k): v for k, v in episode.threshold_noc.items()},
            "dice_curve": episode.dice_curve,
        }
        if include_faithfulness and episode.num_clicks >= 2:
            history = _replay_history(
                predictor=predictor,
                simulator=simulator,
                image=image,
                target=target,
                max_clicks=min(max_clicks, episode.num_clicks),
            )
            trust = _predict_trust(model, image, history, device)
            faith = attribution_faithfulness(
                predictor=predictor,
                image=image,
                target=target,
                history=history,
                predicted_trust=trust,
            )
            case_row["attribution_spearman"] = faith["spearman"]
            if faith["spearman"] == faith["spearman"]:
                faithfulness_values.append(float(faith["spearman"]))
        cases.append(case_row)

    summary = summarise_noc_episodes(episodes, thresholds=thresholds, max_clicks=max_clicks)
    summary["num_cases"] = limit
    summary["cases"] = cases
    if faithfulness_values:
        summary["mean_attribution_spearman"] = sum(faithfulness_values) / len(faithfulness_values)
    else:
        summary["mean_attribution_spearman"] = None
    return summary


def _replay_history(*, predictor, simulator, image, target, max_clicks: int):
    history = []
    prediction = predictor(image, history)
    for click_idx in range(1, max_clicks + 1):
        click = simulator.next_click(prediction, target, round_index=click_idx - 1, history=history)
        if click is None:
            break
        history.append(click)
        prediction = predictor(image, history)
    return history


def _predict_trust(model, image, history, device: torch.device) -> List[float]:
    image_b = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = model(image_b, history)
    trust = output["trust"]
    if trust.numel() == 0:
        return []
    return [float(v) for v in trust[0].detach().cpu().tolist()]
