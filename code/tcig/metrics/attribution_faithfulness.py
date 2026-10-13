"""Attribution faithfulness evaluation for EPICA.

Faithfulness measures whether EPICA's per-interaction trust score
ranks clicks in the same order as their actual contribution to the
prediction. We define the actual contribution as the change in Dice
when click ``i`` is removed from the history (leave-one-out):

    Delta_i = Dice(y_t, gt) - Dice(y^(-i), gt)

A large positive ``Delta_i`` means the click was helping; near zero
means the click was inert; negative means the click was hurting.

If EPICA is faithful, predicted trust ``tau`` should correlate
positively with ``Delta``. We use Spearman rank correlation because
the absolute scale of ``Delta`` and ``tau`` are not directly
comparable; only their orderings matter for the gating role.

The module is self-contained: it does not depend on SciPy and
re-implements Spearman correlation in pure PyTorch with average rank
handling for ties. This keeps faithfulness usable in the AutoDL
environment without extra installs.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor

from .dice import binary_dice


def attribution_faithfulness(
    predictor,
    image: Tensor,
    target: Tensor,
    history: Sequence[object],
    predicted_trust: Sequence[float],
    *,
    dice_threshold: float = 0.5,
) -> dict:
    """Compute Spearman correlation between trust and leave-one-out Dice change.

    Parameters
    ----------
    predictor:
        Callable ``(image, history) -> prediction``. Must be
        deterministic for a given history.
    image:
        Image tensor passed to the predictor unchanged.
    target:
        Reference mask.
    history:
        Full interaction history that produced the current prediction.
    predicted_trust:
        Sequence aligned with ``history`` containing the EPICA trust
        scores for each unit.
    dice_threshold:
        Threshold used when the predictor returns a probability map.

    Returns
    -------
    dict
        ``{"spearman": float, "leave_one_out_deltas": list, "predicted_trust": list}``.
        When ``history`` has fewer than 2 units, ``spearman`` is
        ``float('nan')`` because rank correlation is undefined.
    """
    if len(history) != len(predicted_trust):
        raise ValueError(
            f"history length {len(history)} does not match predicted_trust length "
            f"{len(predicted_trust)}"
        )

    deltas = leave_one_out_dice_deltas(
        predictor=predictor,
        image=image,
        target=target,
        history=history,
        dice_threshold=dice_threshold,
    )

    if len(history) < 2:
        return {
            "spearman": float("nan"),
            "leave_one_out_deltas": deltas,
            "predicted_trust": list(map(float, predicted_trust)),
        }

    trust_t = torch.tensor(list(map(float, predicted_trust)), dtype=torch.float64)
    delta_t = torch.tensor(deltas, dtype=torch.float64)
    rho = _spearman_correlation(trust_t, delta_t)
    return {
        "spearman": float(rho),
        "leave_one_out_deltas": deltas,
        "predicted_trust": list(map(float, predicted_trust)),
    }


def leave_one_out_dice_deltas(
    predictor,
    image: Tensor,
    target: Tensor,
    history: Sequence[object],
    *,
    dice_threshold: float = 0.5,
) -> List[float]:
    """Return ``Delta_i = Dice(y_full) - Dice(y^{-i})`` for each click i.

    The predictor is queried once for the full history and ``len(history)``
    additional times for each leave-one-out subset. For large histories
    consider passing a subset of indices to evaluate to bound cost.
    """
    history_list = list(history)
    full_pred = predictor(image, history_list)
    full_dice = binary_dice(full_pred, target, threshold=dice_threshold)

    deltas: List[float] = []
    for index in range(len(history_list)):
        loo_history = history_list[:index] + history_list[index + 1 :]
        loo_pred = predictor(image, loo_history)
        loo_dice = binary_dice(loo_pred, target, threshold=dice_threshold)
        deltas.append(full_dice - loo_dice)
    return deltas


def _spearman_correlation(x: Tensor, y: Tensor) -> Tensor:
    """Spearman rho with average-rank tie handling.

    Implemented in pure torch on the CPU because the input is a tiny
    1D tensor of click contributions. Returns 0.0 when one of the
    vectors is constant (avoids NaN).
    """
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("Spearman expects two 1D tensors of equal length")

    rx = _average_rank(x.to(dtype=torch.float64))
    ry = _average_rank(y.to(dtype=torch.float64))

    rx_centered = rx - rx.mean()
    ry_centered = ry - ry.mean()
    denom = torch.sqrt((rx_centered ** 2).sum() * (ry_centered ** 2).sum())
    if denom.item() == 0.0:
        return torch.tensor(0.0, dtype=torch.float64)
    return (rx_centered * ry_centered).sum() / denom


def _average_rank(values: Tensor) -> Tensor:
    """Return average-rank assignment for a 1D tensor of values."""
    sorted_values, sorted_indices = torch.sort(values)
    ranks = torch.empty_like(values)
    n = values.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        # Average ranks for the tie group (1-indexed conventionally; we use 0-indexed
        # but Spearman is invariant to additive constants so 0-indexed is fine).
        avg_rank = (i + j) / 2.0
        ranks[sorted_indices[i : j + 1]] = avg_rank
        i = j + 1
    return ranks
