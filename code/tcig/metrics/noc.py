"""Number-of-clicks (NoC) and Dice-versus-clicks evaluation.

This module is intentionally model-agnostic. The interactive loop only
requires that the caller pass an object implementing the
:class:`InteractivePredictor` protocol: given an image and the current
interaction history it returns a prediction. The simulator (any class
exposing ``next_click(prediction, target, round_index, history)``) is
shared with TCIG training, which keeps the NoC numbers and the
training-time click distribution consistent.

Typical use::

    from tcig.clicks import LargestErrorClickSimulator
    from tcig.metrics import interactive_noc_episode, summarise_noc_episodes

    simulator = LargestErrorClickSimulator()
    episodes = []
    for image, target in dataset:
        episode = interactive_noc_episode(
            predictor=my_model_predictor,
            simulator=simulator,
            image=image,
            target=target,
            thresholds=(0.85, 0.9),
            max_clicks=20,
        )
        episodes.append(episode)
    summary = summarise_noc_episodes(episodes, thresholds=(0.85, 0.9))

The result includes per-threshold NoC, Dice-versus-clicks curve and
final Dice. ``summarise_noc_episodes`` aggregates across cases for a
ready-to-paste table row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

import math

import torch
from torch import Tensor

from .dice import binary_dice


class InteractivePredictor(Protocol):
    """Minimum contract the NoC harness expects from a model.

    The predictor receives the image and the full interaction history
    so it can recompute the prediction in whatever way it wants
    (re-running the network from scratch, incremental refinement, or
    cached intermediate state). It must be deterministic for a given
    history if the NoC numbers are to be reproducible.
    """

    def __call__(self, image: Tensor, history: Sequence[object]) -> Tensor:  # pragma: no cover - protocol
        ...


@dataclass
class InteractiveEpisodeResult:
    """Outcome of one interactive episode for a single case.

    Attributes
    ----------
    dice_curve:
        ``[max_clicks + 1]`` Dice values, one per round including the
        zero-click baseline. Truncated if the simulator emits ``None``
        early (no error region left).
    threshold_noc:
        ``{threshold: clicks_to_reach}``. ``None`` indicates the
        threshold was never reached within ``max_clicks``.
    final_dice:
        Last achieved Dice value.
    num_clicks:
        Length of ``dice_curve`` minus 1.
    failure:
        True when none of the requested thresholds are reached.
    """

    dice_curve: List[float]
    threshold_noc: dict
    final_dice: float
    num_clicks: int
    failure: bool


def interactive_noc_episode(
    predictor: InteractivePredictor,
    simulator,
    image: Tensor,
    target: Tensor,
    *,
    thresholds: Iterable[float] = (0.85, 0.9),
    max_clicks: int = 20,
    initial_prediction: Optional[Tensor] = None,
    initial_history: Optional[Sequence[object]] = None,
    dice_threshold: float = 0.5,
) -> InteractiveEpisodeResult:
    """Run one interactive click episode.

    Parameters
    ----------
    predictor:
        Object implementing :class:`InteractivePredictor`.
    simulator:
        Click simulator with a ``next_click`` method following the
        signature used by :class:`LargestErrorClickSimulator`.
    image:
        ``[C, *spatial]`` or ``[*spatial]`` image tensor passed to the
        predictor without modification.
    target:
        Reference label map. Anything > 0 is treated as foreground.
    thresholds:
        Dice levels at which to record NoC.
    max_clicks:
        Maximum number of clicks; the simulator is asked at most this
        many times.
    initial_prediction:
        Optional Round-0 prediction. When omitted, the function asks
        the predictor for a no-click prediction first.
    initial_history:
        Optional history to seed the episode with (e.g. a pre-existing
        adversarial click for robustness evaluations).
    dice_threshold:
        Probability cut-off applied when the predictor returns a
        probability map.

    Returns
    -------
    InteractiveEpisodeResult
    """
    if max_clicks < 1:
        raise ValueError("max_clicks must be at least 1")
    thresholds_tuple = tuple(sorted(set(float(t) for t in thresholds)))

    history: List[object] = list(initial_history) if initial_history is not None else []
    dice_curve: List[float] = []

    if initial_prediction is None:
        prediction = predictor(image, history)
    else:
        prediction = initial_prediction
    dice_curve.append(binary_dice(prediction, target, threshold=dice_threshold))

    threshold_noc = {t: None for t in thresholds_tuple}
    _record_thresholds(dice_curve, threshold_noc, click_index=0)

    for click_idx in range(1, max_clicks + 1):
        click = simulator.next_click(prediction, target, round_index=click_idx - 1, history=history)
        if click is None:
            break
        history.append(click)
        prediction = predictor(image, history)
        dice_curve.append(binary_dice(prediction, target, threshold=dice_threshold))
        _record_thresholds(dice_curve, threshold_noc, click_index=click_idx)
        if all(value is not None for value in threshold_noc.values()):
            # Stop early once all thresholds are reached to save compute,
            # unless the caller wants the full Dice curve through max_clicks.
            break

    final_dice = dice_curve[-1] if dice_curve else 0.0
    failure = all(value is None for value in threshold_noc.values()) if threshold_noc else False
    return InteractiveEpisodeResult(
        dice_curve=dice_curve,
        threshold_noc=threshold_noc,
        final_dice=final_dice,
        num_clicks=len(dice_curve) - 1,
        failure=failure,
    )


def summarise_noc_episodes(
    episodes: Sequence[InteractiveEpisodeResult],
    *,
    thresholds: Iterable[float] = (0.85, 0.9),
    max_clicks: int = 20,
) -> dict:
    """Aggregate a list of episodes into the per-method summary table.

    Returns
    -------
    dict
        Keys include ``mean_final_dice``, ``failure_rate``, per-threshold
        ``noc@<value>``, and a per-click-index ``mean_dice_curve`` with
        ``None`` slots for clicks that some episodes never reached.
    """
    if not episodes:
        return {"mean_final_dice": 0.0, "failure_rate": 0.0, "mean_dice_curve": []}

    thresholds_tuple = tuple(sorted(set(float(t) for t in thresholds)))
    summary: dict = {}
    summary["mean_final_dice"] = sum(ep.final_dice for ep in episodes) / len(episodes)

    failure_count = sum(1 for ep in episodes if ep.failure)
    summary["failure_rate"] = failure_count / len(episodes)

    for threshold in thresholds_tuple:
        reached = [ep.threshold_noc.get(threshold) for ep in episodes]
        reached_clicks = [value for value in reached if value is not None]
        if reached_clicks:
            mean = sum(reached_clicks) / len(reached_clicks)
        else:
            mean = float(max_clicks) + 1.0
        coverage = len(reached_clicks) / len(episodes)
        summary[f"noc@{threshold}"] = mean
        summary[f"coverage@{threshold}"] = coverage

    # Mean Dice curve up to max_clicks. Slots without enough episodes return None.
    mean_curve: List[Optional[float]] = []
    for idx in range(max_clicks + 1):
        values = [ep.dice_curve[idx] for ep in episodes if len(ep.dice_curve) > idx]
        if values:
            mean_curve.append(sum(values) / len(values))
        else:
            mean_curve.append(None)
    summary["mean_dice_curve"] = mean_curve
    summary["dice_curve_coverage"] = [
        sum(1 for ep in episodes if len(ep.dice_curve) > idx) / len(episodes)
        for idx in range(max_clicks + 1)
    ]
    return summary


def _record_thresholds(
    dice_curve: Sequence[float],
    threshold_noc: dict,
    *,
    click_index: int,
) -> None:
    last_dice = dice_curve[-1]
    for threshold, recorded in threshold_noc.items():
        if recorded is None and last_dice >= threshold:
            threshold_noc[threshold] = click_index
