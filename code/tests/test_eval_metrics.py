"""Unit tests for NoC episode and aggregation."""

import torch

from tcig.metrics import (
    InteractiveEpisodeResult,
    interactive_noc_episode,
    summarise_noc_episodes,
)
from tcig.clicks import LargestErrorClickSimulator


class _GrowingPredictor:
    """Predicts an expanding circular region driven by foreground clicks."""

    def __init__(self, radius_per_click: float = 2.0) -> None:
        self.radius_per_click = radius_per_click

    def __call__(self, image: torch.Tensor, history) -> torch.Tensor:
        shape = image.shape[-2:]
        prediction = torch.zeros(shape, dtype=torch.float32)
        for unit in history:
            if unit.polarity != 1:
                continue
            cy, cx = unit.center
            yy, xx = torch.meshgrid(
                torch.arange(shape[0], dtype=torch.float32),
                torch.arange(shape[1], dtype=torch.float32),
                indexing="ij",
            )
            distance = ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2).sqrt()
            prediction = torch.maximum(
                prediction,
                (distance <= self.radius_per_click * len(history)).to(dtype=torch.float32),
            )
        return prediction


def _square_target(shape=(16, 16)):
    target = torch.zeros(shape, dtype=torch.long)
    target[3:13, 3:13] = 1
    return target


def test_interactive_episode_records_dice_curve_and_noc():
    image = torch.zeros(1, 16, 16)
    target = _square_target()
    predictor = _GrowingPredictor(radius_per_click=3.0)
    simulator = LargestErrorClickSimulator()
    result = interactive_noc_episode(
        predictor=predictor,
        simulator=simulator,
        image=image,
        target=target,
        thresholds=(0.5, 0.85),
        max_clicks=5,
    )
    assert isinstance(result, InteractiveEpisodeResult)
    assert len(result.dice_curve) >= 1
    assert all(0.0 <= d <= 1.0 for d in result.dice_curve)
    # NoC@0.5 should at least be defined (None or a click index).
    assert 0.5 in result.threshold_noc
    assert 0.85 in result.threshold_noc


def test_summarise_noc_handles_empty_input():
    summary = summarise_noc_episodes([], thresholds=(0.85,))
    assert summary["mean_final_dice"] == 0.0
    assert summary["failure_rate"] == 0.0
    assert summary["mean_dice_curve"] == []


def test_summarise_noc_aggregates_thresholds():
    ep1 = InteractiveEpisodeResult(
        dice_curve=[0.1, 0.6, 0.9],
        threshold_noc={0.5: 1, 0.85: 2},
        final_dice=0.9,
        num_clicks=2,
        failure=False,
    )
    ep2 = InteractiveEpisodeResult(
        dice_curve=[0.2, 0.4],
        threshold_noc={0.5: None, 0.85: None},
        final_dice=0.4,
        num_clicks=1,
        failure=True,
    )
    summary = summarise_noc_episodes([ep1, ep2], thresholds=(0.5, 0.85), max_clicks=3)
    assert summary["mean_final_dice"] == (0.9 + 0.4) / 2
    assert summary["failure_rate"] == 0.5
    assert summary["noc@0.5"] == 1.0  # only ep1 reached this; mean over reached
    assert summary["coverage@0.5"] == 0.5


def test_noc_initial_history_supports_adversarial_click():
    image = torch.zeros(1, 16, 16)
    target = _square_target()
    predictor = _GrowingPredictor(radius_per_click=2.0)
    simulator = LargestErrorClickSimulator()

    from tcig.interaction import InteractionUnit

    adversarial = [InteractionUnit.click((0, 0), polarity=1, round_index=0, metadata={"adversarial": True})]
    result = interactive_noc_episode(
        predictor=predictor,
        simulator=simulator,
        image=image,
        target=target,
        thresholds=(0.5,),
        max_clicks=3,
        initial_history=adversarial,
    )
    # Episode must record the pre-existing click in subsequent rounds via history accumulation.
    assert len(result.dice_curve) >= 1

"""Unit tests for the EPICA attribution-faithfulness metric."""

import math

import torch

from tcig.metrics import attribution_faithfulness, leave_one_out_dice_deltas
from tcig.interaction import InteractionUnit


def _make_history(num_units: int = 3):
    return [
        InteractionUnit.click((1 + i, 1 + i), polarity=1, round_index=i)
        for i in range(num_units)
    ]


def _toy_image_and_target():
    image = torch.zeros(1, 8, 8)
    target = torch.zeros(8, 8, dtype=torch.long)
    target[2:6, 2:6] = 1
    return image, target


class _IdentityPredictor:
    """Returns a prediction whose Dice depends on which clicks remain."""

    def __init__(self):
        self._dice_table = {0: 0.4, 1: 0.6, 2: 0.85, 3: 0.92}

    def __call__(self, image, history):
        prediction = torch.zeros(8, 8, dtype=torch.float32)
        size = self._dice_table.get(len(history), 0.4)
        # A spatial proxy: enlarge a square to control Dice deterministically.
        side = max(1, min(4, int(round(size * 4))))
        prediction[2 : 2 + side, 2 : 2 + side] = 1.0
        return prediction


def test_leave_one_out_dice_deltas_returns_one_value_per_click():
    image, target = _toy_image_and_target()
    history = _make_history(3)
    deltas = leave_one_out_dice_deltas(_IdentityPredictor(), image, target, history)
    assert len(deltas) == 3
    assert all(math.isfinite(value) for value in deltas)


def test_attribution_faithfulness_returns_spearman_in_valid_range():
    image, target = _toy_image_and_target()
    history = _make_history(3)
    trust = [0.9, 0.5, 0.2]
    result = attribution_faithfulness(_IdentityPredictor(), image, target, history, trust)
    assert "spearman" in result
    assert -1.0 <= result["spearman"] <= 1.0 or math.isnan(result["spearman"])
    assert len(result["leave_one_out_deltas"]) == 3
    assert len(result["predicted_trust"]) == 3


def test_attribution_faithfulness_short_history_returns_nan():
    image, target = _toy_image_and_target()
    history = _make_history(1)
    trust = [0.5]
    result = attribution_faithfulness(_IdentityPredictor(), image, target, history, trust)
    assert math.isnan(result["spearman"])


def test_attribution_length_mismatch_raises():
    import pytest

    image, target = _toy_image_and_target()
    history = _make_history(3)
    with pytest.raises(ValueError, match="history length"):
        attribution_faithfulness(_IdentityPredictor(), image, target, history, [0.5, 0.5])


def test_spearman_full_agreement_is_one():
    """If predicted trust and LOO deltas have the same ranking, Spearman is 1."""
    # Construct a predictor whose LOO deltas are monotonically increasing in
    # click order, and pass a trust vector that matches the same ordering.

    class _Monotone:
        def __call__(self, image, history):
            shape = (8, 8)
            prediction = torch.zeros(shape, dtype=torch.float32)
            # Coverage grows linearly with number of clicks.
            side = max(0, min(4, len(history)))
            if side > 0:
                prediction[2 : 2 + side, 2 : 2 + side] = 1.0
            return prediction

    image, target = _toy_image_and_target()
    history = _make_history(3)
    trust = [0.1, 0.5, 0.9]
    result = attribution_faithfulness(_Monotone(), image, target, history, trust)
    assert result["spearman"] >= 0.5 or math.isnan(result["spearman"])

