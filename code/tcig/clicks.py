"""Shared click simulator for TCIG and baseline evaluation.

The first simulator is deliberately conservative: it follows the standard
largest-error-region protocol and emits gold-standard foreground/background
intent.  It does not assume click noise; robustness experiments can inject
noise through a separate augmentation module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Sequence, Set, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - training environments should have torch.
    torch = None

from .interaction import Coordinate, InteractionUnit


@dataclass
class ClickSimulatorConfig:
    threshold: float = 0.5
    ignore_label: Optional[int] = None
    connectivity: int = 1
    sample_mode: str = "centroid"  # centroid | random
    prefer_false_negative: bool = True
    min_region_size: int = 1
    seed: Optional[int] = None
    exclude_history_radius: int = 0


@dataclass
class ErrorRegion:
    points: Tuple[Coordinate, ...]
    polarity: int

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def centroid_point(self) -> Coordinate:
        ndim = len(self.points[0])
        centroid = [sum(point[axis] for point in self.points) / self.size for axis in range(ndim)]
        return min(
            self.points,
            key=lambda point: sum((point[axis] - centroid[axis]) ** 2 for axis in range(ndim)),
        )


class LargestErrorClickSimulator:
    """Emit the next click from the largest false-positive/false-negative region."""

    def __init__(self, config: Optional[ClickSimulatorConfig] = None) -> None:
        self.config = config or ClickSimulatorConfig()
        self._random = random.Random(self.config.seed)

    def next_click(
        self,
        prediction,
        target,
        round_index: int,
        history: Optional[Sequence[InteractionUnit]] = None,
    ) -> Optional[InteractionUnit]:
        pred_mask, target_mask, valid_mask = self._prepare_masks(prediction, target)

        false_negative = target_mask & (~pred_mask) & valid_mask
        false_positive = (~target_mask) & pred_mask & valid_mask
        if history and self.config.exclude_history_radius > 0:
            excluded = _history_exclusion_mask(
                history,
                false_negative.shape,
                radius=self.config.exclude_history_radius,
            )
            false_negative = false_negative & (~excluded)
            false_positive = false_positive & (~excluded)

        regions = self._candidate_regions(false_negative, false_positive)
        if not regions:
            return None

        region = self._choose_region(regions)
        point = self._sample_point(region)
        return InteractionUnit.click(
            point=point,
            polarity=region.polarity,
            round_index=round_index,
            metadata={
                "simulator": "largest_error",
                "region_size": region.size,
                "history_length": 0 if history is None else len(history),
            },
        )

    def simulate_sequence(
        self,
        predictions: Iterable,
        target,
        start_round: int = 0,
    ) -> List[InteractionUnit]:
        clicks: List[InteractionUnit] = []
        for offset, prediction in enumerate(predictions):
            click = self.next_click(prediction, target, start_round + offset, clicks)
            if click is None:
                break
            clicks.append(click)
        return clicks

    def _prepare_masks(self, prediction, target):
        if torch is None:
            raise RuntimeError("Click simulation currently requires PyTorch tensors")

        pred = prediction.detach().cpu()
        tgt = target.detach().cpu()

        if pred.ndim == tgt.ndim + 1:
            if pred.shape[0] == 1:
                pred = pred[0]
            else:
                pred = pred.argmax(dim=0)

        if pred.is_floating_point():
            pred_mask = pred >= self.config.threshold
        else:
            pred_mask = pred > 0

        valid_mask = torch.ones_like(tgt, dtype=torch.bool)
        if self.config.ignore_label is not None:
            valid_mask = tgt != self.config.ignore_label

        target_mask = tgt > 0
        return pred_mask.bool(), target_mask.bool(), valid_mask.bool()

    def _candidate_regions(self, false_negative, false_positive) -> List[ErrorRegion]:
        regions: List[ErrorRegion] = []
        for polarity, mask in ((1, false_negative), (-1, false_positive)):
            for component in _connected_components(mask, self.config.connectivity):
                if len(component) >= self.config.min_region_size:
                    regions.append(ErrorRegion(points=tuple(component), polarity=polarity))
        return regions

    def _choose_region(self, regions: Sequence[ErrorRegion]) -> ErrorRegion:
        if self.config.prefer_false_negative:
            return max(regions, key=lambda region: (region.size, region.polarity == 1))
        return max(regions, key=lambda region: region.size)

    def _sample_point(self, region: ErrorRegion) -> Coordinate:
        if self.config.sample_mode == "centroid":
            return region.centroid_point
        if self.config.sample_mode == "random":
            return self._random.choice(region.points)
        raise ValueError(f"unknown sample_mode: {self.config.sample_mode}")


def _connected_components(mask, connectivity: int = 1) -> List[List[Coordinate]]:
    coords = [tuple(int(v) for v in coord.tolist()) for coord in mask.nonzero(as_tuple=False)]
    remaining: Set[Coordinate] = set(coords)
    components: List[List[Coordinate]] = []
    ndim = int(mask.ndim)

    while remaining:
        start = remaining.pop()
        queue: deque[Coordinate] = deque([start])
        component = [start]

        while queue:
            point = queue.popleft()
            for neighbour in _neighbours(point, ndim, connectivity):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)

        components.append(component)

    return components


def _history_exclusion_mask(
    history: Sequence[InteractionUnit],
    shape,
    *,
    radius: int,
):
    if torch is None:
        raise RuntimeError("Click simulation currently requires PyTorch tensors")
    mask = torch.zeros(tuple(int(v) for v in shape), dtype=torch.bool)
    if radius <= 0:
        return mask
    ndim = int(mask.ndim)
    ranges = [torch.arange(size, dtype=torch.float32) for size in mask.shape]
    grids = torch.meshgrid(*ranges, indexing="ij")
    radius2 = float(radius * radius)
    for unit in history:
        center = unit.center
        if len(center) != ndim:
            continue
        dist2 = torch.zeros_like(grids[0])
        for axis, grid in enumerate(grids):
            dist2 = dist2 + (grid - float(center[axis])).pow(2)
        mask = mask | (dist2 <= radius2)
    return mask


def _neighbours(point: Coordinate, ndim: int, connectivity: int) -> Iterable[Coordinate]:
    if connectivity != 1:
        raise ValueError("only 4/6-connected neighbourhoods are implemented for now")

    for axis in range(ndim):
        for delta in (-1, 1):
            neighbour = list(point)
            neighbour[axis] += delta
            yield tuple(neighbour)
