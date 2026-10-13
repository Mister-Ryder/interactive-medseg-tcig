"""Canonical representation of a user interaction unit.

TCIG treats clicks, scribbles, and regions as graph nodes.  This module keeps
that node representation small, serialisable, and independent of any specific
training framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

Coordinate = Tuple[int, ...]
InteractionType = Literal["click", "scribble", "region"]


def _as_coordinate(value: Sequence[int]) -> Coordinate:
    if len(value) == 0:
        raise ValueError("coordinate must have at least one dimension")
    return tuple(int(v) for v in value)


def _normalise_points(points: Iterable[Sequence[int]]) -> Tuple[Coordinate, ...]:
    normalised = tuple(_as_coordinate(point) for point in points)
    if not normalised:
        raise ValueError("an interaction unit must contain at least one point")

    ndim = len(normalised[0])
    if any(len(point) != ndim for point in normalised):
        raise ValueError("all points in one interaction unit must have the same dimensionality")

    return normalised


@dataclass(frozen=True)
class InteractionUnit:
    """One node in the cumulative interaction graph.

    Parameters
    ----------
    points:
        One coordinate for a click, or multiple coordinates for scribbles and
        regions. Coordinates are stored in array order, e.g. ``(z, y, x)`` for
        3D tensors and ``(y, x)`` for 2D tensors.
    polarity:
        ``+1`` for foreground intent and ``-1`` for background intent.
    round_index:
        Zero-based interaction round.
    interaction_type:
        ``click``, ``scribble``, or ``region``.
    anatomical_embedding:
        Optional per-unit anatomical-position feature from a frozen TCFF prior.
    metadata:
        Lightweight provenance, such as simulator name or source baseline.
    """

    points: Tuple[Coordinate, ...]
    polarity: int
    round_index: int
    interaction_type: InteractionType = "click"
    anatomical_embedding: Optional[Tuple[float, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _normalise_points(self.points)
        object.__setattr__(self, "points", points)

        if self.polarity not in (-1, 1):
            raise ValueError("polarity must be +1 for foreground or -1 for background")
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if self.interaction_type not in ("click", "scribble", "region"):
            raise ValueError(f"unknown interaction_type: {self.interaction_type}")
        if self.interaction_type == "click" and len(points) != 1:
            raise ValueError("click interactions must contain exactly one point")
        if self.anatomical_embedding is not None:
            object.__setattr__(
                self,
                "anatomical_embedding",
                tuple(float(v) for v in self.anatomical_embedding),
            )

    @classmethod
    def click(
        cls,
        point: Sequence[int],
        polarity: int,
        round_index: int,
        *,
        anatomical_embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "InteractionUnit":
        return cls(
            points=(_as_coordinate(point),),
            polarity=int(polarity),
            round_index=int(round_index),
            interaction_type="click",
            anatomical_embedding=None
            if anatomical_embedding is None
            else tuple(float(v) for v in anatomical_embedding),
            metadata=dict(metadata or {}),
        )

    @property
    def ndim(self) -> int:
        return len(self.points[0])

    @property
    def center(self) -> Tuple[float, ...]:
        count = float(len(self.points))
        return tuple(sum(point[axis] for point in self.points) / count for axis in range(self.ndim))

    @property
    def is_foreground(self) -> bool:
        return self.polarity == 1

    def with_metadata(self, **metadata: Any) -> "InteractionUnit":
        merged = dict(self.metadata)
        merged.update(metadata)
        return InteractionUnit(
            points=self.points,
            polarity=self.polarity,
            round_index=self.round_index,
            interaction_type=self.interaction_type,
            anatomical_embedding=self.anatomical_embedding,
            metadata=merged,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [list(point) for point in self.points],
            "polarity": self.polarity,
            "round_index": self.round_index,
            "interaction_type": self.interaction_type,
            "anatomical_embedding": None
            if self.anatomical_embedding is None
            else list(self.anatomical_embedding),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InteractionUnit":
        return cls(
            points=tuple(tuple(int(v) for v in point) for point in payload["points"]),
            polarity=int(payload["polarity"]),
            round_index=int(payload["round_index"]),
            interaction_type=payload.get("interaction_type", "click"),
            anatomical_embedding=payload.get("anatomical_embedding"),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def sequence_to_dicts(units: Iterable["InteractionUnit"]) -> List[Dict[str, Any]]:
        return [unit.to_dict() for unit in units]

    @staticmethod
    def sequence_from_dicts(payloads: Iterable[Dict[str, Any]]) -> List["InteractionUnit"]:
        return [InteractionUnit.from_dict(payload) for payload in payloads]
