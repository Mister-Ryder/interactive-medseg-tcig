"""Feature builders for the TCIG interaction graph.

The functions here convert canonical ``InteractionUnit`` objects into tensors
that drive the IIRE and EPICA modules. They do not depend on SCC or any
baseline code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from tcig.interaction import InteractionUnit


TYPE_TO_INDEX = {"click": 0, "scribble": 1, "region": 2}


@dataclass(frozen=True)
class InteractionGraphBatch:
    """Batched graph tensors consumed by IIRE and EPICA."""

    node_features: Tensor
    edge_features: Tensor
    node_mask: Tensor


def build_interaction_graph(
    units: Sequence[InteractionUnit],
    *,
    embed_dim: int,
    spatial_scale: float = 32.0,
    affinity_features: Optional[Tensor] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> InteractionGraphBatch:
    """Build one batched interaction graph from a sequence of units.

    Parameters
    ----------
    units:
        Interaction history ordered by round. At least one unit is required.
    embed_dim:
        Desired node feature width for the graph transformer.
    spatial_scale:
        Distance scale for the spatial relation channel.
    affinity_features:
        Optional node descriptors shaped ``[num_nodes, feature_dim]``. When
        provided, cosine similarity fills the fourth edge channel.
    device:
        Optional torch device for the returned tensors.
    """
    if not units:
        raise ValueError("at least one interaction unit is required")
    if embed_dim < 8:
        raise ValueError("embed_dim must be at least 8 for TCIG node features")
    if spatial_scale <= 0:
        raise ValueError("spatial_scale must be positive")

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    node_features = _build_node_features(units, embed_dim, target_device)
    edge_features = _build_edge_features(units, spatial_scale, affinity_features, target_device)
    node_mask = torch.ones(1, len(units), dtype=torch.bool, device=target_device)
    return InteractionGraphBatch(
        node_features=node_features.unsqueeze(0),
        edge_features=edge_features.unsqueeze(0),
        node_mask=node_mask,
    )


def _build_node_features(
    units: Sequence[InteractionUnit],
    embed_dim: int,
    device: torch.device,
) -> Tensor:
    max_ndim = max(unit.ndim for unit in units)
    max_round = max(1, max(unit.round_index for unit in units))
    rows = []

    for unit in units:
        center = torch.tensor(unit.center, dtype=torch.float32, device=device)
        center = F.pad(center, (0, max_ndim - center.numel()))
        coord_scale = torch.clamp(center.abs().max(), min=1.0)
        center = center / coord_scale

        type_one_hot = torch.zeros(3, dtype=torch.float32, device=device)
        type_one_hot[TYPE_TO_INDEX[unit.interaction_type]] = 1.0

        base = torch.cat(
            [
                center,
                torch.tensor(
                    [
                        float(unit.polarity),
                        float(unit.round_index) / float(max_round),
                        float(len(unit.points)),
                    ],
                    dtype=torch.float32,
                    device=device,
                ),
                type_one_hot,
            ]
        )
        rows.append(_repeat_to_width(base, embed_dim))

    return torch.stack(rows, dim=0)


def _build_edge_features(
    units: Sequence[InteractionUnit],
    spatial_scale: float,
    affinity_features: Optional[Tensor],
    device: torch.device,
) -> Tensor:
    centers = torch.tensor([unit.center for unit in units], dtype=torch.float32, device=device)
    rounds = torch.tensor([unit.round_index for unit in units], dtype=torch.float32, device=device)
    polarities = torch.tensor([unit.polarity for unit in units], dtype=torch.float32, device=device)

    distances = torch.cdist(centers, centers, p=2)
    spatial = torch.exp(-distances / spatial_scale)

    round_delta = torch.abs(rounds[:, None] - rounds[None, :])
    temporal = torch.exp(-round_delta)

    polarity = torch.where(
        polarities[:, None] == polarities[None, :],
        torch.ones_like(spatial),
        -torch.ones_like(spatial),
    )

    affinity = torch.zeros_like(spatial)
    if affinity_features is not None:
        if affinity_features.shape[0] != len(units):
            raise ValueError("affinity_features must have one row per interaction unit")
        normalized = F.normalize(affinity_features.to(device=device, dtype=torch.float32), dim=-1)
        affinity = normalized @ normalized.T

    eye = torch.eye(len(units), dtype=torch.bool, device=device)
    spatial = spatial.masked_fill(eye, 0.0)
    temporal = temporal.masked_fill(eye, 0.0)
    polarity = polarity.masked_fill(eye, 0.0)
    affinity = affinity.masked_fill(eye, 0.0)

    return torch.stack([spatial, temporal, polarity, affinity], dim=-1)


def _repeat_to_width(values: Tensor, width: int) -> Tensor:
    repeats = math.ceil(width / values.numel())
    return values.repeat(repeats)[:width]
