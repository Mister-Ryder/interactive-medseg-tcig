"""Affinity edge family.

Two interaction nodes that sit in visually similar contexts often
refer to the same anatomical or pathological structure. Affinity is
the cosine similarity between per-node feature descriptors. The
descriptors are sampled from the backbone at each node's voxel
coordinate (or from any node-level descriptor table the caller wants
to use).

When no descriptors are available, the builder returns an all-zero
matrix so that downstream code can still expect a tensor of the right
shape and dtype.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


def build_affinity_edges(
    node_descriptors: Optional[Tensor],
    *,
    num_nodes: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return an ``[N, N]`` affinity weight matrix.

    Parameters
    ----------
    node_descriptors:
        ``[N, feature_dim]`` per-node descriptor tensor. When ``None``
        the affinity edge channel is filled with zeros.
    num_nodes:
        Used only when ``node_descriptors is None`` so the returned
        tensor still has the right shape.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")

    if node_descriptors is None:
        return torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)

    if node_descriptors.dim() != 2:
        raise ValueError(
            f"node_descriptors must be 2D [N, feature_dim], got "
            f"{tuple(node_descriptors.shape)}"
        )
    if node_descriptors.shape[0] != num_nodes:
        raise ValueError(
            f"node_descriptors has {node_descriptors.shape[0]} rows but "
            f"num_nodes is {num_nodes}"
        )

    descriptors = node_descriptors.to(device=device, dtype=dtype)
    normalised = F.normalize(descriptors, p=2.0, dim=-1, eps=1e-8)
    weights = normalised @ normalised.T

    eye = torch.eye(num_nodes, dtype=torch.bool, device=weights.device)
    return weights.masked_fill(eye, 0.0)
