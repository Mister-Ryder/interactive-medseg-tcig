"""Combine per-family edge matrices into the IIRE graph-transformer tensor.

This module produces the ``[B, N, N, edge_dim]`` tensor that
``GraphTransformerLayer.forward`` expects. The four default channels
follow the canonical TCIG ordering used elsewhere in the codebase:

    channel 0: spatial-topological
    channel 1: temporal
    channel 2: polarity
    channel 3: affinity

Callers that drop or reorder channels must also update the
``edge_dim`` argument to :class:`GraphTransformerConfig`.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def combine_edge_channels(matrices: Sequence[Tensor]) -> Tensor:
    """Stack per-family ``[N, N]`` matrices into a single tensor.

    Parameters
    ----------
    matrices:
        Iterable of edge matrices in the canonical TCIG order
        (topological, temporal, polarity, affinity). All tensors must
        share dtype and device. The shapes must match. Each matrix is
        added as a separate channel in the last dim of the output.

    Returns
    -------
    Tensor
        ``[1, N, N, len(matrices)]`` ready to feed into the graph
        transformer's edge input. The leading batch dim is included
        for symmetry with the rest of the IIRE API; callers that batch
        across cases should ``torch.cat`` along ``dim=0``.
    """
    materialised = list(matrices)
    if not materialised:
        raise ValueError("combine_edge_channels needs at least one matrix")

    base_shape = materialised[0].shape
    base_device = materialised[0].device
    base_dtype = materialised[0].dtype
    if len(base_shape) != 2 or base_shape[0] != base_shape[1]:
        raise ValueError(
            f"each edge matrix must be square [N, N]; got {tuple(base_shape)}"
        )
    for index, matrix in enumerate(materialised[1:], start=1):
        if matrix.shape != base_shape:
            raise ValueError(
                f"edge matrix {index} shape {tuple(matrix.shape)} does not "
                f"match channel 0 shape {tuple(base_shape)}"
            )
        if matrix.device != base_device:
            raise ValueError(f"edge matrix {index} device differs from channel 0")
        if matrix.dtype != base_dtype:
            raise ValueError(f"edge matrix {index} dtype differs from channel 0")

    stacked = torch.stack(materialised, dim=-1)  # [N, N, num_channels]
    return stacked.unsqueeze(0)  # [1, N, N, num_channels]
