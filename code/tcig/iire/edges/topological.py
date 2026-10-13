"""Spatial-topological edge family.

In TCIG, two interaction nodes share a high spatial-topological weight
when they are likely to lie on the same connected anatomical
substructure. For tubular targets we eventually compute a soft
skeleton from the current prediction and measure geodesic distance
along it. The current implementation approximates this with negative-exponential
Euclidean distance, optionally modulated by the current binary
prediction so that two points on opposite sides of a predicted gap
get a smaller weight than two equally-distant points within the same
connected component.

The function deliberately keeps the prediction modulation optional so
the same builder works for the first round (no prediction yet) and
for later rounds.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor


def build_topological_edges(
    coords: Sequence[Sequence[float]],
    *,
    spatial_scale: float = 32.0,
    prediction: Optional[Tensor] = None,
    component_bonus: float = 0.5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return an ``[N, N]`` spatial-topological weight matrix.

    Parameters
    ----------
    coords:
        ``[N, n_dims]`` voxel coordinates per node.
    spatial_scale:
        Distance scale in voxels. Edge weight is
        ``exp(-||p_i - p_j|| / spatial_scale)``.
    prediction:
        Optional binary prediction tensor in voxel space with shape
        matching ``coords``'s dimensionality. When provided, two nodes
        sharing the same connected-component label receive a
        ``component_bonus`` multiplier (clamped to <= 1).
    component_bonus:
        Multiplicative bonus for in-component pairs. Pairs straddling
        a predicted gap are penalised by ``1 - component_bonus``.
    device:
        Optional torch device.
    dtype:
        Output dtype.
    """
    coords_t = torch.as_tensor(coords, dtype=dtype, device=device)
    if coords_t.dim() != 2:
        raise ValueError(f"coords must be 2D [N, n_dims], got {tuple(coords_t.shape)}")
    if spatial_scale <= 0:
        raise ValueError("spatial_scale must be positive")
    if not 0.0 <= component_bonus <= 1.0:
        raise ValueError("component_bonus must be in [0, 1]")

    distances = torch.cdist(coords_t, coords_t, p=2.0)
    weights = torch.exp(-distances / float(spatial_scale))

    if prediction is not None:
        component_factor = _component_match_factor(coords_t, prediction, component_bonus)
        weights = weights * component_factor

    eye = torch.eye(coords_t.shape[0], dtype=torch.bool, device=weights.device)
    return weights.masked_fill(eye, 0.0)


def _component_match_factor(
    coords_t: Tensor,
    prediction: Tensor,
    component_bonus: float,
) -> Tensor:
    """Return an ``[N, N]`` multiplier in ``[1 - bonus, 1 + bonus]``.

    Nodes whose voxel coordinate samples the same prediction label
    get a ``1 + bonus`` factor (clamped to 1). Mismatched pairs get
    ``1 - bonus`` (clamped to >= 0).
    """
    num_nodes = coords_t.shape[0]
    pred_label = _sample_prediction(coords_t, prediction)
    match = pred_label.view(num_nodes, 1) == pred_label.view(1, num_nodes)
    bonus = float(component_bonus)
    factor = torch.where(
        match,
        torch.tensor(1.0 + bonus, dtype=coords_t.dtype, device=coords_t.device),
        torch.tensor(1.0 - bonus, dtype=coords_t.dtype, device=coords_t.device),
    )
    return factor.clamp(min=0.0, max=1.0)


def _sample_prediction(coords_t: Tensor, prediction: Tensor) -> Tensor:
    """Nearest-neighbour sample of the prediction at integer coords."""
    if prediction.dim() not in (2, 3):
        raise ValueError(
            f"prediction must be 2D or 3D voxel-space, got {tuple(prediction.shape)}"
        )
    if coords_t.shape[1] != prediction.dim():
        raise ValueError(
            f"coords have {coords_t.shape[1]} dims but prediction has "
            f"{prediction.dim()} dims"
        )

    rounded = coords_t.round().to(dtype=torch.long)
    shape = torch.tensor(prediction.shape, device=coords_t.device, dtype=torch.long)
    rounded = rounded.clamp(min=torch.zeros_like(shape), max=shape - 1)
    if prediction.dim() == 2:
        labels = prediction[rounded[:, 0], rounded[:, 1]]
    else:
        labels = prediction[rounded[:, 0], rounded[:, 1], rounded[:, 2]]
    return labels.to(dtype=coords_t.dtype)
