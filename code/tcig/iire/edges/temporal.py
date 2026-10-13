"""Temporal edge family.

Two interaction nodes that were created in the same or nearby rounds
should communicate more strongly than nodes separated by many rounds.
This avoids letting a very old click dominate the response when the
user has clearly moved on to refining a different part of the volume.

The weight is ``exp(-|t_i - t_j| / temporal_scale)``. With the default
``temporal_scale = 1.0``, a click that arrived 5 rounds ago contributes
roughly 1% of the weight of a same-round click; this is a stark decay
chosen deliberately so the trust attribution head has the headroom to
recover or down-weight long-tail clicks via EPICA.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor


def build_temporal_edges(
    round_indices: Sequence[int],
    *,
    temporal_scale: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return an ``[N, N]`` temporal-recency weight matrix.

    Parameters
    ----------
    round_indices:
        Round index per node.
    temporal_scale:
        Number of rounds at which the weight decays to 1/e.
    """
    if temporal_scale <= 0:
        raise ValueError("temporal_scale must be positive")

    rounds = torch.as_tensor(round_indices, dtype=dtype, device=device)
    if rounds.dim() != 1:
        raise ValueError(
            f"round_indices must be 1D [N], got {tuple(rounds.shape)}"
        )

    delta = (rounds.unsqueeze(0) - rounds.unsqueeze(1)).abs()
    weights = torch.exp(-delta / float(temporal_scale))

    eye = torch.eye(rounds.shape[0], dtype=torch.bool, device=weights.device)
    return weights.masked_fill(eye, 0.0)
