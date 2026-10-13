"""Polarity-consistency edge family.

Foreground and background clicks should reinforce each other in
polarity-consistent ways. Two foreground clicks ought to amplify
shared evidence, two background clicks ought to amplify shared
absence-of-foreground evidence, and a foreground/background pair
should oppose each other to encourage decisive boundaries.

We encode this as a signed weight:
- ``+1`` for matched polarities,
- ``-1`` for mismatched polarities.

The diagonal is masked to zero so a node does not interact with
itself through this channel. Downstream layers can scale this channel
or combine it with the others through learned weights.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor


def build_polarity_edges(
    polarities: Sequence[int],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return an ``[N, N]`` polarity-consistency weight matrix."""
    pol = torch.as_tensor(polarities, dtype=dtype, device=device)
    if pol.dim() != 1:
        raise ValueError(
            f"polarities must be 1D [N], got {tuple(pol.shape)}"
        )

    weights = torch.where(
        pol.unsqueeze(0) == pol.unsqueeze(1),
        torch.ones_like(pol).unsqueeze(0).expand(pol.shape[0], pol.shape[0]),
        -torch.ones_like(pol).unsqueeze(0).expand(pol.shape[0], pol.shape[0]),
    )

    eye = torch.eye(pol.shape[0], dtype=torch.bool, device=weights.device)
    return weights.to(dtype=dtype).masked_fill(eye, 0.0)
