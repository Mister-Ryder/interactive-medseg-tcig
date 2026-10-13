"""Evidential per-interaction contribution head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DirichletContributionHead(nn.Module):
    """Predict positive/negative contribution evidence for each interaction node.

    The head follows the Subjective-Logic / Evidential-Deep-Learning recipe:

    - A small MLP emits non-negative evidence per class through softplus.
    - Dirichlet parameters are ``alpha = evidence + 1``.
    - Total mass is ``S = sum(alpha)``; uncertainty is ``U = K / S`` for ``K``
      classes; the probability of the positive-contribution class (index 1) is
      ``alpha[1] / S``.

    The trust score that drives EPICA's graph-aggregation weight, the
    voxel-level prompt gate, and the user-facing attribution overlay is

    ``trust = positive_prob * (1 - U)``

    which keeps the full ``[0, 1]`` range available. The Subjective-Logic
    belief mass is also returned so callers that prefer the stricter form
    ``belief = (alpha - 1) / S`` can consume it directly.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, node_features: Tensor) -> dict:
        evidence = F.softplus(self.mlp(node_features))
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        safe_strength = torch.clamp(strength, min=1e-6)
        uncertainty = float(self.num_classes) / safe_strength
        positive_prob = alpha[..., 1] / safe_strength
        belief = (alpha - 1.0) / safe_strength.unsqueeze(-1)
        certainty = (1.0 - uncertainty).clamp(min=0.0)
        trust = positive_prob * certainty

        return {
            "evidence": evidence,
            "alpha": alpha,
            "strength": strength,
            "uncertainty": uncertainty,
            "positive_prob": positive_prob,
            "belief": belief,
            "trust": trust,
        }
