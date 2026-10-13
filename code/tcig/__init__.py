"""TCIG core package."""

from .epica import DirichletContributionHead
from .graph import InteractionGraphBatch, build_interaction_graph

__all__ = [
    "DirichletContributionHead",
    "InteractionGraphBatch",
    "build_interaction_graph",
]
