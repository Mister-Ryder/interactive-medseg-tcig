"""Metric primitives shared by TCIG and the interactive baselines."""

from .dice import binary_dice
from .noc import (
    InteractivePredictor,
    InteractiveEpisodeResult,
    interactive_noc_episode,
    summarise_noc_episodes,
)
from .attribution_faithfulness import (
    attribution_faithfulness,
    leave_one_out_dice_deltas,
)
from .topology import (
    betti0_error,
    binary_cldice,
    component_detection_stats,
    connected_component_count,
    topology_metrics,
)

__all__ = [
    "binary_dice",
    "InteractivePredictor",
    "InteractiveEpisodeResult",
    "interactive_noc_episode",
    "summarise_noc_episodes",
    "attribution_faithfulness",
    "leave_one_out_dice_deltas",
    "betti0_error",
    "binary_cldice",
    "component_detection_stats",
    "connected_component_count",
    "topology_metrics",
]
