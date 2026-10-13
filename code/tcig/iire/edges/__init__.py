"""Edge-family builders for the IIRE interaction graph.

The TCIG plan defines four families:

- ``topological``: same-component / soft-skeleton geodesic affinity from
  the current prediction. The current implementation falls back to negative-exponential
  Euclidean distance with an optional prediction-aware modulation.
- ``temporal``: decaying weight with interaction-round distance.
- ``polarity``: positive for matched foreground/background polarity,
  negative when polarities disagree.
- ``affinity``: cosine similarity between per-node feature samples
  drawn from the backbone at each node's voxel coordinate.

Each builder is a pure function returning a ``[N, N]`` weight matrix
with the diagonal masked to zero. The collection is meant to be
recombined into the four-channel edge tensor consumed by
``GraphTransformerLayer``.

This subpackage is additive: it does not modify
``tcig.graph.build_interaction_graph`` so concurrent edits
to that file remain safe. New code should prefer these builders, and
``graph.py`` can call into them in a future refactor.
"""

from .topological import build_topological_edges
from .temporal import build_temporal_edges
from .polarity import build_polarity_edges
from .affinity import build_affinity_edges
from .combine import combine_edge_channels

__all__ = [
    "build_topological_edges",
    "build_temporal_edges",
    "build_polarity_edges",
    "build_affinity_edges",
    "combine_edge_channels",
]
