"""Behavioural tests for the EPICA Dirichlet contribution head.

These tests pin down the trust math. They cover three regimes:

1. Strong evidence for the positive class drives trust toward 1.
2. Strong evidence for the negative class drives trust toward 0.
3. Uniform/no evidence drives trust toward 0 because uncertainty is high.

A simple fan-out test also confirms that the trust score retains useful
dynamic range when evidence separates across nodes, which is the property the
previous ``sigmoid(positive_prob)`` form silently violated.
"""

import torch
from torch import Tensor, nn

from tcig import DirichletContributionHead


class _FixedEvidenceMLP(nn.Module):
    """Replace the head's MLP with a deterministic logits emitter for tests."""

    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        if logits.ndim != 1:
            raise ValueError("fixed evidence logits must be 1D")
        self.register_buffer("logits", logits)

    def forward(self, node_features: Tensor) -> Tensor:
        target_shape = node_features.shape[:-1] + (self.logits.shape[-1],)
        return self.logits.expand(target_shape)


class _PassThroughEvidenceMLP(nn.Module):
    """Use the first two feature channels as deterministic evidence logits."""

    def forward(self, node_features: Tensor) -> Tensor:
        return node_features[..., :2]


def _build_head_with_fixed_logits(logits: Tensor) -> DirichletContributionHead:
    head = DirichletContributionHead(embed_dim=8, hidden_dim=4)
    head.mlp = _FixedEvidenceMLP(logits)
    return head


def test_strong_positive_evidence_drives_trust_toward_one():
    head = _build_head_with_fixed_logits(torch.tensor([-80.0, 80.0]))
    out = head(torch.zeros(4, 8))

    assert torch.all(out["trust"] > 0.9)
    assert torch.all(out["positive_prob"] > 0.98)
    assert torch.all(out["uncertainty"] < 0.05)


def test_strong_negative_evidence_drives_trust_toward_zero():
    head = _build_head_with_fixed_logits(torch.tensor([80.0, -80.0]))
    out = head(torch.zeros(4, 8))

    assert torch.all(out["trust"] < 0.05)
    assert torch.all(out["positive_prob"] < 0.05)
    assert torch.all(out["uncertainty"] < 0.05)


def test_no_evidence_keeps_trust_near_zero_via_high_uncertainty():
    head = _build_head_with_fixed_logits(torch.tensor([-20.0, -20.0]))
    out = head(torch.zeros(2, 8))

    # softplus(-20) is essentially zero, so alpha ~ 1 and uncertainty ~ 1.
    assert torch.all(out["uncertainty"] > 0.99)
    assert torch.all(out["trust"] < 0.01)


def test_balanced_evidence_keeps_trust_below_half():
    head = _build_head_with_fixed_logits(torch.tensor([2.0, 2.0]))
    out = head(torch.zeros(2, 8))

    # positive_prob is exactly 0.5; certainty < 1, so trust should sit below 0.5.
    assert torch.allclose(out["positive_prob"], torch.full_like(out["positive_prob"], 0.5))
    assert torch.all(out["trust"] < 0.5)
    assert torch.all(out["trust"] > 0.0)


def test_belief_and_alpha_relationship():
    head = _build_head_with_fixed_logits(torch.tensor([1.5, 3.0]))
    out = head(torch.zeros(1, 8))

    expected_belief = (out["alpha"] - 1.0) / out["strength"].unsqueeze(-1)
    assert torch.allclose(out["belief"], expected_belief, atol=1e-6)
    # alpha = evidence + 1; sum should match strength.
    assert torch.allclose(out["alpha"].sum(dim=-1), out["strength"], atol=1e-6)


def test_trust_spans_meaningful_range_when_evidence_separates():
    """Trust should span a useful range when node evidence separates.

    The trust formula should allow explicit positive evidence to approach 1
    and explicit negative evidence to approach 0. This deterministic fan-out
    avoids depending on random initialisation, where evidence is expected to
    be small before training.
    """
    head = DirichletContributionHead(embed_dim=8, hidden_dim=4)
    head.mlp = _PassThroughEvidenceMLP()
    inputs = torch.tensor(
        [
            [80.0, -80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-80.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    out = head(inputs)

    trust = out["trust"]
    assert trust.shape == (4,)
    assert trust.min() < 0.05
    assert trust.max() > 0.9
    assert (trust.max() - trust.min()) > 0.85

"""Unit tests for the four IIRE edge-family builders."""

import torch

from tcig.iire.edges import (
    build_topological_edges,
    build_temporal_edges,
    build_polarity_edges,
    build_affinity_edges,
    combine_edge_channels,
)


def test_topological_edges_diagonal_is_zero_and_symmetric():
    coords = [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]]
    edges = build_topological_edges(coords, spatial_scale=3.0)
    assert edges.shape == (3, 3)
    assert torch.allclose(edges.diagonal(), torch.zeros(3))
    assert torch.allclose(edges, edges.T, atol=1e-6)
    # Closer pair should have a larger weight than farther pair.
    assert edges[0, 1] > edges[0, 2]


def test_topological_edges_with_prediction_bonus():
    coords = [[1.0, 1.0], [4.0, 1.0]]
    pred = torch.zeros(8, 8, dtype=torch.long)
    pred[:, :3] = 1  # left strip
    edges_with = build_topological_edges(coords, spatial_scale=4.0, prediction=pred)
    edges_without = build_topological_edges(coords, spatial_scale=4.0)
    # Different prediction-modulated value than the plain Euclidean weight.
    assert not torch.allclose(edges_with, edges_without)


def test_temporal_edges_decay():
    rounds = [0, 1, 5]
    edges = build_temporal_edges(rounds, temporal_scale=1.0)
    assert edges.shape == (3, 3)
    assert edges[0, 1] > edges[0, 2]
    assert torch.allclose(edges.diagonal(), torch.zeros(3))


def test_polarity_edges_signs():
    polarities = [1, 1, -1]
    edges = build_polarity_edges(polarities)
    assert edges.shape == (3, 3)
    assert edges[0, 1].item() == 1.0
    assert edges[0, 2].item() == -1.0
    assert torch.allclose(edges.diagonal(), torch.zeros(3))


def test_affinity_edges_zero_without_descriptors():
    edges = build_affinity_edges(None, num_nodes=4)
    assert edges.shape == (4, 4)
    assert torch.allclose(edges, torch.zeros_like(edges))


def test_affinity_edges_cosine_with_descriptors():
    descriptors = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edges = build_affinity_edges(descriptors, num_nodes=3)
    assert edges.shape == (3, 3)
    # Parallel descriptors -> cosine 1; orthogonal -> 0.
    assert edges[0, 1].item() > 0.99
    assert abs(edges[0, 2].item()) < 1e-6


def test_combine_edge_channels_assembles_canonical_tensor():
    coords = [[0.0, 0.0], [1.0, 1.0]]
    rounds = [0, 1]
    polarities = [1, -1]
    edges_topo = build_topological_edges(coords)
    edges_time = build_temporal_edges(rounds)
    edges_pol = build_polarity_edges(polarities)
    edges_aff = build_affinity_edges(None, num_nodes=2)
    combined = combine_edge_channels([edges_topo, edges_time, edges_pol, edges_aff])
    assert combined.shape == (1, 2, 2, 4)
    assert torch.allclose(combined[0, :, :, 0], edges_topo)
    assert torch.allclose(combined[0, :, :, 2], edges_pol)

"""Unit tests for the IIRE anisotropic spatial scatter."""

import torch

from tcig.iire.scatter import (
    AnisotropicScatter,
    ScatterConfig,
    build_default_scatter,
    node_coords_from_units,
)
from tcig.interaction import InteractionUnit


def test_scatter_output_shape_3d():
    cfg = ScatterConfig(embed_dim=8, out_channels=4, n_dims=3, work_resolution_factor=2)
    scatter = AnisotropicScatter(cfg)
    nodes = torch.randn(2, 3, 8)
    coords = torch.tensor(
        [[[8.0, 8.0, 8.0], [4.0, 4.0, 4.0], [12.0, 12.0, 12.0]],
         [[2.0, 2.0, 2.0], [6.0, 6.0, 6.0], [10.0, 10.0, 10.0]]],
    )
    trust = torch.tensor([[0.7, 0.5, 0.2], [0.9, 0.8, 0.1]])
    out = scatter(nodes, coords, trust, output_size=(16, 16, 16))
    assert out.shape == (2, 4, 16, 16, 16)
    assert torch.isfinite(out).all()


def test_scatter_output_shape_2d():
    cfg = ScatterConfig(embed_dim=8, out_channels=2, n_dims=2, work_resolution_factor=1)
    scatter = AnisotropicScatter(cfg)
    nodes = torch.randn(1, 2, 8)
    coords = torch.tensor([[[5.0, 5.0], [10.0, 10.0]]])
    trust = torch.tensor([[0.5, 0.5]])
    out = scatter(nodes, coords, trust, output_size=(20, 20))
    assert out.shape == (1, 2, 20, 20)


def test_scatter_zero_trust_yields_zero_output():
    cfg = ScatterConfig(embed_dim=4, out_channels=2, n_dims=2, work_resolution_factor=1)
    scatter = AnisotropicScatter(cfg)
    nodes = torch.randn(1, 2, 4)
    coords = torch.tensor([[[2.0, 2.0], [5.0, 5.0]]])
    trust = torch.zeros(1, 2)
    out = scatter(nodes, coords, trust, output_size=(10, 10))
    assert torch.allclose(out, torch.zeros_like(out))


def test_scatter_mask_zeroes_invalid_nodes():
    cfg = ScatterConfig(embed_dim=4, out_channels=2, n_dims=2, work_resolution_factor=1)
    scatter = AnisotropicScatter(cfg)
    nodes = torch.randn(1, 3, 4)
    coords = torch.tensor([[[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]]])
    trust = torch.ones(1, 3)
    mask_all = torch.ones(1, 3, dtype=torch.bool)
    mask_partial = torch.tensor([[True, True, False]])
    out_all = scatter(nodes, coords, trust, output_size=(10, 10), node_mask=mask_all)
    out_part = scatter(nodes, coords, trust, output_size=(10, 10), node_mask=mask_partial)
    assert not torch.allclose(out_all, out_part)


def test_polarity_sign_flips_contribution():
    cfg = ScatterConfig(
        embed_dim=4,
        out_channels=2,
        n_dims=2,
        work_resolution_factor=1,
        use_polarity_sign=True,
    )
    scatter = AnisotropicScatter(cfg)
    nodes = torch.randn(1, 1, 4)
    coords = torch.tensor([[[5.0, 5.0]]])
    trust = torch.ones(1, 1)
    pos = scatter(nodes, coords, trust, output_size=(10, 10), node_polarities=torch.tensor([[1.0]]))
    neg = scatter(nodes, coords, trust, output_size=(10, 10), node_polarities=torch.tensor([[-1.0]]))
    assert torch.allclose(pos, -neg, atol=1e-5)


def test_node_coords_from_units_packs_variable_lengths():
    seqs = [
        [InteractionUnit.click((1, 2, 3), polarity=1, round_index=0),
         InteractionUnit.click((4, 5, 6), polarity=-1, round_index=1)],
        [InteractionUnit.click((7, 8, 9), polarity=1, round_index=0)],
    ]
    coords, polarities, mask = node_coords_from_units(seqs, n_dims=3)
    assert coords.shape == (2, 2, 3)
    assert polarities.shape == (2, 2)
    assert mask.shape == (2, 2)
    assert mask.tolist() == [[True, True], [True, False]]
    assert coords[1, 1].tolist() == [0.0, 0.0, 0.0]


def test_default_scatter_factory_returns_module():
    scatter = build_default_scatter(embed_dim=16, out_channels=4, n_dims=3)
    assert isinstance(scatter, AnisotropicScatter)
    assert scatter.config.embed_dim == 16
    assert scatter.config.out_channels == 4


def test_invalid_output_size_raises():
    import pytest

    cfg = ScatterConfig(embed_dim=4, out_channels=2, n_dims=3, work_resolution_factor=1)
    scatter = AnisotropicScatter(cfg)
    nodes = torch.randn(1, 1, 4)
    coords = torch.tensor([[[1.0, 1.0, 1.0]]])
    trust = torch.ones(1, 1)
    with pytest.raises(ValueError, match="output_size"):
        scatter(nodes, coords, trust, output_size=(8, 8))


def test_invalid_n_dims_raises():
    import pytest

    with pytest.raises(ValueError, match="n_dims"):
        AnisotropicScatter(ScatterConfig(embed_dim=4, out_channels=2, n_dims=4))

import torch

from tcig.interaction import InteractionUnit
from tcig import build_interaction_graph


def test_build_interaction_graph_shapes_and_mask():
    units = [
        InteractionUnit.click((3, 4), polarity=1, round_index=0),
        InteractionUnit.click((7, 9), polarity=-1, round_index=1),
    ]

    graph = build_interaction_graph(units, embed_dim=16)

    assert graph.node_features.shape == (1, 2, 16)
    assert graph.edge_features.shape == (1, 2, 2, 4)
    assert graph.node_mask.tolist() == [[True, True]]
    assert torch.allclose(torch.diagonal(graph.edge_features[0, :, :, 0]), torch.zeros(2))

