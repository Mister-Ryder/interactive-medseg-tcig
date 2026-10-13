"""Reusable TCIG segmentation model with a history-based API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tcig.interaction import InteractionUnit
from tcig.epica import DirichletContributionHead
from tcig.iire import GraphTransformer, GraphTransformerConfig
from tcig.iire.scatter import AnisotropicScatter, ScatterConfig, node_coords_from_units
from tcig.graph import build_interaction_graph


History = Sequence[InteractionUnit]
HistoryBatch = Union[History, Sequence[History]]
InteractionMode = Literal["tcig", "prompt_only", "no_interaction"]


@dataclass
class TCIGSegmenterConfig:
    embed_dim: int = 32
    base_channels: int = 8
    graph_layers: int = 2
    graph_heads: int = 4
    edge_dim: int = 4
    prompt_sigma: float = 5.0
    scatter_init_sigma: float = 4.0
    scatter_work_resolution_factor: int = 2
    scatter_gain_init: float = 0.1


class TCIGSegmenter(nn.Module):
    """Tiny but complete TCIG model for TopCoW bring-up.

    The model consumes an image and a cumulative interaction history. Prompt
    maps, graph tensors, EPICA trust, IIRE updates, and spatial scatter are all
    built inside `forward`, which makes eval and training distribution-aligned.
    """

    def __init__(self, config: TCIGSegmenterConfig) -> None:
        super().__init__()
        if config.embed_dim % config.graph_heads != 0:
            raise ValueError("embed_dim must be divisible by graph_heads")
        if not 0.0 < config.scatter_gain_init < 1.0:
            raise ValueError("scatter_gain_init must be in (0, 1)")

        self.config = config
        self.epica = DirichletContributionHead(
            embed_dim=config.embed_dim,
            hidden_dim=config.base_channels * 4,
        )
        self.iire = GraphTransformer(
            GraphTransformerConfig(
                embed_dim=config.embed_dim,
                num_heads=config.graph_heads,
                num_layers=config.graph_layers,
                edge_dim=config.edge_dim,
                dropout=0.0,
            )
        )
        self.scatter = AnisotropicScatter(
            ScatterConfig(
                embed_dim=config.embed_dim,
                out_channels=config.base_channels,
                n_dims=3,
                init_sigma=config.scatter_init_sigma,
                work_resolution_factor=config.scatter_work_resolution_factor,
                upsample_mode="trilinear",
                use_polarity_sign=True,
            )
        )
        init_logit = torch.logit(torch.tensor(float(config.scatter_gain_init)))
        self.scatter_gain_logit = nn.Parameter(init_logit)
        self.encoder = nn.Sequential(
            nn.Conv3d(3, config.base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(config.base_channels),
            nn.GELU(),
            nn.Conv3d(config.base_channels, config.base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(config.base_channels),
            nn.GELU(),
        )
        self.head = nn.Conv3d(config.base_channels, 1, kernel_size=1)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(
        self,
        image: Tensor,
        histories: HistoryBatch,
        *,
        interaction_mode: InteractionMode = "tcig",
    ) -> Dict[str, Tensor]:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("image must be shaped [B, 1, D, H, W]")
        if interaction_mode not in ("tcig", "prompt_only", "no_interaction"):
            raise ValueError("interaction_mode must be tcig, prompt_only, or no_interaction")

        history_batch = _normalise_histories(histories, batch_size=image.shape[0])
        if interaction_mode == "no_interaction":
            history_batch = [[] for _ in history_batch]
        if len(history_batch) != 1:
            raise ValueError("TCIG currently supports batch_size=1")

        prompts = build_prompt_maps(
            history_batch,
            output_size=image.shape[2:],
            sigma=self.config.prompt_sigma,
            device=image.device,
            dtype=image.dtype,
        )
        features = self.encoder(torch.cat([image, prompts], dim=1))

        history = list(history_batch[0])
        if history and interaction_mode == "tcig":
            graph_batch = build_interaction_graph(
                history,
                embed_dim=self.config.embed_dim,
                device=image.device,
            )
            pre = self.epica(graph_batch.node_features)
            updated = self.iire(
                graph_batch.node_features,
                graph_batch.edge_features,
                node_trust=pre["trust"],
                node_mask=graph_batch.node_mask,
            )
            post = self.epica(updated)
            node_coords, node_polarities, node_mask = node_coords_from_units(
                [history],
                n_dims=3,
                device=image.device,
                dtype=image.dtype,
            )
            prompt_context = self.scatter(
                node_features=updated,
                node_coords=node_coords,
                node_trust=post["trust"],
                output_size=features.shape[2:],
                node_mask=node_mask & graph_batch.node_mask,
                node_polarities=node_polarities,
            )
            scatter_gain = torch.sigmoid(self.scatter_gain_logit)
            spatial_context = scatter_gain * prompt_context
            trust = post["trust"]
            uncertainty = post["uncertainty"]
            node_mask_out = graph_batch.node_mask
        else:
            spatial_context = torch.zeros_like(features)
            scatter_gain = torch.sigmoid(self.scatter_gain_logit)
            trust = image.new_zeros((image.shape[0], len(history)))
            uncertainty = image.new_zeros((image.shape[0], len(history)))
            node_mask_out = torch.zeros((image.shape[0], len(history)), dtype=torch.bool, device=image.device)

        logits = self.head(features + spatial_context)
        return {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "trust": trust,
            "uncertainty": uncertainty,
            "node_mask": node_mask_out,
            "spatial_context": spatial_context,
            "scatter_gain": scatter_gain,
            "num_interactions": torch.tensor([len(history)], device=image.device),
        }

    @torch.no_grad()
    def predict(self, image: Tensor, history: History, *, interaction_mode: InteractionMode = "tcig") -> Tensor:
        was_training = self.training
        self.eval()
        output = self.forward(image, history, interaction_mode=interaction_mode)
        if was_training:
            self.train()
        return output["prob"]

    def config_dict(self) -> Dict[str, object]:
        return asdict(self.config)


def build_prompt_maps(
    histories: Sequence[History],
    *,
    output_size: Sequence[int],
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if len(output_size) != 3:
        raise ValueError("TopCoW prompt maps require 3D output_size")
    if sigma <= 0:
        raise ValueError("prompt sigma must be positive")

    shape = tuple(int(v) for v in output_size)
    maps = torch.zeros((len(histories), 2, *shape), dtype=dtype, device=device)
    for batch_index, history in enumerate(histories):
        for unit in history:
            channel = 0 if unit.polarity == 1 else 1
            kernel = _gaussian_kernel(unit.center, shape=shape, sigma=sigma, device=device, dtype=dtype)
            maps[batch_index, channel] = torch.maximum(maps[batch_index, channel], kernel)
    return maps.clamp_(0.0, 1.0)


def _normalise_histories(histories: HistoryBatch, *, batch_size: int) -> List[List[InteractionUnit]]:
    history_list = list(histories)
    if batch_size == 1 and (not history_list or isinstance(history_list[0], InteractionUnit)):
        return [list(history_list)]  # type: ignore[list-item]
    if len(history_list) != batch_size:
        raise ValueError("histories must contain one history per batch item")
    return [list(history) for history in history_list]  # type: ignore[arg-type]


def _gaussian_kernel(
    center: Sequence[float],
    *,
    shape: Tuple[int, int, int],
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if len(center) != 3:
        raise ValueError("TopCoW interactions must use (z, y, x) coordinates")
    axes = [torch.arange(size, dtype=torch.float32, device=device) for size in shape]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    cz, cy, cx = (float(v) for v in center)
    dist2 = (zz - cz).pow(2) + (yy - cy).pow(2) + (xx - cx).pow(2)
    return torch.exp(-dist2 / (2.0 * float(sigma) * float(sigma))).to(dtype=dtype)
