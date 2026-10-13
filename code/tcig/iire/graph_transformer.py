"""Small graph transformer used by IIRE.

This is a clean-room TCIG component.  It implements generic relation-biased
message passing over interaction-unit nodes; it does not copy or depend on SCC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class GraphTransformerConfig:
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    edge_dim: int = 4
    use_trust_bias: bool = True


class GraphTransformerLayer(nn.Module):
    """Relation-biased self-attention over interaction nodes."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        edge_dim: int = 4,
        dropout: float = 0.0,
        use_trust_bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_trust_bias = use_trust_bias

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        nodes: Tensor,
        edge_features: Tensor,
        node_trust: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Update node features.

        Parameters
        ----------
        nodes:
            Tensor shaped ``[batch, num_nodes, embed_dim]``.
        edge_features:
            Tensor shaped ``[batch, num_nodes, num_nodes, edge_dim]``. The four
            default channels correspond to spatial/topological, temporal,
            polarity, and affinity relations.
        node_trust:
            Optional tensor shaped ``[batch, num_nodes]`` from EPICA. It biases
            messages from low-trust source nodes downward.
        node_mask:
            Optional boolean tensor shaped ``[batch, num_nodes]`` where true
            marks valid nodes.
        """
        residual = nodes
        x = self.norm1(nodes)
        batch, num_nodes, _ = x.shape

        q = self.q_proj(x).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        edge_bias = self.edge_proj(edge_features).permute(0, 3, 1, 2)
        logits = logits + edge_bias

        if node_trust is not None and self.use_trust_bias:
            source_bias = torch.clamp(node_trust, min=1e-4).log().unsqueeze(1).unsqueeze(2)
            logits = logits + source_bias

        if node_mask is not None:
            key_mask = (~node_mask.bool()).unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(key_mask, torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        updated = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, num_nodes, self.embed_dim)
        updated = self.out_proj(updated)
        x = residual + self.dropout(updated)
        x = x + self.dropout(self.ffn(self.norm2(x)))

        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class GraphTransformer(nn.Module):
    """Stacked IIRE graph transformer."""

    def __init__(self, config: GraphTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    edge_dim=config.edge_dim,
                    dropout=config.dropout,
                    use_trust_bias=config.use_trust_bias,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(
        self,
        nodes: Tensor,
        edge_features: Tensor,
        node_trust: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = nodes
        for layer in self.layers:
            x = layer(x, edge_features, node_trust=node_trust, node_mask=node_mask)
        return x
