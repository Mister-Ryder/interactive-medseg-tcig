"""Anisotropic spatial scatter for IIRE prompt-feature injection.

This module implements the spatially-aware scatter described in the TCIG
research plan section 4.3.3:

    A_l(v) = sum_i  tau_i * polarity_i * K_i(v) * Wproj(h_i_out)

where:
    h_i_out      = node feature i after the IIRE graph transformer
    tau_i        = EPICA trust score for node i
    polarity_i   = +1 for foreground / -1 for background prompts
    K_i(v)       = anisotropic Gaussian kernel centred at the node's
                   voxel coordinate
    Wproj        = linear projection from embed_dim to out_channels

The kernel is anisotropic in the sense that each spatial axis has its
own learnable log-sigma, so the model can favour elongated scattering
along the slowest-varying direction (useful for tubular structures
where future iterations will replace the Euclidean distance with a
soft-skeleton geodesic). A single global anisotropy is shared across
nodes to keep the parameter count tiny; per-node anisotropy can
be enabled by toggling ``per_node_sigma=True``.

The output is a dense prompt feature map ready to be added to a
backbone decoder feature at the matching spatial scale. For a 3D run
with a (D, H, W) decoder feature shape, the scatter is computed at the
quarter-resolution grid by default and bilinearly upsampled to the
target shape for efficiency.

Public API
----------
- ``ScatterConfig``: dataclass with all hyperparameters.
- ``AnisotropicScatter``: ``nn.Module`` whose ``forward`` produces the
  scattered prompt-feature map.

Integration
-----------
The TCIG segmenter uses scatter roughly like this::

    scatter = AnisotropicScatter(ScatterConfig(embed_dim=32, out_channels=8, n_dims=3))
    ...
    prompt_feature = scatter(
        node_features=updated,           # [B, N, D] from IIRE
        node_coords=batch["node_coords"], # [B, N, n_dims] voxel coords
        node_trust=post["trust"],         # [B, N]
        node_polarities=batch["node_polarities"],  # [B, N]
        node_mask=graph_batch.node_mask,           # [B, N]
        output_size=image.shape[2:],               # (D, H, W)
    )
    features = encoder(image_with_prompts) + prompt_feature

The prompt feature is per-voxel and respects spatial locality, which is
what the TCIG paper claim ("IIRE benefit > backbone-only by routing
interaction influence through anatomical neighbourhoods") actually
requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ScatterConfig:
    """Hyperparameters for :class:`AnisotropicScatter`.

    Attributes
    ----------
    embed_dim:
        Width of node features ``h_i_out`` coming out of IIRE.
    out_channels:
        Channels of the produced prompt feature map.
    n_dims:
        Spatial dimensionality. 2 for retinal fundus, 3 for CoW/coronary.
    init_sigma:
        Initial scattering radius in voxels. Sigma is parameterised via
        log-space for positivity and unconstrained optimisation.
    sigma_min:
        Numerical floor on sigma after the exp; prevents degeneracy.
    sigma_max:
        Soft cap on sigma; prevents the kernel from becoming uniform.
    per_node_sigma:
        If True, predict a per-node log-sigma residual from the node
        feature so that different clicks can have different scattering
        radii. Off by default to keep parameter count tiny.
    work_resolution_factor:
        Compute the scatter at ``output_size / factor`` and upsample.
        Default 4 trades a small spatial-precision loss for substantial
        memory and compute savings on 3D volumes.
    upsample_mode:
        ``trilinear`` for 3D scatter, ``bilinear`` for 2D. ``nearest``
        falls back when align_corners cannot be guaranteed for the
        target backbone.
    use_polarity_sign:
        If True, multiply each node's kernel by ``polarity_i``. This
        causes background clicks to subtract from the prompt feature,
        mirroring the additive/subtractive convention of disk prompts.
    eps:
        Small constant added to denominators for numerical safety.
    """

    embed_dim: int
    out_channels: int
    n_dims: int = 3
    init_sigma: float = 4.0
    sigma_min: float = 0.5
    sigma_max: float = 32.0
    per_node_sigma: bool = False
    work_resolution_factor: int = 4
    upsample_mode: str = "trilinear"
    use_polarity_sign: bool = True
    eps: float = 1e-6


class AnisotropicScatter(nn.Module):
    """Spatially-aware scatter of IIRE node features.

    This module is intentionally backbone-agnostic. It consumes the
    graph transformer's ``[B, N, D]`` node features plus voxel
    coordinates and EPICA trust, and emits a dense ``[B, out_channels,
    *output_size]`` prompt feature map. Callers add or concatenate this
    map into their decoder/encoder features at the matching scale.

    Implementation notes
    --------------------
    - The scatter kernel ``K_i(v)`` is implemented as a separable
      anisotropic Gaussian. For each spatial axis ``a`` the squared
      distance is normalised by ``sigma_a^2`` independently, so a
      single ``sigma`` parameter per axis controls anisotropy.
    - Because N (number of interaction nodes) is bounded (<= 64 by
      design in TCIG), the dense ``[B, N, *output_size]`` kernel tensor
      fits comfortably in GPU memory at quarter resolution. We compute
      it explicitly rather than via gather/scatter ops so the gradient
      pathway is simple and CUDA-friendly.
    - Invalid nodes (``node_mask == False``) are zeroed before the
      weighted sum, which preserves the right gradient flow without
      requiring index gathers.
    """

    def __init__(self, config: ScatterConfig) -> None:
        super().__init__()
        if config.n_dims not in (2, 3):
            raise ValueError("n_dims must be 2 or 3 for IIRE scatter")
        if config.embed_dim < 1 or config.out_channels < 1:
            raise ValueError("embed_dim and out_channels must be positive")
        if config.work_resolution_factor < 1:
            raise ValueError("work_resolution_factor must be at least 1")
        if config.upsample_mode not in ("trilinear", "bilinear", "nearest"):
            raise ValueError(
                "upsample_mode must be 'trilinear', 'bilinear', or 'nearest'"
            )

        self.config = config
        # log_sigma is one scalar per spatial axis, initialised so that
        # exp(log_sigma) == init_sigma.
        init_log_sigma = torch.log(
            torch.full((config.n_dims,), float(config.init_sigma))
        )
        self.log_sigma = nn.Parameter(init_log_sigma)

        if config.per_node_sigma:
            # Predict a single log-sigma residual per node, broadcast
            # across all spatial axes.
            self.node_log_sigma = nn.Linear(config.embed_dim, 1)
        else:
            self.node_log_sigma = None

        self.proj = nn.Linear(config.embed_dim, config.out_channels)

    def forward(
        self,
        node_features: Tensor,
        node_coords: Tensor,
        node_trust: Tensor,
        output_size: Sequence[int],
        node_mask: Optional[Tensor] = None,
        node_polarities: Optional[Tensor] = None,
    ) -> Tensor:
        """Scatter graph nodes into a dense prompt feature map.

        Parameters
        ----------
        node_features:
            ``[B, N, embed_dim]`` graph transformer outputs.
        node_coords:
            ``[B, N, n_dims]`` voxel coordinates of each node. The
            coordinates are interpreted in the ``output_size`` voxel
            frame; if upstream code uses normalised coords, multiply by
            ``output_size`` before calling.
        node_trust:
            ``[B, N]`` EPICA trust score in ``[0, 1]``.
        output_size:
            Target spatial shape ``(D, H, W)`` for 3D or ``(H, W)`` for
            2D, in voxels.
        node_mask:
            Optional ``[B, N]`` boolean mask. True means valid node.
        node_polarities:
            Optional ``[B, N]`` of +/-1. Only used when
            ``config.use_polarity_sign`` is True.

        Returns
        -------
        Tensor
            ``[B, out_channels, *output_size]`` prompt feature map.
        """
        cfg = self.config
        if len(output_size) != cfg.n_dims:
            raise ValueError(
                f"output_size {tuple(output_size)} mismatches n_dims={cfg.n_dims}"
            )

        batch, num_nodes, embed_dim = node_features.shape
        if embed_dim != cfg.embed_dim:
            raise ValueError(
                f"node_features embed_dim {embed_dim} does not match "
                f"ScatterConfig.embed_dim {cfg.embed_dim}"
            )
        if node_coords.shape != (batch, num_nodes, cfg.n_dims):
            raise ValueError(
                f"node_coords shape {tuple(node_coords.shape)} expected "
                f"({batch}, {num_nodes}, {cfg.n_dims})"
            )
        if node_trust.shape != (batch, num_nodes):
            raise ValueError(
                f"node_trust shape {tuple(node_trust.shape)} expected "
                f"({batch}, {num_nodes})"
            )

        device = node_features.device
        dtype = node_features.dtype
        # Choose the working resolution.
        work_size = tuple(max(1, int(s) // cfg.work_resolution_factor) for s in output_size)
        scale = [
            float(w_size) / float(o_size) for w_size, o_size in zip(work_size, output_size)
        ]
        scale_t = torch.tensor(scale, dtype=dtype, device=device)
        coords_work = node_coords.to(dtype=dtype) * scale_t  # [B, N, n_dims]

        # Build the grid of voxel coordinates at the working resolution.
        # grid shape: [n_dims, *work_size]
        ranges = [torch.arange(s, device=device, dtype=dtype) for s in work_size]
        mesh = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack(mesh, dim=0)  # [n_dims, *work_size]

        # Compute squared per-axis distances. We use broadcasting:
        # node_coords[B, N, n_dims, 1, 1, (1)]
        # grid[n_dims, *work_size]
        # diff[B, N, n_dims, *work_size]
        expand_dims = (1,) * cfg.n_dims
        node_coords_b = coords_work.view(
            batch, num_nodes, cfg.n_dims, *expand_dims
        )
        grid_b = grid.view(1, 1, cfg.n_dims, *work_size)
        diff = node_coords_b - grid_b

        # Anisotropic sigma per axis, optionally per node.
        log_sigma_axes = self.log_sigma  # [n_dims]
        if self.node_log_sigma is not None:
            node_log_residual = self.node_log_sigma(node_features).squeeze(-1)
            log_sigma_per_node = (
                log_sigma_axes.view(1, 1, cfg.n_dims)
                + node_log_residual.unsqueeze(-1)
            )
        else:
            log_sigma_per_node = log_sigma_axes.view(1, 1, cfg.n_dims).expand(
                batch, num_nodes, cfg.n_dims
            )

        sigma = (
            torch.exp(log_sigma_per_node)
            .clamp(min=cfg.sigma_min, max=cfg.sigma_max)
            .view(batch, num_nodes, cfg.n_dims, *expand_dims)
        )

        squared = (diff / (sigma + cfg.eps)).pow(2)
        # Sum over the n_dims axis to get a scalar distance per voxel
        # per node. The axis index after view is exactly 2.
        squared_sum = squared.sum(dim=2)  # [B, N, *work_size]
        kernel = torch.exp(-0.5 * squared_sum)

        # Apply polarity if requested.
        if cfg.use_polarity_sign and node_polarities is not None:
            sign = node_polarities.to(dtype=dtype).view(
                batch, num_nodes, *expand_dims
            )
            kernel = kernel * sign

        # Apply trust weighting.
        kernel = kernel * node_trust.to(dtype=dtype).view(
            batch, num_nodes, *expand_dims
        )

        if node_mask is not None:
            mask = node_mask.to(dtype=dtype).view(batch, num_nodes, *expand_dims)
            kernel = kernel * mask

        # Project node features to out_channels then weighted-sum across nodes.
        projected = self.proj(node_features)  # [B, N, C_out]
        projected = projected.permute(0, 2, 1).contiguous()  # [B, C_out, N]

        # We want, for each (b, c, *v):
        #   out[b, c, v] = sum_n  projected[b, c, n] * kernel[b, n, v]
        # Reshape kernel to [B, N, prod_work].
        prod_work = 1
        for s in work_size:
            prod_work *= s
        kernel_flat = kernel.view(batch, num_nodes, prod_work)  # [B, N, V]
        out_flat = torch.bmm(projected, kernel_flat)  # [B, C_out, V]
        out_work = out_flat.view(batch, cfg.out_channels, *work_size)

        # Upsample to the requested output size if necessary.
        if tuple(work_size) == tuple(output_size):
            return out_work

        if cfg.n_dims == 3:
            mode = "trilinear" if cfg.upsample_mode != "nearest" else "nearest"
        else:
            mode = "bilinear" if cfg.upsample_mode != "nearest" else "nearest"

        align_corners = False if mode != "nearest" else None
        out = F.interpolate(out_work, size=tuple(output_size), mode=mode, align_corners=align_corners)
        return out


def build_default_scatter(
    embed_dim: int,
    out_channels: int,
    *,
    n_dims: int = 3,
    init_sigma: float = 4.0,
) -> AnisotropicScatter:
    """Convenience factory for standard TCIG scatter."""
    return AnisotropicScatter(
        ScatterConfig(
            embed_dim=embed_dim,
            out_channels=out_channels,
            n_dims=n_dims,
            init_sigma=init_sigma,
        )
    )


def node_coords_from_units(
    unit_sequences: Sequence[Sequence],
    *,
    n_dims: int,
    pad_value: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pack a list of variable-length InteractionUnit sequences into tensors.

    Parameters
    ----------
    unit_sequences:
        ``[batch][num_units_b]`` lists of InteractionUnit-like objects
        with a ``.center`` tuple, a ``.polarity`` int and ``.ndim``.
    n_dims:
        Expected spatial dimensionality.

    Returns
    -------
    Tuple of tensors ``(coords, polarities, mask)`` shaped
    ``[B, N_max, n_dims]``, ``[B, N_max]`` and ``[B, N_max]`` (bool).
    """
    batch = len(unit_sequences)
    if batch == 0:
        raise ValueError("unit_sequences must contain at least one batch element")
    max_nodes = max(1, max(len(seq) for seq in unit_sequences))
    coords = torch.full((batch, max_nodes, n_dims), pad_value, dtype=dtype, device=device)
    polarities = torch.zeros((batch, max_nodes), dtype=dtype, device=device)
    mask = torch.zeros((batch, max_nodes), dtype=torch.bool, device=device)

    for b, seq in enumerate(unit_sequences):
        for n, unit in enumerate(seq):
            if unit.ndim != n_dims:
                raise ValueError(
                    f"unit at batch {b} index {n} has ndim={unit.ndim} but "
                    f"scatter expects n_dims={n_dims}"
                )
            center: Tuple[float, ...] = tuple(float(v) for v in unit.center)
            coords[b, n] = torch.tensor(center, dtype=dtype, device=device)
            polarities[b, n] = float(unit.polarity)
            mask[b, n] = True

    return coords, polarities, mask
