"""Numerically stable binary Dice used by NoC and attribution metrics."""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor


def binary_dice(
    prediction: Union[Tensor, "ArrayLike"],
    target: Union[Tensor, "ArrayLike"],
    *,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """Return the binary Dice between a probability/binary prediction and target.

    Parameters
    ----------
    prediction:
        Either a probability map in ``[0, 1]`` (then ``threshold`` is
        applied) or an already-binarised mask.
    target:
        Reference binary mask. Anything > 0 is treated as foreground.
    threshold:
        Used only when ``prediction`` is floating point.
    smooth:
        Numerical smoothing constant added to numerator and denominator
        to make Dice well-defined for empty predictions and targets.

    Returns
    -------
    float
        Dice score in ``[0, 1]``.
    """
    pred_t = torch.as_tensor(prediction)
    tgt_t = torch.as_tensor(target)

    if pred_t.is_floating_point():
        pred_bin = (pred_t >= float(threshold)).to(dtype=torch.float32)
    else:
        pred_bin = (pred_t > 0).to(dtype=torch.float32)

    tgt_bin = (tgt_t > 0).to(dtype=torch.float32)

    if pred_bin.shape != tgt_bin.shape:
        raise ValueError(
            f"prediction shape {tuple(pred_bin.shape)} != target shape "
            f"{tuple(tgt_bin.shape)}"
        )

    intersection = (pred_bin * tgt_bin).sum()
    denom = pred_bin.sum() + tgt_bin.sum()
    return float((2.0 * intersection + smooth) / (denom + smooth))


# Sentinel for type hints used by docstrings; not imported at runtime.
ArrayLike = Tensor
