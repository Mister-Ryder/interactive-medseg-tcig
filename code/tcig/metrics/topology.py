"""Topology and connectivity metrics for vessel-like binary masks."""

from __future__ import annotations

from typing import Dict, Tuple
import warnings

import numpy as np
warnings.filterwarnings("ignore", message="A NumPy version")
from scipy import ndimage
from skimage.morphology import skeletonize_3d


def binary_dice_np(prediction, target, *, smooth: float = 1e-7) -> float:
    pred = _binary(prediction)
    gt = _binary(target)
    intersection = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    return float((2.0 * intersection + smooth) / (denom + smooth))


def binary_cldice(prediction, target) -> float:
    """Compute centerline Dice for binary 3D masks.

    This follows the official TopCoW-style implementation: skeletonize both
    masks, measure prediction skeleton covered by GT and GT skeleton covered by
    prediction, then take the harmonic mean.
    """
    pred = _binary(prediction)
    gt = _binary(target)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    pred_skel = skeletonize_3d(pred.astype(np.uint8)).astype(bool)
    gt_skel = skeletonize_3d(gt.astype(np.uint8)).astype(bool)
    if pred_skel.sum() == 0 or gt_skel.sum() == 0:
        return 0.0

    tprec = float((pred_skel & gt).sum()) / float(pred_skel.sum())
    tsens = float((gt_skel & pred).sum()) / float(gt_skel.sum())
    if tprec + tsens <= 0.0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def betti0_error(prediction, target, *, connectivity: int = 1) -> int:
    pred_count = connected_component_count(prediction, connectivity=connectivity)
    gt_count = connected_component_count(target, connectivity=connectivity)
    return int(abs(pred_count - gt_count))


def connected_component_count(mask, *, connectivity: int = 1) -> int:
    _, count = ndimage.label(_binary(mask).astype(np.int32), structure=_structure(mask, connectivity))
    return int(count)


def component_detection_stats(prediction, target, *, connectivity: int = 1) -> Dict[str, float]:
    """Return overlap-based component precision/recall/F1.

    A component is counted as detected if it overlaps at least one component in
    the other mask. This is intentionally simple and stable for sparse vessel
    masks; it complements beta-0 error by exposing over- and under-detection.
    """
    pred = _binary(prediction)
    gt = _binary(target)
    structure = _structure(pred, connectivity)
    pred_labels, pred_count = ndimage.label(pred.astype(np.int32), structure=structure)
    gt_labels, gt_count = ndimage.label(gt.astype(np.int32), structure=structure)

    if pred_count == 0 and gt_count == 0:
        return {
            "component_precision": 1.0,
            "component_recall": 1.0,
            "component_f1": 1.0,
            "matched_pred_components": 0,
            "matched_gt_components": 0,
        }
    if pred_count == 0 or gt_count == 0:
        return {
            "component_precision": 0.0,
            "component_recall": 0.0,
            "component_f1": 0.0,
            "matched_pred_components": 0,
            "matched_gt_components": 0,
        }

    overlap = (pred_labels > 0) & (gt_labels > 0)
    matched_pred = int(len(np.unique(pred_labels[overlap])))
    matched_gt = int(len(np.unique(gt_labels[overlap])))
    precision = float(matched_pred / pred_count) if pred_count else 0.0
    recall = float(matched_gt / gt_count) if gt_count else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall > 0.0 else 0.0
    return {
        "component_precision": precision,
        "component_recall": recall,
        "component_f1": f1,
        "matched_pred_components": matched_pred,
        "matched_gt_components": matched_gt,
    }


def topology_metrics(prediction, target, *, connectivity: int = 1) -> Dict[str, float]:
    pred = _binary(prediction)
    gt = _binary(target)
    pred_components = connected_component_count(pred, connectivity=connectivity)
    gt_components = connected_component_count(gt, connectivity=connectivity)
    stats = component_detection_stats(pred, gt, connectivity=connectivity)
    return {
        "dice": binary_dice_np(pred, gt),
        "cldice": binary_cldice(pred, gt),
        "b0_error": int(abs(pred_components - gt_components)),
        "pred_components": pred_components,
        "gt_components": gt_components,
        **stats,
    }


def _binary(mask) -> np.ndarray:
    return np.asarray(mask) > 0


def _structure(mask, connectivity: int):
    ndim = int(np.asarray(mask).ndim)
    return ndimage.generate_binary_structure(ndim, connectivity)
