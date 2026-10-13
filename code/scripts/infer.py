"""Run TCIG checkpoint inference and export visual inspection artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch

from tcig.data import TopCoWVolumeDataset
from tcig.metrics.dice import binary_dice
from tcig.clicks import ClickSimulatorConfig, LargestErrorClickSimulator
from tcig.model import TCIGSegmenter, TCIGSegmenterConfig
from tcig.checkpoint import load_checkpoint


@dataclass
class InferConfig:
    manifest: str
    data_root: str
    checkpoint: str
    output_dir: str
    split: str = "val"
    val_fraction: float = 0.2
    max_cases: int = 4
    case_ids: Optional[Tuple[str, ...]] = None
    max_clicks: int = 8
    exclude_history_radius: int = 2
    thresholds: Tuple[float, ...] = (0.85, 0.9)
    patch_size: Optional[Tuple[int, int, int]] = (32, 64, 64)
    seed: int = 7
    device: str = "cuda"
    save_npz: bool = True
    embed_dim: int = 32
    base_channels: int = 8
    prompt_sigma: float = 5.0
    scatter_init_sigma: float = 4.0


def main() -> int:
    args = parse_args()
    config = InferConfig(
        manifest=args.manifest,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        val_fraction=args.val_fraction,
        max_cases=args.max_cases,
        case_ids=tuple(args.case_ids) if args.case_ids else None,
        max_clicks=args.max_clicks,
        exclude_history_radius=args.exclude_history_radius,
        thresholds=tuple(args.thresholds),
        patch_size=None if args.full_volume else tuple(args.patch_size),
        seed=args.seed,
        device=args.device,
        save_npz=not args.no_npz,
        embed_dim=args.embed_dim,
        base_channels=args.base_channels,
        prompt_sigma=args.prompt_sigma,
        scatter_init_sigma=args.scatter_init_sigma,
    )
    _validate_config(config)

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model_config = _model_config_from_checkpoint(
        Path(config.checkpoint),
        fallback=TCIGSegmenterConfig(
            embed_dim=config.embed_dim,
            base_channels=config.base_channels,
            prompt_sigma=config.prompt_sigma,
            scatter_init_sigma=config.scatter_init_sigma,
        ),
    )
    model = TCIGSegmenter(model_config).to(device)
    checkpoint_payload = load_checkpoint(Path(config.checkpoint), model=model, map_location=device)
    model.eval()

    dataset = TopCoWVolumeDataset(
        Path(config.manifest),
        Path(config.data_root),
        patch_size=config.patch_size,
        split=config.split,
        val_fraction=config.val_fraction,
        seed=config.seed,
        max_cases=0,
        deterministic=True,
    )
    indices = _select_indices(dataset, case_ids=config.case_ids, max_cases=config.max_cases)
    simulator = LargestErrorClickSimulator(
        ClickSimulatorConfig(threshold=0.5, exclude_history_radius=config.exclude_history_radius)
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for index in indices:
        row = _run_case(
            model=model,
            item=dataset[index],
            simulator=simulator,
            device=device,
            max_clicks=config.max_clicks,
            thresholds=config.thresholds,
            output_dir=output_dir,
            save_npz=config.save_npz,
        )
        cases.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    summary = _summarise_cases(cases, thresholds=config.thresholds, max_clicks=config.max_clicks)
    payload = {
        "ok": True,
        "config": asdict(config),
        "model_config": model.config_dict(),
        "checkpoint_step": checkpoint_payload.get("step"),
        "summary": summary,
        "cases": cases,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved inference artifacts to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TCIG predictions for visual inspection.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--max-clicks", type=int, default=8)
    parser.add_argument("--exclude-history-radius", type=int, default=2)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.85, 0.9))
    parser.add_argument("--patch-size", type=int, nargs=3, default=(32, 64, 64))
    parser.add_argument("--full-volume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-npz", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--prompt-sigma", type=float, default=5.0)
    parser.add_argument("--scatter-init-sigma", type=float, default=4.0)
    return parser.parse_args()


def _run_case(
    *,
    model: TCIGSegmenter,
    item: Dict[str, object],
    simulator: LargestErrorClickSimulator,
    device: torch.device,
    max_clicks: int,
    thresholds: Sequence[float],
    output_dir: Path,
    save_npz: bool,
) -> Dict[str, object]:
    case_id = str(item["case_id"])
    image = item["image"].to(device=device, dtype=torch.float32)  # [1, D, H, W]
    target = item["label"][0].detach().cpu()
    image_b = image.unsqueeze(0)
    history = []
    probabilities: List[np.ndarray] = []
    dice_curve: List[float] = []
    trust_curve: List[List[float]] = []
    uncertainty_curve: List[List[float]] = []

    for round_index in range(max_clicks + 1):
        with torch.no_grad():
            output = model(image_b, history)
        prob = output["prob"][0, 0].detach().cpu()
        probabilities.append(prob.numpy().astype(np.float32))
        dice_curve.append(binary_dice(prob, target, threshold=0.5))
        trust_curve.append(_tensor_row(output["trust"]))
        uncertainty_curve.append(_tensor_row(output["uncertainty"]))
        if round_index == max_clicks:
            break
        click = simulator.next_click(prob, target, round_index=round_index, history=history)
        if click is None:
            break
        history.append(click)

    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image_np = image[0].detach().cpu().numpy()
    target_np = target.numpy().astype(np.float32)
    final_prob = probabilities[-1]
    final_pred = final_prob >= 0.5
    slice_index = _choose_slice(target_np, final_prob)
    _save_montage(
        case_dir / "montage.png",
        image=image_np,
        target=target_np,
        probabilities=probabilities,
        history=history,
        slice_index=slice_index,
    )
    _save_overlay(
        case_dir / "final_overlay.png",
        image=image_np[slice_index],
        target=target_np[slice_index] > 0,
        prediction=final_pred[slice_index],
        clicks=[unit.center for unit in history if int(round(unit.center[0])) == slice_index],
    )
    if save_npz:
        np.savez_compressed(
            case_dir / "prediction.npz",
            image=image_np,
            label=target_np.astype(np.uint8),
            final_probability=final_prob,
            final_prediction=final_pred.astype(np.uint8),
            dice_curve=np.asarray(dice_curve, dtype=np.float32),
            click_points=np.asarray([unit.center for unit in history], dtype=np.float32),
            click_polarities=np.asarray([unit.polarity for unit in history], dtype=np.int8),
            origin=np.asarray(item["origin"], dtype=np.int64),
        )
    threshold_noc = {
        str(threshold): _first_reached(dice_curve, threshold)
        for threshold in thresholds
    }
    return {
        "case_id": case_id,
        "origin": [int(v) for v in item["origin"].tolist()],
        "slice_index": int(slice_index),
        "num_clicks": len(history),
        "dice_curve": dice_curve,
        "final_dice": dice_curve[-1],
        "threshold_noc": threshold_noc,
        "clicks": [unit.to_dict() for unit in history],
        "trust_curve": trust_curve,
        "uncertainty_curve": uncertainty_curve,
        "artifacts": {
            "montage": str(case_dir / "montage.png"),
            "final_overlay": str(case_dir / "final_overlay.png"),
            "npz": str(case_dir / "prediction.npz") if save_npz else None,
        },
    }


def _save_montage(
    path: Path,
    *,
    image: np.ndarray,
    target: np.ndarray,
    probabilities: Sequence[np.ndarray],
    history: Sequence[object],
    slice_index: int,
) -> None:
    rounds = sorted(set([0, len(probabilities) - 1] + [idx for idx in (1, 2, 3, 5, 8, 12, 20) if idx < len(probabilities)]))
    panels = []
    for round_index in rounds:
        clicks = [unit.center for unit in history[:round_index] if int(round(unit.center[0])) == slice_index]
        panel = _make_overlay_panel(
            image=image[slice_index],
            target=target[slice_index] > 0,
            prediction=probabilities[round_index][slice_index] >= 0.5,
            clicks=clicks,
            title=f"clicks={round_index}",
        )
        panels.append(panel)
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(path)


def _save_overlay(path: Path, *, image: np.ndarray, target: np.ndarray, prediction: np.ndarray, clicks) -> None:
    _make_overlay_panel(image=image, target=target, prediction=prediction, clicks=clicks, title="final").save(path)


def _make_overlay_panel(
    *,
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    clicks,
    title: str,
) -> Image.Image:
    base = _normalise_uint8(image)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    # Green = target, red = prediction, yellow = overlap.
    rgb[target, 1] = np.maximum(rgb[target, 1], 210)
    rgb[prediction, 0] = np.maximum(rgb[prediction, 0], 230)
    overlap = target & prediction
    rgb[overlap, 0] = 235
    rgb[overlap, 1] = 220
    rgb[overlap, 2] = 35
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").resize((256, 256))
    canvas = Image.new("RGB", (256, 286), color=(255, 255, 255))
    canvas.paste(img, (0, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    scale_y = 256.0 / max(1, image.shape[0])
    scale_x = 256.0 / max(1, image.shape[1])
    for point in clicks:
        _, y, x = point
        cx = int(round(float(x) * scale_x))
        cy = 30 + int(round(float(y) * scale_y))
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=(0, 180, 255), width=2)
    return canvas


def _normalise_uint8(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        hi = float(arr.max()) if float(arr.max()) > lo else lo + 1.0
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _choose_slice(target: np.ndarray, probability: np.ndarray) -> int:
    target_area = target.reshape(target.shape[0], -1).sum(axis=1)
    pred_area = probability.reshape(probability.shape[0], -1).sum(axis=1)
    score = target_area + 0.25 * pred_area
    return int(np.argmax(score))


def _tensor_row(values: torch.Tensor) -> List[float]:
    if values.numel() == 0:
        return []
    return [float(v) for v in values[0].detach().cpu().tolist()]


def _first_reached(dice_curve: Sequence[float], threshold: float) -> Optional[int]:
    for index, value in enumerate(dice_curve):
        if value >= threshold:
            return index
    return None


def _summarise_cases(cases: Sequence[Dict[str, object]], *, thresholds: Sequence[float], max_clicks: int) -> Dict[str, object]:
    if not cases:
        return {"num_cases": 0, "mean_final_dice": 0.0}
    mean_curve = []
    for index in range(max_clicks + 1):
        values = [case["dice_curve"][index] for case in cases if len(case["dice_curve"]) > index]
        mean_curve.append(sum(values) / len(values) if values else None)
    summary: Dict[str, object] = {
        "num_cases": len(cases),
        "mean_final_dice": sum(float(case["final_dice"]) for case in cases) / len(cases),
        "mean_dice_curve": mean_curve,
    }
    for threshold in thresholds:
        values = [case["threshold_noc"][str(threshold)] for case in cases]
        reached = [value for value in values if value is not None]
        summary[f"noc@{threshold}"] = sum(reached) / len(reached) if reached else float(max_clicks) + 1.0
        summary[f"coverage@{threshold}"] = len(reached) / len(cases)
    return summary


def _select_indices(dataset: TopCoWVolumeDataset, *, case_ids: Optional[Sequence[str]], max_cases: int) -> List[int]:
    if case_ids:
        wanted = set(case_ids)
        indices = [idx for idx, row in enumerate(dataset.records) if str(row.get("case_id")) in wanted]
        missing = sorted(wanted - {str(dataset.records[idx].get("case_id")) for idx in indices})
        if missing:
            raise ValueError(f"case ids not found in selected split: {missing}")
        return indices
    limit = len(dataset) if max_cases <= 0 else min(max_cases, len(dataset))
    return list(range(limit))


def _validate_config(config: InferConfig) -> None:
    if config.max_cases < 0:
        raise ValueError("--max-cases must be non-negative")
    if config.max_clicks < 1:
        raise ValueError("--max-clicks must be at least 1")
    if config.exclude_history_radius < 0:
        raise ValueError("--exclude-history-radius must be non-negative")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    if any(value <= 0 or value > 1 for value in config.thresholds):
        raise ValueError("--thresholds must be in (0, 1]")
    if config.patch_size is not None and (len(config.patch_size) != 3 or any(v < 2 for v in config.patch_size)):
        raise ValueError("--patch-size must contain three values >= 2")


def _model_config_from_checkpoint(path: Path, *, fallback: TCIGSegmenterConfig) -> TCIGSegmenterConfig:
    payload = torch.load(path, map_location="cpu")
    config = payload.get("config", {})
    model_config = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model_config, dict):
        return fallback
    values = fallback.__dict__.copy()
    values.update({key: value for key, value in model_config.items() if key in values})
    return TCIGSegmenterConfig(**values)


if __name__ == "__main__":
    sys.exit(main())
