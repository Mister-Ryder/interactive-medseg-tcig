"""Interactive TopCoW evaluation with connectivity/topology metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from tcig.data import TopCoWVolumeDataset
from tcig.metrics.topology import topology_metrics
from tcig.clicks import ClickSimulatorConfig, LargestErrorClickSimulator
from tcig.model import TCIGSegmenter, TCIGSegmenterConfig
from tcig.checkpoint import load_checkpoint


@dataclass
class ConnectivityEvalConfig:
    manifest: str
    data_root: str
    checkpoint: str
    output_dir: str
    split: str = "val"
    val_fraction: float = 0.2
    max_cases: int = 0
    max_clicks: int = 20
    exclude_history_radius: int = 2
    dice_thresholds: Tuple[float, ...] = (0.85, 0.9)
    cldice_thresholds: Tuple[float, ...] = (0.85, 0.9)
    patch_size: Optional[Tuple[int, int, int]] = (32, 64, 64)
    seed: int = 7
    device: str = "cuda"
    embed_dim: int = 32
    base_channels: int = 8
    prompt_sigma: float = 5.0
    scatter_init_sigma: float = 4.0
    method: str = "tcig"


def main() -> int:
    warnings.filterwarnings("ignore", message="A NumPy version")
    args = parse_args()
    config = ConnectivityEvalConfig(
        manifest=args.manifest,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        val_fraction=args.val_fraction,
        max_cases=args.max_cases,
        max_clicks=args.max_clicks,
        exclude_history_radius=args.exclude_history_radius,
        dice_thresholds=tuple(args.dice_thresholds),
        cldice_thresholds=tuple(args.cldice_thresholds),
        patch_size=None if args.full_volume else tuple(args.patch_size),
        seed=args.seed,
        device=args.device,
        embed_dim=args.embed_dim,
        base_channels=args.base_channels,
        prompt_sigma=args.prompt_sigma,
        scatter_init_sigma=args.scatter_init_sigma,
        method=args.method,
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
        max_cases=config.max_cases,
        deterministic=True,
    )
    simulator = LargestErrorClickSimulator(
        ClickSimulatorConfig(threshold=0.5, exclude_history_radius=config.exclude_history_radius)
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    cases: List[Dict[str, object]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        case_rows, case_summary = _evaluate_case(
            model=model,
            item=item,
            simulator=simulator,
            device=device,
            max_clicks=config.max_clicks,
            dice_thresholds=config.dice_thresholds,
            cldice_thresholds=config.cldice_thresholds,
            method=config.method,
        )
        rows.extend(case_rows)
        cases.append(case_summary)
        print(
            f"{case_summary['case_id']}: final Dice={case_summary['final_dice']:.4f} "
            f"clDice={case_summary['final_cldice']:.4f} b0={case_summary['final_b0_error']}",
            flush=True,
        )

    summary = _summarise(rows, cases, config)
    payload = {
        "ok": True,
        "config": asdict(config),
        "model_config": model.config_dict(),
        "checkpoint_step": checkpoint_payload.get("step"),
        "summary": summary,
        "cases": cases,
    }
    _write_json(output_dir / "connectivity_summary.json", payload)
    _write_csv(output_dir / "per_case_round_metrics.csv", rows)
    _write_markdown(output_dir / "connectivity_report.md", payload)
    _plot_curves(output_dir / "connectivity_curves.png", summary)
    _plot_case_heatmap(output_dir / "case_final_metrics.png", cases)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved connectivity evaluation to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TCIG with interactive connectivity metrics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--exclude-history-radius", type=int, default=2)
    parser.add_argument("--dice-thresholds", type=float, nargs="+", default=(0.85, 0.9))
    parser.add_argument("--cldice-thresholds", type=float, nargs="+", default=(0.85, 0.9))
    parser.add_argument("--patch-size", type=int, nargs=3, default=(32, 64, 64))
    parser.add_argument("--full-volume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--prompt-sigma", type=float, default=5.0)
    parser.add_argument("--scatter-init-sigma", type=float, default=4.0)
    parser.add_argument(
        "--method",
        choices=("tcig", "prompt_only", "no_interaction"),
        default="tcig",
        help=(
            "Evaluation method: full TCIG, prompt-only clicks without graph "
            "context, or no-interaction zero-click baseline."
        ),
    )
    return parser.parse_args()


def _evaluate_case(
    *,
    model: TCIGSegmenter,
    item: Dict[str, object],
    simulator: LargestErrorClickSimulator,
    device: torch.device,
    max_clicks: int,
    dice_thresholds: Sequence[float],
    cldice_thresholds: Sequence[float],
    method: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    case_id = str(item["case_id"])
    image = item["image"].to(device=device, dtype=torch.float32)
    target = item["label"][0].detach().cpu().numpy() > 0
    image_b = image.unsqueeze(0)
    history = []
    rows: List[Dict[str, object]] = []

    for click_count in range(max_clicks + 1):
        with torch.no_grad():
            output = model(image_b, history, interaction_mode=method)
        prob = output["prob"][0, 0].detach().cpu()
        pred = prob.numpy() >= 0.5
        metrics = topology_metrics(pred, target)
        rows.append(
            {
                "case_id": case_id,
                "clicks": click_count,
                "num_history": len(history),
                **metrics,
            }
        )
        if click_count == max_clicks:
            break
        click = simulator.next_click(prob, torch.from_numpy(target), round_index=click_count, history=history)
        if click is None:
            break
        history.append(click)

    dice_curve = [float(row["dice"]) for row in rows]
    cldice_curve = [float(row["cldice"]) for row in rows]
    b0_curve = [int(row["b0_error"]) for row in rows]
    final = rows[-1]
    return rows, {
        "case_id": case_id,
        "num_rounds": len(rows),
        "num_clicks": len(rows) - 1,
        "final_dice": float(final["dice"]),
        "final_cldice": float(final["cldice"]),
        "final_b0_error": int(final["b0_error"]),
        "final_component_f1": float(final["component_f1"]),
        "dice_curve": dice_curve,
        "cldice_curve": cldice_curve,
        "b0_curve": b0_curve,
        "dice_noc": {str(t): _first_at_least(dice_curve, t) for t in dice_thresholds},
        "cldice_noc": {str(t): _first_at_least(cldice_curve, t) for t in cldice_thresholds},
        "b0_exact_noc": _first_at_most(b0_curve, 0),
    }


def _summarise(rows: Sequence[Dict[str, object]], cases: Sequence[Dict[str, object]], config: ConnectivityEvalConfig) -> Dict[str, object]:
    mean_by_click = []
    for click_count in range(config.max_clicks + 1):
        click_rows = [row for row in rows if int(row["clicks"]) == click_count]
        if not click_rows:
            mean_by_click.append({"clicks": click_count, "coverage": 0.0})
            continue
        mean_by_click.append(
            {
                "clicks": click_count,
                "coverage": len(click_rows) / len(cases),
                "dice": _mean(click_rows, "dice"),
                "cldice": _mean(click_rows, "cldice"),
                "b0_error": _mean(click_rows, "b0_error"),
                "pred_components": _mean(click_rows, "pred_components"),
                "gt_components": _mean(click_rows, "gt_components"),
                "component_precision": _mean(click_rows, "component_precision"),
                "component_recall": _mean(click_rows, "component_recall"),
                "component_f1": _mean(click_rows, "component_f1"),
            }
        )
    final_cases = list(cases)
    summary: Dict[str, object] = {
        "num_cases": len(cases),
        "max_clicks": config.max_clicks,
        "mean_final_dice": float(np.mean([case["final_dice"] for case in final_cases])) if final_cases else 0.0,
        "std_final_dice": float(np.std([case["final_dice"] for case in final_cases])) if final_cases else 0.0,
        "mean_final_cldice": float(np.mean([case["final_cldice"] for case in final_cases])) if final_cases else 0.0,
        "std_final_cldice": float(np.std([case["final_cldice"] for case in final_cases])) if final_cases else 0.0,
        "mean_final_b0_error": float(np.mean([case["final_b0_error"] for case in final_cases])) if final_cases else 0.0,
        "std_final_b0_error": float(np.std([case["final_b0_error"] for case in final_cases])) if final_cases else 0.0,
        "mean_final_component_f1": float(np.mean([case["final_component_f1"] for case in final_cases])) if final_cases else 0.0,
        "mean_by_click": mean_by_click,
    }
    for threshold in config.dice_thresholds:
        values = [case["dice_noc"][str(threshold)] for case in final_cases]
        summary[f"dice_noc@{threshold}"] = _mean_optional_noc(values, config.max_clicks)
        summary[f"dice_coverage@{threshold}"] = _coverage(values)
    for threshold in config.cldice_thresholds:
        values = [case["cldice_noc"][str(threshold)] for case in final_cases]
        summary[f"cldice_noc@{threshold}"] = _mean_optional_noc(values, config.max_clicks)
        summary[f"cldice_coverage@{threshold}"] = _coverage(values)
    b0_values = [case["b0_exact_noc"] for case in final_cases]
    summary["b0_exact_noc"] = _mean_optional_noc(b0_values, config.max_clicks)
    summary["b0_exact_coverage"] = _coverage(b0_values)
    return summary


def _plot_curves(path: Path, summary: Dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["mean_by_click"]
    clicks = [row["clicks"] for row in rows if "dice" in row]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(clicks, [row["dice"] for row in rows if "dice" in row], marker="o")
    axes[0, 0].set_title("Dice vs clicks")
    axes[0, 1].plot(clicks, [row["cldice"] for row in rows if "cldice" in row], marker="o", color="#2f9e44")
    axes[0, 1].set_title("clDice vs clicks")
    axes[1, 0].plot(clicks, [row["b0_error"] for row in rows if "b0_error" in row], marker="o", color="#d9480f")
    axes[1, 0].set_title("Betti-0 error vs clicks")
    axes[1, 1].plot(clicks, [row["component_f1"] for row in rows if "component_f1" in row], marker="o", color="#6741d9")
    axes[1, 1].set_title("Component F1 vs clicks")
    for ax in axes.flat:
        ax.set_xlabel("clicks")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_case_heatmap(path: Path, cases: Sequence[Dict[str, object]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not cases:
        return
    names = [case["case_id"] for case in cases]
    score_values = np.asarray(
        [
            [case["final_dice"], case["final_cldice"], case["final_component_f1"]]
            for case in cases
        ],
        dtype=np.float32,
    )
    b0_values = [case["final_b0_error"] for case in cases]
    fig, (ax_scores, ax_b0) = plt.subplots(
        1,
        2,
        figsize=(12, max(5, 0.28 * len(cases))),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    im = ax_scores.imshow(score_values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_scores.set_yticks(range(len(names)))
    ax_scores.set_yticklabels(names, fontsize=7)
    ax_scores.set_xticks(range(3))
    ax_scores.set_xticklabels(["Dice", "clDice", "CompF1"])
    ax_scores.set_title("Final overlap/connectivity scores")
    fig.colorbar(im, ax=ax_scores, fraction=0.04, pad=0.02)

    ax_b0.barh(range(len(names)), b0_values, color="#d9480f")
    ax_b0.set_yticks(range(len(names)))
    ax_b0.set_yticklabels([])
    ax_b0.invert_yaxis()
    ax_b0.set_xlabel("B0 error")
    ax_b0.set_title("Final beta-0 error")
    ax_b0.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown(path: Path, payload: Dict[str, object]) -> None:
    summary = payload["summary"]
    config = payload.get("config", {})
    method = config.get("method", "tcig") if isinstance(config, dict) else "tcig"
    lines = [
        "# TCIG TopCoW Connectivity Evaluation",
        "",
        f"- checkpoint step: `{payload.get('checkpoint_step')}`",
        f"- method: `{method}`",
        f"- cases: `{summary['num_cases']}`",
        f"- max clicks: `{summary['max_clicks']}`",
        f"- mean final Dice: `{summary['mean_final_dice']:.4f} +/- {summary['std_final_dice']:.4f}`",
        f"- mean final clDice: `{summary['mean_final_cldice']:.4f} +/- {summary['std_final_cldice']:.4f}`",
        f"- mean final beta-0 error: `{summary['mean_final_b0_error']:.2f} +/- {summary['std_final_b0_error']:.2f}`",
        f"- mean final component F1: `{summary['mean_final_component_f1']:.4f}`",
        "",
        "## NoC",
        "",
    ]
    for key, value in summary.items():
        if "noc@" in key or key == "b0_exact_noc":
            coverage_key = key.replace("noc", "coverage") if key != "b0_exact_noc" else "b0_exact_coverage"
            lines.append(f"- `{key}`: `{value:.3f}`; coverage `{summary.get(coverage_key, 0):.3f}`")
    lines.extend(["", "## Mean Curves", "", "| clicks | Dice | clDice | beta-0 error | component F1 |", "|---:|---:|---:|---:|---:|"])
    for row in summary["mean_by_click"]:
        if "dice" not in row:
            continue
        lines.append(
            f"| {row['clicks']} | {row['dice']:.4f} | {row['cldice']:.4f} | "
            f"{row['b0_error']:.2f} | {row['component_f1']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _first_at_least(values: Sequence[float], threshold: float) -> Optional[int]:
    for index, value in enumerate(values):
        if float(value) >= float(threshold):
            return index
    return None


def _first_at_most(values: Sequence[float], threshold: float) -> Optional[int]:
    for index, value in enumerate(values):
        if float(value) <= float(threshold):
            return index
    return None


def _mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _mean_optional_noc(values: Sequence[Optional[int]], max_clicks: int) -> float:
    if not values:
        return float(max_clicks + 1)
    penalty = float(max_clicks + 1)
    return float(np.mean([penalty if value is None else float(value) for value in values]))


def _coverage(values: Sequence[Optional[int]]) -> float:
    return float(sum(value is not None for value in values) / len(values)) if values else 0.0


def _validate_config(config: ConnectivityEvalConfig) -> None:
    if config.max_cases < 0:
        raise ValueError("--max-cases must be non-negative")
    if config.max_clicks < 1:
        raise ValueError("--max-clicks must be at least 1")
    if config.exclude_history_radius < 0:
        raise ValueError("--exclude-history-radius must be non-negative")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    for value in config.dice_thresholds + config.cldice_thresholds:
        if value <= 0 or value > 1:
            raise ValueError("thresholds must be in (0, 1]")
    if config.patch_size is not None and (len(config.patch_size) != 3 or any(v < 2 for v in config.patch_size)):
        raise ValueError("--patch-size must contain three values >= 2")
    if config.method not in ("tcig", "prompt_only", "no_interaction"):
        raise ValueError("--method must be tcig, prompt_only, or no_interaction")


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
