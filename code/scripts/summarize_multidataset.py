"""Summarise multi-dataset TCIG connectivity evaluations.

The script consumes output directories created by ``scripts/eval.py`` and
writes paper-facing tables, figures, and a Chinese analysis report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


SUMMARY_KEYS = [
    ("Dice", "mean_final_dice", "std_final_dice", True),
    ("clDice", "mean_final_cldice", "std_final_cldice", True),
    ("B0 error", "mean_final_b0_error", "std_final_b0_error", False),
    ("Comp. F1", "mean_final_component_f1", None, True),
    ("Dice NoC@0.90", "dice_noc@0.9", None, False),
    ("clDice NoC@0.90", "cldice_noc@0.9", None, False),
    ("B0 exact NoC", "b0_exact_noc", None, False),
]


def main() -> int:
    args = parse_args()
    runs = _load_runs(args.eval)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_summary_csv(output_dir / "multidataset_summary.csv", runs)
    _write_case_csv(output_dir / "multidataset_cases.csv", runs)
    _write_table_tex(output_dir / "multidataset_topology_table.tex", runs, args.caption, args.label)
    curves_path = output_dir / "multidataset_connectivity_curves.png"
    bars_path = output_dir / "multidataset_final_metrics.png"
    _plot_curves(curves_path, runs)
    _plot_final_bars(bars_path, runs)
    _write_figures_tex(output_dir / "multidataset_figures.tex", curves_path.name, bars_path.name)
    _write_report(output_dir / "multidataset_topology_report_zh.md", runs, curves_path.name, bars_path.name)
    print(f"saved multi-dataset summary to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TeX tables, figures, and Chinese report from TCIG eval runs.")
    parser.add_argument(
        "--eval",
        action="append",
        nargs=2,
        metavar=("DATASET", "DIR"),
        required=True,
        help="Dataset name and the corresponding scripts/eval.py output directory.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--caption",
        default="Connectivity-aware interactive evaluation on external datasets.",
    )
    parser.add_argument("--label", default="tab:tcig_multidataset_topology")
    return parser.parse_args()


def _load_runs(entries: Sequence[Tuple[str, str]]) -> List[Dict[str, object]]:
    runs = []
    for dataset, directory in entries:
        root = Path(directory)
        summary_path = root / "connectivity_summary.json"
        case_csv = root / "per_case_round_metrics.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing summary: {summary_path}")
        if not case_csv.is_file():
            raise FileNotFoundError(f"missing per-case CSV: {case_csv}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        method = config.get("method", "tcig") if isinstance(config, dict) else "tcig"
        runs.append(
            {
                "dataset": dataset,
                "method": method,
                "directory": root,
                "payload": payload,
                "summary": payload["summary"],
                "cases": payload.get("cases", []),
                "rows": _read_csv_rows(case_csv),
            }
        )
    return runs


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                if key == "case_id":
                    parsed[key] = value
                elif key in ("clicks", "num_history"):
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _write_summary_csv(path: Path, runs: Sequence[Dict[str, object]]) -> None:
    fields = [
        "dataset",
        "method",
        "num_cases",
        "max_clicks",
        "mean_final_dice",
        "std_final_dice",
        "mean_final_cldice",
        "std_final_cldice",
        "mean_final_b0_error",
        "std_final_b0_error",
        "mean_final_component_f1",
        "dice_noc@0.9",
        "dice_coverage@0.9",
        "cldice_noc@0.9",
        "cldice_coverage@0.9",
        "b0_exact_noc",
        "b0_exact_coverage",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            summary = run["summary"]
            writer.writerow(
                {
                    field: run[field] if field in ("dataset", "method") else summary.get(field, "")
                    for field in fields
                }
            )


def _write_case_csv(path: Path, runs: Sequence[Dict[str, object]]) -> None:
    fields = [
        "dataset",
        "method",
        "case_id",
        "num_clicks",
        "final_dice",
        "final_cldice",
        "final_b0_error",
        "final_component_f1",
        "b0_exact_noc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for case in run["cases"]:
                writer.writerow(
                    {
                        "dataset": run["dataset"],
                        "method": run["method"],
                        "case_id": case["case_id"],
                        "num_clicks": case["num_clicks"],
                        "final_dice": case["final_dice"],
                        "final_cldice": case["final_cldice"],
                        "final_b0_error": case["final_b0_error"],
                        "final_component_f1": case["final_component_f1"],
                        "b0_exact_noc": case.get("b0_exact_noc"),
                    }
                )


def _write_table_tex(path: Path, runs: Sequence[Dict[str, object]], caption: str, label: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Dataset & Method & Dice $\\uparrow$ & clDice $\\uparrow$ & B0 err. $\\downarrow$ & Comp. F1 $\\uparrow$ & Dice NoC@0.90 $\\downarrow$ & clDice NoC@0.90 $\\downarrow$ & B0 NoC $\\downarrow$ & B0 cov. $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for run in runs:
        summary = run["summary"]
        lines.append(
            " & ".join(
                [
                    _tex_escape(str(run["dataset"])),
                    _tex_escape(_method_label(str(run["method"]))),
                    _mean_std(summary, "mean_final_dice", "std_final_dice"),
                    _mean_std(summary, "mean_final_cldice", "std_final_cldice"),
                    _mean_std(summary, "mean_final_b0_error", "std_final_b0_error"),
                    _fmt(summary.get("mean_final_component_f1")),
                    _fmt(summary.get("dice_noc@0.9")),
                    _fmt(summary.get("cldice_noc@0.9")),
                    _fmt(summary.get("b0_exact_noc")),
                    _fmt(summary.get("b0_exact_coverage")),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{_tex_escape(caption)}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figures_tex(path: Path, curves_name: str, bars_name: str) -> None:
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.98\\linewidth]{{{curves_name}}}",
        "\\caption{Interactive topology-sensitive metric curves across datasets.}",
        "\\label{fig:tcig_multidataset_curves}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.98\\linewidth]{{{bars_name}}}",
        "\\caption{Final overlap and connectivity metrics after the interaction budget.}",
        "\\label{fig:tcig_multidataset_final_metrics}",
        "\\end{figure}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_curves(path: Path, runs: Sequence[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("dice", "Dice", axes[0, 0]),
        ("cldice", "clDice", axes[0, 1]),
        ("b0_error", "Betti-0 error", axes[1, 0]),
        ("component_f1", "Component F1", axes[1, 1]),
    ]
    for metric, title, ax in specs:
        for run in runs:
            rows = [row for row in run["summary"]["mean_by_click"] if metric in row]
            ax.plot(
                [row["clicks"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                label=f"{run['dataset']} / {_method_label(str(run['method']))}",
            )
        ax.set_title(title)
        ax.set_xlabel("Clicks")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Score")
    axes[0, 1].set_ylabel("Score")
    axes[1, 0].set_ylabel("Error")
    axes[1, 1].set_ylabel("Score")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_final_bars(path: Path, runs: Sequence[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [f"{run['dataset']}\n{_method_label(str(run['method']))}" for run in runs]
    x = np.arange(len(names))
    width = 0.22
    dice = [run["summary"]["mean_final_dice"] for run in runs]
    cldice = [run["summary"]["mean_final_cldice"] for run in runs]
    comp = [run["summary"]["mean_final_component_f1"] for run in runs]
    b0 = [run["summary"]["mean_final_b0_error"] for run in runs]

    fig, (ax_scores, ax_b0) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1.2]})
    ax_scores.bar(x - width, dice, width, label="Dice")
    ax_scores.bar(x, cldice, width, label="clDice")
    ax_scores.bar(x + width, comp, width, label="Comp. F1")
    ax_scores.set_xticks(x)
    ax_scores.set_xticklabels(names)
    ax_scores.set_ylim(0.0, 1.0)
    ax_scores.grid(True, axis="y", alpha=0.3)
    ax_scores.legend()

    ax_b0.bar(names, b0, color="#d9480f")
    ax_b0.set_ylabel("B0 error")
    ax_b0.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_report(path: Path, runs: Sequence[Dict[str, object]], curves_name: str, bars_name: str) -> None:
    lines = [
        "# 多数据集拓扑交互式评估报告",
        "",
        "## 方法",
        "",
        "本报告使用当前 TCIG checkpoint 在 IXI、Lausanne 和 CROWN 三个外部数据集上进行交互式评估。评估协议采用 largest-error click simulator：每一轮根据当前预测与二值标注之间的最大错误连通区域生成前景或背景点击，然后重新前向推理。每个病例记录 0-click 到最大点击预算内的完整曲线。",
        "",
        "实验设置为每个数据集使用全部 20 个本地病例，最大交互预算为 20 次点击。checkpoint、manifest、data root、method 等配置保存在各数据集子目录的 `connectivity_summary.json` 中。",
        "",
        "对照方法包括：`tcig` 为完整方法；`prompt_only` 使用同样的点击高斯 prompt，但禁用 IIRE/EPICA/空间散射图上下文；`no_interaction` 完全忽略点击，等价于同一模型的 zero-click 非交互预测。这样可以分别观察交互 prompt 的作用和图式交互建模的增益。",
        "",
        "拓扑相关指标包括 Dice、clDice、Betti-0 误差、预测/标注连通分量数、连通分量 precision/recall/F1，以及达到 Dice 或 clDice 阈值所需点击数 NoC。Betti-0 exact NoC 表示第一次达到连通分量数量完全匹配所需的点击数；若未达到，则记为最大点击数加一并在 coverage 中反映失败比例。",
        "",
        "## 结果文件",
        "",
        f"- 跨数据集曲线图：`{curves_name}`",
        f"- 最终指标柱状图：`{bars_name}`",
        "- TeX 表格：`multidataset_topology_table.tex`",
        "- TeX 图片片段：`multidataset_figures.tex`",
        "- 汇总 CSV：`multidataset_summary.csv`",
        "- 病例级 CSV：`multidataset_cases.csv`",
        "",
        "## 汇总结果",
        "",
        "| 数据集 | 方法 | 病例数 | Dice | clDice | B0误差 | 分量F1 | Dice NoC@0.90 | clDice NoC@0.90 | B0 exact NoC | B0 exact coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        summary = run["summary"]
        lines.append(
            "| {dataset} | {method} | {num_cases} | {dice} | {cldice} | {b0} | {comp} | {dice_noc} | {cldice_noc} | {b0_noc} | {b0_cov} |".format(
                dataset=run["dataset"],
                method=_method_label(str(run["method"])),
                num_cases=summary["num_cases"],
                dice=_mean_std_md(summary, "mean_final_dice", "std_final_dice"),
                cldice=_mean_std_md(summary, "mean_final_cldice", "std_final_cldice"),
                b0=_mean_std_md(summary, "mean_final_b0_error", "std_final_b0_error"),
                comp=_fmt(summary.get("mean_final_component_f1")),
                dice_noc=_fmt(summary.get("dice_noc@0.9")),
                cldice_noc=_fmt(summary.get("cldice_noc@0.9")),
                b0_noc=_fmt(summary.get("b0_exact_noc")),
                b0_cov=_fmt(summary.get("b0_exact_coverage")),
            )
        )
    lines.extend(["", "## 结果分析", ""])
    lines.extend(_analysis_paragraphs(runs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _analysis_paragraphs(runs: Sequence[Dict[str, object]]) -> List[str]:
    if not runs:
        return ["未发现可分析的评估结果。"]
    best_dice = max(runs, key=lambda run: run["summary"]["mean_final_dice"])
    best_cldice = max(runs, key=lambda run: run["summary"]["mean_final_cldice"])
    worst_b0 = max(runs, key=lambda run: run["summary"]["mean_final_b0_error"])
    best_b0_cov = max(runs, key=lambda run: run["summary"].get("b0_exact_coverage", 0.0))
    tcig_vs_prompt = _method_deltas(runs, "tcig", "prompt_only")
    prompt_vs_no = _method_deltas(runs, "prompt_only", "no_interaction")
    paragraphs = [
        f"从重叠精度看，`{best_dice['dataset']} / {_method_label(str(best_dice['method']))}` 的最终 Dice 最高。clDice 最高的设置为 `{best_cldice['dataset']} / {_method_label(str(best_cldice['method']))}`，它更直接反映中心线和管状结构一致性，因此可作为血管拓扑质量的主要观察点。",
        "",
        f"从连通性看，`{worst_b0['dataset']} / {_method_label(str(worst_b0['method']))}` 的平均 Betti-0 误差最高，说明该设置下仍存在较明显的断裂、粘连或分量数量偏差。`{best_b0_cov['dataset']} / {_method_label(str(best_b0_cov['method']))}` 的 B0 exact coverage 最高，表示交互点击更容易把连通分量数量校正到与标注一致。",
        "",
    ]
    if tcig_vs_prompt:
        paragraphs.extend(
            [
                _delta_sentence("TCIG 相对 prompt-only", tcig_vs_prompt),
                "",
            ]
        )
    if prompt_vs_no:
        paragraphs.extend(
            [
                _delta_sentence("prompt-only 相对 no-interaction", prompt_vs_no),
                "",
            ]
        )
    paragraphs.extend(
        [
            "需要注意的是，Dice/clDice 的 NoC 与 B0 exact NoC 衡量的是不同目标：前者关注达到重叠或中心线质量阈值所需点击数，后者关注连通分量数量是否完全匹配。若 Dice 较高但 B0 exact coverage 较低，说明模型能恢复大部分血管体素，但仍可能保留局部断裂或错误连接；这正是拓扑交互式指标相对单纯 Dice 更有价值的地方。",
            "",
            "总体上，当前评估框架已经能同时保存病例级曲线、数据集级统计、论文表格和跨数据集图。后续如果加入真实交互式 UI，可复用相同指标协议，只需要把 largest-error simulator 生成的点击替换为真实用户点击记录。",
        ]
    )
    return paragraphs


def _method_deltas(runs: Sequence[Dict[str, object]], method: str, baseline: str) -> List[Dict[str, float]]:
    by_key = {(run["dataset"], run["method"]): run for run in runs}
    deltas = []
    for dataset in sorted({run["dataset"] for run in runs}):
        current = by_key.get((dataset, method))
        base = by_key.get((dataset, baseline))
        if current is None or base is None:
            continue
        current_summary = current["summary"]
        base_summary = base["summary"]
        deltas.append(
            {
                "dataset": str(dataset),
                "dice": float(current_summary["mean_final_dice"]) - float(base_summary["mean_final_dice"]),
                "cldice": float(current_summary["mean_final_cldice"]) - float(base_summary["mean_final_cldice"]),
                "b0": float(base_summary["mean_final_b0_error"]) - float(current_summary["mean_final_b0_error"]),
                "comp_f1": float(current_summary["mean_final_component_f1"]) - float(base_summary["mean_final_component_f1"]),
            }
        )
    return deltas


def _delta_sentence(title: str, deltas: Sequence[Dict[str, float]]) -> str:
    mean_dice = sum(row["dice"] for row in deltas) / len(deltas)
    mean_cldice = sum(row["cldice"] for row in deltas) / len(deltas)
    mean_b0 = sum(row["b0"] for row in deltas) / len(deltas)
    mean_comp = sum(row["comp_f1"] for row in deltas) / len(deltas)
    return (
        f"{title} 的平均变化为：Dice {mean_dice:+.3f}，clDice {mean_cldice:+.3f}，"
        f"B0 误差下降 {mean_b0:+.3f}，分量 F1 {mean_comp:+.3f}。"
    )


def _mean_std(summary: Dict[str, object], mean_key: str, std_key: str) -> str:
    mean = _fmt(summary.get(mean_key))
    std = _fmt(summary.get(std_key))
    return f"{mean} $\\pm$ {std}"


def _mean_std_md(summary: Dict[str, object], mean_key: str, std_key: str) -> str:
    mean = _fmt(summary.get(mean_key))
    std = _fmt(summary.get(std_key))
    return f"{mean} ± {std}"


def _fmt(value: object) -> str:
    if value is None or value == "":
        return "--"
    return f"{float(value):.3f}"


def _tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def _method_label(method: str) -> str:
    return {
        "tcig": "TCIG",
        "prompt_only": "Prompt-only",
        "no_interaction": "No-interaction",
    }.get(method, method)


if __name__ == "__main__":
    sys.exit(main())
