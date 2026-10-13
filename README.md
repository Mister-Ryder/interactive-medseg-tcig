# Interactive MedSeg TCIG

Public research snapshot of topology-sensitive interactive medical image segmentation on Circle of Willis (CoW) data. The active implementation combines prediction-driven click simulation, interaction-history encoding (IIRE), per-interaction trust estimates (EPICA), anisotropic feature scatter, and connectivity-aware evaluation.

## Contents

| Path | Purpose |
| --- | --- |
| `code/tcig/` | TCIG model, interaction, graph, training, and evaluation modules. |
| `code/scripts/` | Manifest creation, training, evaluation, inference, and summary entry points. See [`code/scripts/README.md`](code/scripts/README.md). |
| `code/tests/` | Unit and pipeline tests. |
| `records/experiments/` | Selected formal TopCoW and aggregate multidataset results. |
| [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) | SHA-256 hashes of files in this snapshot. |

## Data and reproducibility

The public snapshot excludes processed medical images and labels, case-level tables, dataset manifests containing case identifiers, checkpoints, predictions, logs, and internal working records. Build a local manifest from datasets you are authorized to use with `code/scripts/build_manifest.py`; the run commands are documented in `code/scripts/README.md`. Aggregate results and plots are provided under `records/experiments/`.

This repository is a selected snapshot of the current TCIG code and published aggregate results. It does not include the full dataset or a complete training environment. No project-wide reuse license is declared.

---

本仓库公开 TCIG 在 CoW 数据上的算法代码和汇总实验结果。原始医学影像、标签、逐病例表、病例标识清单、模型权重及内部工作记录未公开。运行前请按 `code/scripts/README.md` 在有使用权限的数据上重新生成 manifest 并配置环境。
