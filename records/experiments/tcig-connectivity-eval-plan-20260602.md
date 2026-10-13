# TCIG Connectivity-Aware Interactive Evaluation Plan

Date: 2026-06-02

## Motivation

TopCoW and CoW-style vessel segmentation cannot be judged by overlap alone.
A prediction can have high Dice while still breaking or over-splitting vessel
components. For TCIG, the evaluation must therefore measure whether additional
interactions repair topology/connectivity, not only whether they increase Dice.

## Metric Stack

Primary interactive metrics:

- Dice-vs-clicks curve.
- `Dice-NoC@0.85` and `Dice-NoC@0.90`.
- clDice-vs-clicks curve.
- `clDice-NoC@0.85` and `clDice-NoC@0.90`.
- beta-0 error-vs-clicks curve: absolute difference between predicted and GT
  connected-component counts.
- `B0-exact-NoC`: first click count where beta-0 error reaches 0.

Diagnostic connectivity metrics:

- predicted component count.
- GT component count.
- overlap-based component precision/recall/F1.

The implementation follows the TopCoW official metric style for binary Dice,
3D clDice via `skeletonize_3d`, and beta-0 via connected components.

## Current Formal Evaluation

Command:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /home/admin/miniconda3/envs/cu122py39
PYTHONPATH=code python code/scripts/eval.py \
  --manifest data/manifests/topcow_manifest.jsonl \
  --data-root data/TopCoW_processed \
  --checkpoint outputs/tcig_topcow_v1_formal_20260526/checkpoint_step_19000.pt \
  --output-dir outputs/tcig_topcow_v1_formal_20260526/connectivity_eval_step_19000_val25 \
  --split val \
  --max-cases 25 \
  --max-clicks 20 \
  --device cuda
```

Summary:

- Cases: 25 validation patches.
- Checkpoint: step 19,000.
- Mean final Dice: `0.9666 +/- 0.0124`.
- Mean final clDice: `0.9846 +/- 0.0224`.
- Mean final beta-0 error: `0.48 +/- 0.81`.
- Mean final component F1: `0.9023`.
- `Dice-NoC@0.85`: `0.48`, coverage `1.0`.
- `Dice-NoC@0.90`: `1.00`, coverage `1.0`.
- `clDice-NoC@0.85`: `0.92`, coverage `1.0`.
- `clDice-NoC@0.90`: `1.56`, coverage `1.0`.
- `B0-exact-NoC`: `4.45`, coverage `0.8`.

Interpretation:

The model reaches strong overlap and centerline scores quickly, but exact
component-count recovery is harder. This is useful evidence for the TCIG paper:
connectivity-sensitive metrics expose a correction burden that Dice alone hides.

## Saved Artifacts

- `connectivity_summary.json`: machine-readable summary and per-case curves.
- `per_case_round_metrics.csv`: one row per case per click count.
- `connectivity_report.md`: paper-facing short report.
- `connectivity_curves.png`: mean Dice, clDice, beta-0 error and component F1
  curves.
- `case_final_metrics.png`: per-case final score and beta-0 diagnostics.

## Next Evaluation Additions

- Run the same protocol for disk-prompt and no-IIRE/no-EPICA ablations.
- Add a topology-latency table: clicks required for Dice target versus clicks
  required for beta-0 exactness.
- For qualitative figures, pair this report with `infer.py`
  overlays on predefined high beta-0-error cases.
