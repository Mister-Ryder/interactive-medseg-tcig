# Scripts

Current project entrypoints are intentionally minimal:

- `build_manifest.py`: build a JSONL manifest from processed TopCoW-style arrays.
- `train.py`: train the current TCIG interactive segmentation model.
- `eval.py`: run the current connectivity-aware interactive evaluation.
- `infer.py`: export visual prediction artifacts for qualitative inspection.
- `summarize_multidataset.py`: aggregate multiple connectivity evaluations
  into CSV, TeX tables, figures, and a Chinese Markdown report.

Use the migrated local dataset path explicitly. Do not reuse old
`/root/autodl-tmp/...` paths unless that directory exists on the current
server.

## Environment

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /home/admin/miniconda3/envs/cu122py39
export PYTHONPATH=code
```

## Manifest

```bash
python code/scripts/build_manifest.py \
  --root data/TopCoW_processed \
  --output data/manifests/topcow_manifest.jsonl \
  --dataset topcow_mr_processed \
  --relative-to data/TopCoW_processed \
  --image-dir image \
  --label-dir binary_label \
  --multiclass-label-dir multiclass_mask
```

## Train

```bash
python code/scripts/train.py \
  --manifest data/manifests/topcow_manifest.jsonl \
  --data-root data/TopCoW_processed \
  --output-dir outputs/tcig_topcow_formal \
  --device cuda \
  --steps 20000 \
  --max-clicks 20 \
  --loss-rounds all \
  --val-max-cases 25 \
  --eval-every 1000 \
  --save-every 1000 \
  --tensorboard-dir tf-logs/tcig_topcow_formal \
  --tensorboard-flush-secs 5
```

## Connectivity Evaluation

```bash
python code/scripts/eval.py \
  --manifest data/manifests/topcow_manifest.jsonl \
  --data-root data/TopCoW_processed \
  --checkpoint outputs/tcig_topcow_v1_formal_20260526/checkpoint_step_19000.pt \
  --output-dir outputs/tcig_topcow_v1_formal_20260526/connectivity_eval_step_19000_val25 \
  --split val \
  --max-cases 25 \
  --max-clicks 20 \
  --device cuda
```

Use `--method` to run matched baselines under the same click simulator and
metrics:

- `tcig`: full TCIG/IIRE/EPICA interaction graph.
- `prompt_only`: same click prompt maps, without graph context.
- `no_interaction`: zero-click non-interactive prediction repeated across
  rounds.

This writes:

- `connectivity_summary.json`
- `per_case_round_metrics.csv`
- `connectivity_report.md`
- `connectivity_curves.png`
- `case_final_metrics.png`

The protocol records Dice, clDice, beta-0 error, component counts, and
component F1 at every click count.

## Multi-Dataset Topology Summary

```bash
python code/scripts/summarize_multidataset.py \
  --eval IXI outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/ixi \
  --eval IXI outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/ixi_prompt_only \
  --eval IXI outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/ixi_no_interaction \
  --eval Lausanne outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/lausanne \
  --eval Lausanne outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/lausanne_prompt_only \
  --eval Lausanne outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/lausanne_no_interaction \
  --eval CROWN outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/crown \
  --eval CROWN outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/crown_prompt_only \
  --eval CROWN outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/crown_no_interaction \
  --output-dir outputs/tcig_topcow_v1_formal_20260526/multidataset_topology_19000/summary
```

Small paper-facing copies of the generated report, TeX snippets, CSV files,
and figures are archived under
`records/experiments/tcig_multidataset_topology_20260602/`.

## Visual Inference

```bash
python code/scripts/infer.py \
  --manifest data/manifests/topcow_manifest.jsonl \
  --data-root data/TopCoW_processed \
  --checkpoint outputs/tcig_topcow_v1_formal_20260526/checkpoint_step_19000.pt \
  --output-dir outputs/tcig_topcow_v1_formal_20260526/inference_step_19000_val \
  --split val \
  --max-cases 8 \
  --max-clicks 8 \
  --device cuda
```

This writes per-case overlays, montages, prediction arrays, and
`inference_summary.json`.
