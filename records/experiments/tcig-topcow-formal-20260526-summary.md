# TCIG TopCoW Formal Run Summary

Date reviewed: 2026-06-01

## Migrated Local Paths

- Data root now used locally: `data/TopCoW_processed`
- Manifest: `data/manifests/topcow_manifest.jsonl`
- Saved run outputs: `outputs/tcig_topcow_v1_formal_20260526`
- Saved TensorBoard events: `tf-logs/tcig_topcow_v1_formal_20260526`

The original training summary records the old remote data path
`/root/autodl-tmp/topcow_data/TopCoW_processed`. For post-migration evaluation
and inference, pass the current path explicitly with `--data-root
data/TopCoW_processed`.

## Training Setup

- Entry point: `code/scripts/train.py`
- Steps: 20,000
- Max clicks during training/eval: 20
- Patch size: `[32, 64, 64]`
- Train cases: full train split from the 125-case manifest
- Validation cases: 25
- Checkpoints/evals: every 1,000 steps
- Model: `TCIGSegmenter`, embed dim 32, base channels 8

## Training Trend

- First loss: `1.2182308435440063`
- Last loss: `0.07535271346569061`
- First 100-step mean loss: `0.660632`
- Last 100-step mean loss: `0.063467`
- First 100-step mean final-round Dice: `0.499438`
- Last 100-step mean final-round Dice: `0.959302`
- Last 100-step mean trust: `0.213672`
- Last 100-step mean trust std: `0.009875`
- Last 100-step mean scatter gain: `0.138286`

Interpretation: the TCIG training loop converged strongly on the cropped TopCoW
validation protocol. EPICA trust no longer appears completely flat, but its
dynamic range is still modest and should be validated with faithfulness metrics
before making attribution claims.

## Validation Checkpoints

Best saved eval among the 1,000-step checkpoints:

- Best mean final Dice: step 19,000, `0.940416955947876`
- Best `noc@0.85`: step 19,000, `0.48`
- Best `noc@0.90`: step 19,000, `1.0`
- Failure rate at step 19,000: `0.0`

Last checkpoint eval:

- Step 20,000 mean final Dice: `0.9372389650344849`
- Step 20,000 `noc@0.85`: `0.92`
- Step 20,000 `noc@0.90`: `1.72`
- Step 20,000 failure rate: `0.0`

Recommendation: use `checkpoint_step_19000.pt` for current qualitative
inspection and short follow-up evaluations; keep `checkpoint_last.pt` as the
completed-run endpoint.

## Visual Inference Smoke

Command used after migration:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /home/admin/miniconda3/envs/cu122py39
PYTHONPATH=code python code/scripts/infer.py \
  --manifest data/manifests/topcow_manifest.jsonl \
  --data-root data/TopCoW_processed \
  --checkpoint outputs/tcig_topcow_v1_formal_20260526/checkpoint_step_19000.pt \
  --output-dir outputs/tcig_topcow_v1_formal_20260526/inference_step_19000_val2 \
  --split val \
  --max-cases 2 \
  --max-clicks 8 \
  --device cuda
```

Result on two validation cases:

- Mean final Dice: `0.9602000713348389`
- Mean Dice curve: `[0.916224, 0.948325, 0.950541, 0.940315, 0.949175, 0.958049, 0.955911, 0.960666, 0.960200]`
- `noc@0.85`: `0.0`, coverage `1.0`
- `noc@0.90`: `0.5`, coverage `1.0`

Artifacts:

- `outputs/tcig_topcow_v1_formal_20260526/inference_step_19000_val2/inference_summary.json`
- Representative validation montages under the inference output directory (not included in this public snapshot).

This confirms the saved checkpoint can run against the migrated local dataset
path and produce inspectable prediction overlays, not only TensorBoard curves.
