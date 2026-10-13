# Code

Current implementation of TCIG, a Trust-Calibrated Interaction Graph for
topology-sensitive interactive medical image segmentation.

## Active Pipeline

```text
tcig/                 Core research package: data loader, interaction protocol,
                      TCIGSegmenter, IIRE, EPICA, training loop, evaluation
                      metrics, and checkpoint helpers.
scripts/              Stable command-line entrypoints for manifest, train,
                      eval, and infer.
tests/                Compact regression tests grouped by capability.
```

The repository root `data/` directory is reserved for dataset artifacts and
manifests. Dataset loading code lives in `code/tcig/data.py`, so code modules
and actual medical data no longer share the same namespace.

## Entrypoints

- `scripts/build_manifest.py`
- `scripts/train.py`
- `scripts/eval.py`
- `scripts/infer.py`

`scripts/eval.py` is the current paper-facing evaluation entrypoint. It reports
Dice, clDice, beta-0 error, component counts, component F1, and interaction
NoC-style topology recovery metrics.

## Research Modules

- `tcig/iire/`: multi-relational graph encoding over interaction units.
- `tcig/epica.py`: evidential per-interaction trust estimates.
- `tcig/iire/scatter.py`: trust-weighted graph features injected into the 3D decoder
  feature map.
- `tcig/train_loop.py`: prediction-driven multi-round training logic used by
  `scripts/train.py`.
- `tcig/evaluate.py` and `tcig/metrics/`: NoC, attribution, and topology /
  connectivity metrics used by `scripts/eval.py`.

## Development Notes

- Keep raw medical data, processed datasets, checkpoints, predictions, and
  TensorBoard event files outside Git.
- Prefer extending the current `TCIGSegmenter` and `scripts/train.py` /
  `scripts/eval.py` flow over adding parallel experimental entrypoints.
- New experimental variants should be exposed as arguments or small config
  options on the current entrypoints unless they require a genuinely different
  execution contract.
