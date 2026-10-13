"""Train TCIG on TopCoW with prediction-driven multi-round clicks."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcig.data import TopCoWVolumeDataset, collate_topcow_batch
from tcig.evaluate import evaluate_topcow_noc
from tcig.clicks import ClickSimulatorConfig, LargestErrorClickSimulator
from tcig.model import TCIGSegmenter, TCIGSegmenterConfig
from tcig.train_loop import run_multiround_training_step
from tcig.checkpoint import load_checkpoint, save_checkpoint


@dataclass
class TrainConfig:
    manifest: str
    data_root: str
    output_dir: str
    steps: int = 10000
    max_clicks: int = 5
    exclude_history_radius: int = 2
    loss_rounds: str = "all"
    patch_size: Optional[Tuple[int, int, int]] = (32, 64, 64)
    val_fraction: float = 0.2
    train_max_cases: int = 0
    val_max_cases: int = 16
    eval_every: int = 1000
    save_every: int = 1000
    learning_rate: float = 1e-3
    seed: int = 7
    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 0
    embed_dim: int = 32
    base_channels: int = 8
    prompt_sigma: float = 5.0
    scatter_init_sigma: float = 4.0
    tensorboard_dir: Optional[str] = None
    tensorboard_flush_secs: int = 30
    resume: Optional[str] = None


def main() -> int:
    args = parse_args()
    config = TrainConfig(
        manifest=args.manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        steps=args.steps,
        max_clicks=args.max_clicks,
        exclude_history_radius=args.exclude_history_radius,
        loss_rounds=args.loss_rounds,
        patch_size=None if args.full_volume else tuple(args.patch_size),
        val_fraction=args.val_fraction,
        train_max_cases=args.train_max_cases,
        val_max_cases=args.val_max_cases,
        eval_every=args.eval_every,
        save_every=args.save_every,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embed_dim=args.embed_dim,
        base_channels=args.base_channels,
        prompt_sigma=args.prompt_sigma,
        scatter_init_sigma=args.scatter_init_sigma,
        tensorboard_dir=args.tensorboard_dir,
        tensorboard_flush_secs=args.tensorboard_flush_secs,
        resume=args.resume,
    )
    _validate_config(config)
    _set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TopCoWVolumeDataset(
        Path(config.manifest),
        Path(config.data_root),
        patch_size=config.patch_size,
        split="train",
        val_fraction=config.val_fraction,
        seed=config.seed,
        max_cases=config.train_max_cases,
        deterministic=False,
    )
    val_dataset = TopCoWVolumeDataset(
        Path(config.manifest),
        Path(config.data_root),
        patch_size=config.patch_size,
        split="val",
        val_fraction=config.val_fraction,
        seed=config.seed,
        max_cases=config.val_max_cases,
        deterministic=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_topcow_batch,
    )
    iterator = _cycle(loader)

    model_config = TCIGSegmenterConfig(
        embed_dim=config.embed_dim,
        base_channels=config.base_channels,
        prompt_sigma=config.prompt_sigma,
        scatter_init_sigma=config.scatter_init_sigma,
    )
    model = TCIGSegmenter(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_step = 0
    history: List[Dict[str, object]] = []
    if config.resume:
        payload = load_checkpoint(Path(config.resume), model=model, optimizer=optimizer, map_location=device)
        start_step = int(payload.get("step", 0))
        history = list(payload.get("history", []))

    simulator = LargestErrorClickSimulator(
        ClickSimulatorConfig(
            threshold=0.5,
            exclude_history_radius=config.exclude_history_radius,
        )
    )
    writer = _create_summary_writer(config)
    if writer is not None:
        writer.add_text("config/json", json.dumps(asdict(config), indent=2), 0)
        writer.add_text("model_config/json", json.dumps(model.config_dict(), indent=2), 0)

    model.train()
    try:
        for step in range(start_step + 1, config.steps + 1):
            batch = next(iterator)
            row = run_multiround_training_step(
                model=model,
                batch=batch,
                simulator=simulator,
                optimizer=optimizer,
                device=device,
                max_clicks=config.max_clicks,
                loss_rounds=config.loss_rounds,
            )
            row.update(
                {
                    "step": step,
                    "case_id": batch["case_id"],
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            history.append(row)
            _log_train_row(writer, row, step)
            print(json.dumps(row), flush=True)

            if config.eval_every > 0 and step % config.eval_every == 0:
                eval_summary = evaluate_topcow_noc(
                    model=model,
                    dataset=val_dataset,
                    device=device,
                    thresholds=(0.85, 0.9),
                    max_clicks=config.max_clicks,
                    max_cases=config.val_max_cases,
                    include_faithfulness=False,
                )
                _write_json(output_dir / f"eval_step_{step}.json", eval_summary)
                _log_eval_summary(writer, eval_summary, step)
                model.train()

            if config.save_every > 0 and step % config.save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_step_{step}.pt",
                    model=model,
                    optimizer=optimizer,
                    config=_checkpoint_config(config, model_config),
                    step=step,
                    history=history,
                )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    summary = {
        "ok": True,
        "config": asdict(config),
        "model_config": model.config_dict(),
        "device": str(device),
        "torch_version": torch.__version__,
        "num_train_cases": len(train_dataset),
        "num_val_cases": len(val_dataset),
        "first_loss": history[0]["loss"] if history else None,
        "last_loss": history[-1]["loss"] if history else None,
        "history": history,
    }
    _write_json(output_dir / "train_summary.json", summary)
    save_checkpoint(
        output_dir / "checkpoint_last.pt",
        model=model,
        optimizer=optimizer,
        config=_checkpoint_config(config, model_config),
        step=config.steps,
        history=history,
    )
    print(f"saved TCIG summary and checkpoint to {output_dir}")
    if config.tensorboard_dir:
        print(f"saved tensorboard events to {config.tensorboard_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCIG on TopCoW.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--max-clicks", type=int, default=5)
    parser.add_argument("--exclude-history-radius", type=int, default=2)
    parser.add_argument("--loss-rounds", choices=("all", "final", "random"), default="all")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(32, 64, 64))
    parser.add_argument("--full-volume", action="store_true", help="Train on full volumes instead of patches.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--train-max-cases", type=int, default=0)
    parser.add_argument("--val-max-cases", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--prompt-sigma", type=float, default=5.0)
    parser.add_argument("--scatter-init-sigma", type=float, default=4.0)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--tensorboard-flush-secs", type=int, default=30)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _validate_config(config: TrainConfig) -> None:
    if config.steps < 1:
        raise ValueError("--steps must be at least 1")
    if config.max_clicks < 1:
        raise ValueError("--max-clicks must be at least 1")
    if config.exclude_history_radius < 0:
        raise ValueError("--exclude-history-radius must be non-negative")
    if config.batch_size != 1:
        raise ValueError("TCIG currently supports --batch-size 1 only")
    if config.patch_size is not None and (len(config.patch_size) != 3 or any(v < 2 for v in config.patch_size)):
        raise ValueError("--patch-size must contain three values >= 2")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    if config.eval_every < 0 or config.save_every < 0:
        raise ValueError("--eval-every and --save-every must be non-negative")
    if config.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if config.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if config.embed_dim < 8 or config.embed_dim % 4 != 0:
        raise ValueError("--embed-dim must be >= 8 and divisible by 4")
    if config.base_channels < 1:
        raise ValueError("--base-channels must be positive")
    if config.prompt_sigma <= 0 or config.scatter_init_sigma <= 0:
        raise ValueError("prompt/scatter sigmas must be positive")
    if config.tensorboard_flush_secs < 1:
        raise ValueError("--tensorboard-flush-secs must be at least 1")


def _checkpoint_config(config: TrainConfig, model_config: TCIGSegmenterConfig) -> Dict[str, object]:
    return {"train": asdict(config), "model": asdict(model_config)}


def _cycle(loader: DataLoader) -> Iterable[Dict[str, object]]:
    while True:
        for batch in loader:
            yield batch


def _create_summary_writer(config: TrainConfig):
    if not config.tensorboard_dir:
        return None
    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(config.tensorboard_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir), flush_secs=config.tensorboard_flush_secs)


def _log_train_row(writer, row: Dict[str, object], step: int) -> None:
    if writer is None:
        return
    tags = {
        "loss/train": "loss",
        "dice/initial": "dice_initial",
        "dice/final": "dice_final",
        "interaction/num_clicks": "num_clicks",
        "interaction/rounds": "rounds",
        "trust/mean": "trust_mean",
        "trust/std": "trust_std",
        "uncertainty/mean": "uncertainty_mean",
        "scatter/gain": "scatter_gain",
        "scatter/context_abs_mean": "spatial_context_abs_mean",
        "optim/lr": "learning_rate",
    }
    for tag, key in tags.items():
        value = row.get(key)
        if isinstance(value, (int, float)):
            writer.add_scalar(tag, float(value), step)


def _log_eval_summary(writer, summary: Dict[str, object], step: int) -> None:
    if writer is None:
        return
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"eval/{key}", float(value), step)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    sys.exit(main())
