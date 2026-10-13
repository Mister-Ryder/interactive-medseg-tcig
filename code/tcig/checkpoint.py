"""Checkpoint helpers shared by TCIG training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer=None,
    config: Optional[Dict[str, Any]] = None,
    step: int = 0,
    history=None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": dict(config or {}),
        "step": int(step),
        "history": list(history or []),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model,
    optimizer=None,
    map_location="cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
