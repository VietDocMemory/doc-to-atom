from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    modules: dict[str, nn.Module],
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizer": None if optimizer is None else optimizer.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    modules: dict[str, nn.Module],
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name])
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"])
