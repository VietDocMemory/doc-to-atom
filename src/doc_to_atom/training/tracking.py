"""Optional experiment-tracking helpers with no runtime dependency by default."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def flatten_scalars(payload: dict[str, Any], prefix: str = "") -> dict[str, float | int]:
    """Flatten scalar metrics while deliberately excluding example text and predictions."""
    flattened: dict[str, float | int] = {}
    for name, value in payload.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            flattened.update(flatten_scalars(value, key))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            flattened[key] = value
    return flattened


def start_wandb(
    *,
    project: str,
    entity: str | None,
    run_name: str | None,
    mode: str,
    config: dict[str, Any],
) -> tuple[Any, Callable[[dict[str, Any], int], None]]:
    """Initialize W&B only when the caller explicitly asks for tracking."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError("W&B tracking requires `uv sync --extra train`.") from exc
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        config=config,
    )

    def log(metrics: dict[str, Any], step: int) -> None:
        run.log(flatten_scalars(metrics), step=step)

    return run, log
