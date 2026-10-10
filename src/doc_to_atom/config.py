from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if base_name := config.get("_base_"):
        base = load_config(path.parent / base_name)
        config = _merge(base, config)
    return config


def require_disclosed_training_values(config: dict[str, Any]) -> None:
    training = config["training"]
    missing = [name for name, value in training["loss_weights"].items() if value is None]
    if training.get("sparse_target_density") is None:
        missing.append("sparse_target_density/rho")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"PAPER_UNSPECIFIED values must be supplied before training: {joined}. "
            "Use a separate assumption config; do not alter the paper config silently."
        )


def write_run_manifest(config: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    deviations: list[str] = []
    runtime = config.get("runtime", {})
    if runtime.get("profile") == "resource_constrained":
        deviations.extend(
            [
                "single-GPU/resource-constrained profile",
                "effective batch differs from paper" if config["training"]["effective_global_batch"] != 128 else "",
                "loss coefficients and rho are assumptions because the paper does not publish them",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"config": config, "reproduction_deviations": [item for item in deviations if item]},
            indent=2,
        ),
        encoding="utf-8",
    )

