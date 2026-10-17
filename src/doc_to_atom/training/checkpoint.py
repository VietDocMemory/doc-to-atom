from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .state import TrainerState

SCHEMA_VERSION = 2
COMMIT_MARKER = "COMMITTED"


class CheckpointError(RuntimeError):
    """Raised for corrupt, partial, or unsupported checkpoints."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when checkpoint provenance differs from the requested run."""


def _json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _capture_rng() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    numpy_state = np.random.get_state()
    metadata = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
    }
    tensors = {"cpu": torch.get_rng_state().cpu()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        tensors.update(
            {f"cuda_{index}": value.cpu() for index, value in enumerate(torch.cuda.get_rng_state_all())}
        )
    return metadata, tensors


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _restore_rng(metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> None:
    random.setstate(_nested_tuple(metadata["python"]))
    numpy_state = metadata["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(tensors["cpu"].cpu())
    cuda_states = [
        tensors[f"cuda_{index}"]
        for index in range(torch.cuda.device_count())
        if f"cuda_{index}" in tensors
    ]
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def _module_tensors(
    modules: Mapping[str, nn.Module],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    tensors: dict[str, torch.Tensor] = {}
    index: dict[str, list[str]] = {}
    for module_name, module in modules.items():
        keys: list[str] = []
        for key, value in module.state_dict().items():
            if not isinstance(value, torch.Tensor):
                raise CheckpointError(f"non-tensor state is unsupported: {module_name}.{key}")
            storage_key = f"{module_name}::{key}"
            tensors[storage_key] = value.detach().cpu().contiguous()
            keys.append(key)
        index[module_name] = keys
    return tensors, index


def _load_modules(
    modules: Mapping[str, nn.Module], checkpoint: Path, module_index: Mapping[str, list[str]]
) -> None:
    tensors = load_file(checkpoint / "model.safetensors", device="cpu")
    for module_name, module in modules.items():
        if module_name not in module_index:
            raise CheckpointError(f"module missing from checkpoint: {module_name}")
        state_dict = {
            key: tensors[f"{module_name}::{key}"] for key in module_index[module_name]
        }
        module.load_state_dict(state_dict, strict=True)


def _validate_fingerprints(
    state: TrainerState, expected: Mapping[str, str] | None
) -> None:
    if not expected:
        return
    actual = {
        "config": state.config_fingerprint,
        "dataset": state.dataset_fingerprint,
        "model": state.model_fingerprint,
        "prompt": state.prompt_fingerprint,
    }
    mismatches = [name for name, value in expected.items() if actual.get(name) != value]
    if mismatches:
        raise IncompatibleCheckpointError(
            "checkpoint fingerprint mismatch: " + ", ".join(sorted(mismatches))
        )


def save_training_checkpoint(
    modules: Mapping[str, nn.Module],
    path: str | Path,
    *,
    state: TrainerState,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> Path:
    """Write checkpoint v2 to a sibling temp directory and publish it atomically."""
    if state.schema_version != SCHEMA_VERSION:
        raise CheckpointError(f"cannot write trainer schema {state.schema_version}")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    backup: Path | None = None
    try:
        tensors, module_index = _module_tensors(modules)
        save_file(tensors, temporary / "model.safetensors")
        _json_dump(temporary / "modules.json", module_index)
        _json_dump(temporary / "state.json", state.to_dict())
        training_state = {
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
        }
        torch.save(training_state, temporary / "training.pt")
        rng_metadata, rng_tensors = _capture_rng()
        _json_dump(temporary / "rng.json", rng_metadata)
        save_file(rng_tensors, temporary / "rng.safetensors")
        (temporary / COMMIT_MARKER).write_text("v2\n", encoding="utf-8")

        if target.exists():
            backup = target.with_name(f".{target.name}.old-{uuid.uuid4().hex}")
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup is not None:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink(missing_ok=True)
        return target
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_training_checkpoint(
    modules: Mapping[str, nn.Module],
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    expected_fingerprints: Mapping[str, str] | None = None,
    restore_rng: bool = True,
    allow_legacy_weights: bool = False,
) -> TrainerState:
    checkpoint = Path(path).resolve()
    if checkpoint.is_file():
        if not allow_legacy_weights:
            raise CheckpointError("legacy v1 checkpoint requires explicit weights-only migration")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for name, module in modules.items():
            module.load_state_dict(payload["modules"][name])
        step = int(payload.get("step", 0))
        return TrainerState(optimizer_step=step, curriculum_step=step, exact_resume=False)
    if not checkpoint.is_dir() or not (checkpoint / COMMIT_MARKER).is_file():
        raise CheckpointError(f"checkpoint is missing or not committed: {checkpoint}")
    raw_state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    state = TrainerState.from_dict(raw_state)
    if state.schema_version != SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema: {state.schema_version}")
    _validate_fingerprints(state, expected_fingerprints)
    module_index = json.loads((checkpoint / "modules.json").read_text(encoding="utf-8"))
    _load_modules(modules, checkpoint, module_index)
    training_state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
    if optimizer is not None and training_state["optimizer"] is not None:
        optimizer.load_state_dict(training_state["optimizer"])
    if scheduler is not None and training_state["scheduler"] is not None:
        scheduler.load_state_dict(training_state["scheduler"])
    if scaler is not None and training_state["scaler"] is not None:
        scaler.load_state_dict(training_state["scaler"])
    if restore_rng:
        rng_metadata = json.loads((checkpoint / "rng.json").read_text(encoding="utf-8"))
        rng_tensors = load_file(checkpoint / "rng.safetensors", device="cpu")
        _restore_rng(rng_metadata, rng_tensors)
    return state


def save_checkpoint(
    modules: dict[str, nn.Module],
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> None:
    """Compatibility wrapper that writes checkpoint v2 with a minimal TrainerState."""
    save_training_checkpoint(
        modules,
        path,
        state=TrainerState(optimizer_step=step, curriculum_step=step),
        optimizer=optimizer,
    )


def load_checkpoint(
    modules: dict[str, nn.Module],
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Compatibility wrapper returning the historic integer step."""
    state = load_training_checkpoint(
        modules,
        path,
        optimizer=optimizer,
        allow_legacy_weights=True,
    )
    return state.optimizer_step


__all__ = [
    "CheckpointError",
    "IncompatibleCheckpointError",
    "TrainerState",
    "load_checkpoint",
    "load_training_checkpoint",
    "save_checkpoint",
    "save_training_checkpoint",
]
