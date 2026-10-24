from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
import uuid
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from doc_to_atom.runtime.run import atomic_write_json

from .state import TrainerState

SCHEMA_VERSION = 2
COMMIT_MARKER = "COMMITTED"
_STEP_DIRECTORY = re.compile(r"step-[0-9]{8,}\Z")


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
    state: TrainerState, expected: Mapping[str, str | Collection[str]] | None
) -> None:
    if not expected:
        return
    actual = {
        "config": state.config_fingerprint,
        "dataset": state.dataset_fingerprint,
        "model": state.model_fingerprint,
        "prompt": state.prompt_fingerprint,
    }
    mismatches = []
    for name, value in expected.items():
        candidates = {value} if isinstance(value, str) else value
        if actual.get(name) not in candidates:
            mismatches.append(name)
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
    expected_fingerprints: Mapping[str, str | Collection[str]] | None = None,
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


def _distributed_failures(errors: list[str | None], phase: str) -> None:
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise CheckpointError(
            f"distributed checkpoint {phase} failed; " + "; ".join(failures)
        )


def _validate_distributed_topology(runtime, state: TrainerState) -> None:
    expected = {
        "world_size": int(runtime.world_size),
        "distributed_backend": str(runtime.backend),
        "accumulation_steps": int(runtime.accumulation_steps),
    }
    actual = {
        "world_size": state.world_size,
        "distributed_backend": state.distributed_backend,
        "accumulation_steps": state.accumulation_steps,
    }
    mismatches = [
        f"{name}: checkpoint={actual[name]!r}, runtime={value!r}"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if mismatches:
        raise IncompatibleCheckpointError(
            "distributed checkpoint topology mismatch: " + ", ".join(mismatches)
        )


def save_distributed_checkpoint(
    runtime,
    modules: Mapping[str, nn.Module],
    path: str | Path,
    *,
    state: TrainerState,
) -> Path:
    """Collectively save, then atomically publish, a distributed checkpoint.

    Every rank must call this function. Accelerate owns optimizer, scheduler,
    scaler, and per-rank RNG state. Rank zero additionally writes a portable
    trainable-weights checkpoint for evaluation and non-sharded inspection.
    """
    _validate_distributed_topology(runtime, state)
    target = Path(path).resolve()
    incomplete = target.with_name(f".{target.name}.incomplete")
    if runtime.is_main_process:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise CheckpointError(f"refusing to overwrite committed checkpoint: {target}")
        if incomplete.exists():
            shutil.rmtree(incomplete)
        incomplete.mkdir(parents=True)
    runtime.wait_for_everyone()

    local_error = None
    try:
        runtime.save_state(incomplete / "accelerate")
    except Exception as exc:  # each rank reports before anyone can publish
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_failures(runtime.collective_errors(local_error), "Accelerate save")
    runtime.wait_for_everyone()

    portable_error = None
    if runtime.is_main_process:
        try:
            save_training_checkpoint(
                modules,
                incomplete / "portable",
                state=state,
                optimizer=None,
                scheduler=None,
                scaler=None,
            )
            _json_dump(incomplete / "state.json", state.to_dict())
            _json_dump(
                incomplete / "distributed.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "backend": runtime.backend,
                    "world_size": runtime.world_size,
                    "accumulation_steps": runtime.accumulation_steps,
                    "accelerate_state": "accelerate",
                    "portable_checkpoint": "portable",
                },
            )
        except Exception as exc:
            portable_error = f"{type(exc).__name__}: {exc}"
    _distributed_failures(runtime.collective_errors(portable_error), "portable save")
    runtime.wait_for_everyone()

    if runtime.is_main_process:
        (incomplete / COMMIT_MARKER).write_text("distributed-v2\n", encoding="utf-8")
        os.replace(incomplete, target)
    runtime.wait_for_everyone()
    if not (target / COMMIT_MARKER).is_file():
        raise CheckpointError(f"distributed checkpoint was not committed: {target}")
    return target


def load_distributed_checkpoint(
    runtime,
    path: str | Path,
    *,
    expected_fingerprints: Mapping[str, str | Collection[str]] | None = None,
) -> TrainerState:
    """Validate shared control state before collectively restoring Accelerate."""
    checkpoint = Path(path).resolve()
    if not checkpoint.is_dir() or not (checkpoint / COMMIT_MARKER).is_file():
        raise CheckpointError(f"checkpoint is missing or not committed: {checkpoint}")
    state = TrainerState.from_dict(
        json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    )
    if state.schema_version != SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema: {state.schema_version}")
    _validate_fingerprints(state, expected_fingerprints)
    _validate_distributed_topology(runtime, state)
    runtime.load_state(checkpoint / "accelerate")
    runtime.wait_for_everyone()
    return state


def publish_checkpoint_pointer(
    pointer_path: str | Path,
    checkpoint_path: str | Path,
    *,
    optimizer_step: int,
) -> None:
    """Atomically point at an already committed sibling checkpoint."""
    pointer = Path(pointer_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not (checkpoint / COMMIT_MARKER).is_file():
        raise CheckpointError(f"cannot publish pointer to uncommitted checkpoint: {checkpoint}")
    if checkpoint.parent != pointer.parent:
        raise CheckpointError("checkpoint pointer target must be a sibling directory")
    atomic_write_json(
        pointer,
        {"path": checkpoint.name, "optimizer_step": int(optimizer_step)},
    )


def verify_committed_checkpoint(
    path: str | Path, *, optimizer_step: int, distributed: bool
) -> None:
    """Check the committed checkpoint structure before it replaces a live pointer."""
    checkpoint = Path(path).resolve()
    required = ["COMMITTED", "state.json"]
    if distributed:
        required += [
            "distributed.json", "portable/COMMITTED", "portable/model.safetensors",
            "portable/training.pt", "portable/rng.json", "portable/rng.safetensors",
        ]
    else:
        required += [
            "model.safetensors", "modules.json", "training.pt", "rng.json",
            "rng.safetensors",
        ]
    if not checkpoint.is_dir() or any(
        not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
        for name in required
    ):
        raise CheckpointError(f"checkpoint is incomplete: {checkpoint}")
    try:
        state = TrainerState.from_dict(json.loads((checkpoint / "state.json").read_text()))
    except (OSError, ValueError, TypeError) as exc:
        raise CheckpointError(f"invalid checkpoint state: {checkpoint}") from exc
    if state.schema_version != SCHEMA_VERSION or state.optimizer_step != optimizer_step:
        raise CheckpointError(f"checkpoint step or schema mismatch: {checkpoint}")
    if distributed and not any(
        item.is_file() and item.stat().st_size > 0
        for item in (checkpoint / "accelerate").rglob("*")
    ):
        raise CheckpointError(f"distributed checkpoint has no Accelerate state: {checkpoint}")


def _pointer_target(checkpoint_dir: Path, name: str) -> Path | None:
    pointer = checkpoint_dir / f"{name}.json"
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        basename = payload["path"]
        step = payload["optimizer_step"]
        if not isinstance(basename, str) or not _STEP_DIRECTORY.fullmatch(basename):
            raise ValueError("pointer path must be a step directory basename")
        if type(step) is not int:
            raise ValueError("pointer step must be an integer")
        candidate = checkpoint_dir / basename
        target = candidate.resolve()
        if candidate.is_symlink() or target.parent != checkpoint_dir:
            raise ValueError("pointer target escapes checkpoint directory")
        verify_committed_checkpoint(
            target, optimizer_step=step,
            distributed=(target / "distributed.json").exists(),
        )
        return target
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise CheckpointError(f"invalid {name} checkpoint pointer: {pointer}: {exc}") from exc


def prune_unreferenced_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """Remove only committed step directories not referenced by best or last."""
    root = Path(checkpoint_dir).resolve()
    best = _pointer_target(root, "best")
    last = _pointer_target(root, "last")
    if last is None:
        raise CheckpointError(f"cannot prune without a valid last pointer: {root}")
    protected = {target for target in (best, last) if target is not None}
    removed: list[Path] = []
    for candidate in sorted(root.iterdir()):
        if (
            not _STEP_DIRECTORY.fullmatch(candidate.name)
            or candidate.is_symlink()
            or not candidate.is_dir()
            or not (candidate / COMMIT_MARKER).is_file()
        ):
            continue
        target = candidate.resolve()
        if target.parent != root or target in protected:
            continue
        shutil.rmtree(target)
        removed.append(target)
    return removed


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
    "load_distributed_checkpoint",
    "load_training_checkpoint",
    "publish_checkpoint_pointer",
    "prune_unreferenced_checkpoints",
    "save_checkpoint",
    "save_distributed_checkpoint",
    "save_training_checkpoint",
    "verify_committed_checkpoint",
]
