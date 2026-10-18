from __future__ import annotations

import hashlib
import importlib.metadata
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from doc_to_atom.training.checkpoint import COMMIT_MARKER


class MolabGateError(RuntimeError):
    """Raised when the Molab ZeRO-2 correctness gate is not runnable or complete."""


@dataclass(frozen=True)
class RuntimeCapabilities:
    gpu_count: int
    nccl_available: bool
    torch_version: str
    cuda_version: str | None
    deepspeed_version: str | None

    @classmethod
    def detect(cls) -> "RuntimeCapabilities":
        try:
            deepspeed_version = importlib.metadata.version("deepspeed")
        except importlib.metadata.PackageNotFoundError:
            deepspeed_version = None
        return cls(
            gpu_count=torch.cuda.device_count(),
            nccl_available=torch.distributed.is_nccl_available(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            deepspeed_version=deepspeed_version,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(child)))
    return digest.hexdigest()


def _verify_writable(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / f".write-probe-{os.getpid()}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    except OSError as exc:
        raise MolabGateError(f"output is not writable: {output_dir}: {exc}") from exc
    finally:
        probe.unlink(missing_ok=True)


def validate_preflight(
    config: Mapping[str, Any],
    *,
    world_size: int,
    output_dir: str | Path,
    config_paths: Sequence[str | Path],
    capabilities: RuntimeCapabilities | None = None,
) -> dict[str, Any]:
    """Validate the two-GPU gate before allocating a model or launching training."""
    capabilities = capabilities or RuntimeCapabilities.detect()
    blockers: list[str] = []
    if capabilities.gpu_count < world_size:
        blockers.append(
            f"GPU count is {capabilities.gpu_count}, but world_size={world_size} is required"
        )
    if not capabilities.nccl_available:
        blockers.append("NCCL backend is unavailable")
    if capabilities.deepspeed_version is None:
        blockers.append("DeepSpeed is not installed")
    if blockers:
        raise MolabGateError("Molab ZeRO-2 preflight failed: " + "; ".join(blockers))

    training = config["training"]
    computed_batch = (
        int(training["micro_batch_size"])
        * int(training["gradient_accumulation_steps"])
        * int(world_size)
    )
    declared_batch = int(training["effective_global_batch"])
    if declared_batch != computed_batch:
        raise MolabGateError(
            "effective_global_batch mismatch: "
            f"declared={declared_batch}, computed={computed_batch}"
        )

    resolved_paths = [Path(value).resolve() for value in config_paths]
    missing = [str(path) for path in resolved_paths if not path.is_file()]
    if missing:
        raise MolabGateError("missing configuration files: " + ", ".join(missing))
    output = Path(output_dir).resolve()
    _verify_writable(output)
    return {
        "world_size": int(world_size),
        "effective_global_batch": computed_batch,
        "output_dir": str(output),
        "versions": {
            "torch": capabilities.torch_version,
            "cuda": capabilities.cuda_version,
            "deepspeed": capabilities.deepspeed_version,
        },
        "gpu_count": capabilities.gpu_count,
        "nccl_available": capabilities.nccl_available,
        "config_sha256": {str(path): _sha256(path) for path in resolved_paths},
    }


def validate_optimizer_shards(
    checkpoint: str | Path, *, world_size: int
) -> dict[str, Any]:
    """Prove a committed ZeRO checkpoint contains rank-sharded optimizer state."""
    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / COMMIT_MARKER).is_file():
        raise MolabGateError(f"distributed checkpoint is not committed: {checkpoint}")
    accelerate_state = checkpoint / "accelerate"
    shards = sorted(accelerate_state.rglob("*optim_states.pt"))
    rank_zero_file = accelerate_state / "optimizer.bin"
    if len(shards) < world_size:
        suffix = (
            "; optimizer.bin exists but cannot prove rank-sharded ZeRO-2 state"
            if rank_zero_file.is_file()
            else ""
        )
        raise MolabGateError(
            f"rank-sharded ZeRO-2 optimizer state is incomplete: "
            f"found {len(shards)}, expected at least {world_size}{suffix}"
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _tree_sha256(checkpoint),
        "optimizer_shard_count": len(shards),
        "optimizer_shards": [str(path.relative_to(checkpoint)) for path in shards],
        "capabilities": asdict(RuntimeCapabilities.detect()),
    }


__all__ = [
    "MolabGateError",
    "RuntimeCapabilities",
    "validate_optimizer_shards",
    "validate_preflight",
]
