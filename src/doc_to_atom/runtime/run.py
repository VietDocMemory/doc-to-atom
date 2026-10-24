"""Local run artifacts are the source of truth; remote tracking is a mirror."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from doc_to_atom.config import config_fingerprint, config_fingerprint_candidates, load_config
from doc_to_atom.credentials import redact_secrets

_DIRECTORIES = (
    "config",
    "manifests",
    "logs",
    "checkpoints",
    "memory",
    "predictions",
    "metrics",
    "reports",
    "wandb",
)


class RunContextError(RuntimeError):
    """Raised when a run directory cannot be created or resumed safely."""


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(redact_secrets(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def atomic_write_yaml(path: Path, value: Any) -> None:
    atomic_write_text(
        path, yaml.safe_dump(redact_secrets(value), allow_unicode=True, sort_keys=True)
    )


def _verify_config_snapshot(path: Path, fingerprint: str) -> None:
    try:
        matches = config_fingerprint(load_config(path)) == fingerprint
    except (ValueError, OSError, TypeError, yaml.YAMLError):
        raise RunContextError("resolved config snapshot cannot be verified") from None
    if not matches:
        raise RunContextError(
            "resolved config snapshot fingerprint differs from the effective training config; "
            "supply the original resolved config with all training overrides to recover the run"
        )


def _sanitize_run_metadata(root: Path) -> None:
    """Protect metadata written by older versions before tracking a resumed run."""
    for directory in (root / "config", root / "manifests"):
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix not in {".yaml", ".yml", ".json"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
                value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
                if value != redact_secrets(value):
                    writer = atomic_write_json if path.suffix == ".json" else atomic_write_yaml
                    writer(path, value)
            except (ValueError, OSError, TypeError, yaml.YAMLError):
                raise RunContextError("cannot sanitize historical run metadata before tracking") from None


def _git_state(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _package_versions() -> dict[str, str]:
    wanted = {"torch", "transformers", "accelerate", "deepspeed", "wandb", "doc-to-atom"}
    result: dict[str, str] = {}
    for name in sorted(wanted):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return result


@dataclass(frozen=True)
class RunContext:
    root: Path
    run_id: str
    config: dict[str, Any]
    config_fingerprint: str

    @property
    def run_manifest_path(self) -> Path:
        return self.root / "manifests" / "run.json"

    @property
    def events_path(self) -> Path:
        return self.root / "logs" / "events.jsonl"

    @property
    def warnings_path(self) -> Path:
        return self.root / "logs" / "warnings.jsonl"

    @property
    def wandb_dir(self) -> Path:
        return self.root / "wandb"

    @classmethod
    def create(
        cls,
        output_dir: str | Path,
        *,
        config: dict[str, Any],
        command: Sequence[str],
        run_id: str | None = None,
        resume: bool = False,
        now: datetime | None = None,
        hostname: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> "RunContext":
        root = Path(output_dir).resolve()
        manifest_path = root / "manifests" / "run.json"
        fingerprint = config_fingerprint(config)
        if resume:
            if not manifest_path.is_file():
                raise RunContextError(f"cannot resume: missing run manifest at {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_fingerprint") not in config_fingerprint_candidates(config):
                raise RunContextError("cannot resume: config fingerprint differs from the original run")
            snapshot = root / "config" / "resolved.yaml"
            _verify_config_snapshot(snapshot, fingerprint)
            _sanitize_run_metadata(root)
            _verify_config_snapshot(snapshot, fingerprint)
            return cls(root, str(manifest["run_id"]), config, fingerprint)
        if manifest_path.exists():
            raise RunContextError(f"output directory already contains a run: {root}")

        instant = now or datetime.now(UTC)
        resolved_id = run_id or f"{instant:%Y%m%dT%H%M%SZ}-{fingerprint[:8]}-{uuid.uuid4().hex[:6]}"
        for name in _DIRECTORIES:
            (root / name).mkdir(parents=True, exist_ok=True)
        context = cls(root, resolved_id, config, fingerprint)
        atomic_write_yaml(root / "config" / "resolved.yaml", config)
        _verify_config_snapshot(root / "config" / "resolved.yaml", fingerprint)
        atomic_write_json(root / "config" / "assumptions.json", config.get("assumptions", []))
        captured_environment = dict(environment) if environment is not None else {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "NCCL_", "WANDB_", "HF_", "PYTORCH_"))
        }
        atomic_write_json(
            root / "manifests" / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": _package_versions(),
                "environment": captured_environment,
            },
        )
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "run_id": resolved_id,
                "created_at": instant.isoformat(),
                "hostname": hostname or socket.gethostname(),
                "command": list(command),
                "config_fingerprint": fingerprint,
                "profile": config.get("profile"),
                "active_assumption_ids": [item["id"] for item in config.get("assumptions", [])],
                "active_deviation_ids": [item["id"] for item in config.get("deviations", [])],
                "git": _git_state(Path.cwd()),
            },
        )
        return context

    def log_event(self, event: str, data: Mapping[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "data": redact_secrets(dict(data)),
        }
        self._append_jsonl(self.events_path, record)

    def log_warning(self, code: str, message: str, **data: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "code": code,
            "message": message,
            "data": redact_secrets(data),
        }
        self._append_jsonl(self.warnings_path, record)

    @staticmethod
    def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

