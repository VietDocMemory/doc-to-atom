"""W&B tracking with local-first logging and offline fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .run import RunContext, redact_secrets


def _wandb_initialize(**kwargs: Any):
    import wandb

    return wandb.init(**kwargs)


@dataclass
class RunTracker:
    context: RunContext
    mode: str
    _run: Any = None

    @property
    def run_id(self) -> str:
        return str(getattr(self._run, "id", self.context.run_id))

    @classmethod
    def create(
        cls,
        context: RunContext,
        *,
        mode: str,
        project: str,
        initializer: Callable[..., Any] | None = None,
        entity: str | None = None,
        group: str | None = None,
    ) -> "RunTracker":
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("tracking mode must be online, offline, or disabled")
        if mode == "disabled":
            return cls(context=context, mode=mode)
        initialize = initializer or _wandb_initialize
        kwargs = {
            "project": project,
            "entity": entity,
            "group": group,
            "id": context.run_id,
            "resume": "allow",
            "dir": str(context.wandb_dir),
            "config": redact_secrets(context.config),
        }
        try:
            run = initialize(mode=mode, **kwargs)
            return cls(context=context, mode=mode, _run=run)
        except Exception as exc:
            if mode != "online":
                raise
            context.log_warning("wandb_online_fallback", str(exc), requested_mode="online")
            run = initialize(mode="offline", **kwargs)
            return cls(context=context, mode="offline", _run=run)

    def log_metrics(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        cleaned = redact_secrets(dict(values))
        self.context.log_event("metrics", {"step": step, "values": cleaned})
        if self._run is not None:
            self._run.log(cleaned, step=step)

    def log_config(self) -> None:
        path = self.context.root / "config" / "resolved.yaml"
        self.context.log_event(
            "config", {"path": str(path), "fingerprint": self.context.config_fingerprint}
        )
        if self._run is not None:
            self._run.save(str(path), base_path=str(self.context.root), policy="now")

    def log_file(self, path: str | Path) -> None:
        resolved = Path(path).resolve()
        self.context.log_event("artifact", {"path": str(resolved), "kind": "file"})
        if self._run is not None:
            self._run.save(str(resolved), base_path=str(self.context.root), policy="now")

    def log_manifest(self, path: str | Path) -> None:
        self.log_file(path)

    def log_checkpoint(self, path: str | Path) -> None:
        resolved = Path(path)
        if resolved.is_dir():
            for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
                self.log_file(child)
        else:
            self.log_file(resolved)

    def _artifact_status(self, handle: Any, *, checkpoint: Path, optimizer_step: int, name: str) -> dict:
        status = "disabled" if self._run is None else "offline_pending"
        if self._run is not None and self.mode == "online":
            if handle is None or not callable(getattr(handle, "wait", None)):
                raise RuntimeError("online W&B artifact did not return a waitable upload handle")
            handle.wait()
            status = "uploaded"
        return {
            "status": status,
            "checkpoint": checkpoint.name,
            "optimizer_step": optimizer_step,
            "artifact": name,
        }

    def log_best_weights(self, checkpoint: str | Path, *, optimizer_step: int) -> dict:
        """Upload one model-only best artifact, never exact-resume state or raw responses."""
        root = Path(checkpoint).resolve()
        weights = root / "model.safetensors"
        if not weights.is_file():
            weights = root / "portable" / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"best model weights do not exist: {root}")
        self.context.log_event(
            "best_model", {"path": str(weights), "optimizer_step": optimizer_step}
        )
        name = f"{self.context.run_id}-best-model"
        if self._run is None:
            return self._artifact_status(None, checkpoint=root, optimizer_step=optimizer_step, name=name)
        import wandb

        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata={
                "optimizer_step": optimizer_step,
                "config_fingerprint": self.context.config_fingerprint,
            },
        )
        artifact.add_file(str(weights), name="model.safetensors")
        artifact.add_file(
            str(self.context.root / "config" / "resolved.yaml"),
            name="config/resolved.yaml",
        )
        artifact.add_file(str(self.context.run_manifest_path), name="manifests/run.json")
        handle = self._run.log_artifact(artifact, aliases=["best"])
        return self._artifact_status(handle, checkpoint=root, optimizer_step=optimizer_step, name=name)

    def log_last_checkpoint(self, checkpoint: str | Path, *, optimizer_step: int) -> dict:
        """Publish the full committed checkpoint, including optimizer and RNG state."""
        root = Path(checkpoint).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"last checkpoint does not exist: {root}")
        name = f"{self.context.run_id}-last-checkpoint"
        files = [
            path for path in sorted(root.rglob("*"))
            if path.is_file() and ".raw-responses" not in path.name
        ]
        if not files:
            raise FileNotFoundError(f"last checkpoint is empty: {root}")
        self.context.log_event(
            "last_checkpoint", {"path": str(root), "optimizer_step": optimizer_step}
        )
        if self._run is None:
            return self._artifact_status(None, checkpoint=root, optimizer_step=optimizer_step, name=name)
        import wandb

        artifact = wandb.Artifact(
            name=name, type="checkpoint",
            metadata={
                "optimizer_step": optimizer_step,
                "config_fingerprint": self.context.config_fingerprint,
            },
        )
        for path in files:
            artifact.add_file(str(path), name=f"checkpoint/{path.relative_to(root).as_posix()}")
        artifact.add_file(
            str(self.context.root / "config" / "resolved.yaml"),
            name="config/resolved.yaml",
        )
        artifact.add_file(str(self.context.run_manifest_path), name="manifests/run.json")
        handle = self._run.log_artifact(artifact, aliases=["last"])
        return self._artifact_status(handle, checkpoint=root, optimizer_step=optimizer_step, name=name)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

