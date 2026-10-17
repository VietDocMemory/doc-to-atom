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

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

