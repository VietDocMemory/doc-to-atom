from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import doc_to_atom.runtime.run as run_module
import doc_to_atom.runtime.tracking as tracking_module


def _config() -> dict:
    return {
        "profile": {"name": "test", "family": "experiments"},
        "assumptions": [],
        "deviations": [],
        "training": {"learning_rate": 0.001},
    }


def test_run_context_creates_contract_and_redacted_manifest(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 10, 17, 8, 30, tzinfo=UTC)
    config = _config() | {"provider": {"api_key": "do-not-write", "model": "fake"}}

    context = run_module.RunContext.create(
        tmp_path / "run",
        config=config,
        command=["doc2atom-train", "--api-key", "also-secret"],
        now=fixed_time,
        hostname="molab-node",
        environment={"CUDA_VISIBLE_DEVICES": "0,1", "WANDB_API_KEY": "third-secret"},
    )

    expected = {
        "config",
        "manifests",
        "logs",
        "checkpoints",
        "memory",
        "predictions",
        "metrics",
        "reports",
        "wandb",
    }
    assert {path.name for path in context.root.iterdir() if path.is_dir()} == expected
    manifest = json.loads(context.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["created_at"] == "2026-10-17T08:30:00+00:00"
    assert manifest["hostname"] == "molab-node"
    assert manifest["config_fingerprint"] == context.config_fingerprint
    assert manifest["command"] == ["doc2atom-train", "--api-key", "<redacted>"]
    all_text = "\n".join(
        path.read_text(encoding="utf-8") for path in context.root.rglob("*") if path.is_file()
    )
    assert "do-not-write" not in all_text
    assert "also-secret" not in all_text
    assert "third-secret" not in all_text


def test_run_id_is_reused_only_for_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    first = run_module.RunContext.create(
        output, config=_config(), command=["train"], run_id="fixed-run"
    )
    resumed = run_module.RunContext.create(
        output, config=_config(), command=["train", "--resume"], resume=True
    )

    assert first.run_id == resumed.run_id == "fixed-run"
    with pytest.raises(run_module.RunContextError, match="already contains a run"):
        run_module.RunContext.create(output, config=_config(), command=["train"])


def test_resume_rejects_a_changed_config(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_module.RunContext.create(output, config=_config(), command=["train"])
    changed = _config()
    changed["training"]["learning_rate"] = 0.5

    with pytest.raises(run_module.RunContextError, match="config fingerprint"):
        run_module.RunContext.create(output, config=changed, command=["train"], resume=True)


def test_event_records_are_structured_and_secrets_are_redacted(tmp_path: Path) -> None:
    context = run_module.RunContext.create(tmp_path / "run", config=_config(), command=["train"])

    context.log_event("validation", {"loss": 1.25, "access_token": "hidden"})

    event = json.loads(context.events_path.read_text(encoding="utf-8").strip())
    assert event["event"] == "validation"
    assert event["data"] == {"loss": 1.25, "access_token": "<redacted>"}
    assert event["run_id"] == context.run_id


class _FakeRun:
    def __init__(self, run_id: str = "wandb-id") -> None:
        self.id = run_id
        self.summary = {}
        self.logged = []

    def log(self, values: dict, step: int | None = None) -> None:
        self.logged.append((values, step))

    def save(self, path: str, base_path: str | None = None, policy: str = "now") -> None:
        return None

    def finish(self) -> None:
        return None


def test_tracker_disabled_mode_is_local_noop(tmp_path: Path) -> None:
    context = run_module.RunContext.create(tmp_path / "run", config=_config(), command=["train"])
    tracker = tracking_module.RunTracker.create(context, mode="disabled", project="tests")

    tracker.log_metrics({"loss": 2.0}, step=1)
    tracker.finish()

    assert tracker.mode == "disabled"
    assert json.loads(context.events_path.read_text(encoding="utf-8").splitlines()[-1])[
        "event"
    ] == "metrics"


def test_tracker_falls_back_from_online_to_offline(tmp_path: Path) -> None:
    context = run_module.RunContext.create(tmp_path / "run", config=_config(), command=["train"])
    attempted_modes: list[str] = []

    def initialize(**kwargs):
        attempted_modes.append(kwargs["mode"])
        if kwargs["mode"] == "online":
            raise ConnectionError("network unavailable")
        return _FakeRun()

    tracker = tracking_module.RunTracker.create(
        context, mode="online", project="tests", initializer=initialize
    )
    tracker.log_metrics({"loss": 0.5}, step=3)

    assert attempted_modes == ["online", "offline"]
    assert tracker.mode == "offline"
    assert tracker.run_id == "wandb-id"
    warnings = context.warnings_path.read_text(encoding="utf-8")
    assert "network unavailable" in warnings


def test_tracker_offline_logs_config_manifests_and_checkpoints_locally(tmp_path: Path) -> None:
    context = run_module.RunContext.create(tmp_path / "run", config=_config(), command=["train"])
    initialized_modes: list[str] = []

    def initialize(**kwargs):
        initialized_modes.append(kwargs["mode"])
        return _FakeRun()

    tracker = tracking_module.RunTracker.create(
        context, mode="offline", project="tests", initializer=initialize
    )
    checkpoint = context.root / "checkpoints" / "state.pt"
    checkpoint.write_bytes(b"state")
    tracker.log_config()
    tracker.log_manifest(context.run_manifest_path)
    tracker.log_checkpoint(checkpoint)

    events = [
        json.loads(line) for line in context.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert initialized_modes == ["offline"]
    assert [event["event"] for event in events] == ["config", "artifact", "artifact"]

