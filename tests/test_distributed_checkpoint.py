from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from doc_to_atom.training.checkpoint import (
    CheckpointError,
    IncompatibleCheckpointError,
    load_distributed_checkpoint,
    prune_unreferenced_checkpoints,
    publish_checkpoint_pointer,
    save_distributed_checkpoint,
)
from doc_to_atom.training.state import TrainerState


class FakeDistributedRuntime:
    def __init__(
        self,
        *,
        main=True,
        fail_save=False,
        backend="accelerate",
        world_size=2,
        accumulation_steps=2,
    ):
        self.is_main_process = main
        self.rank = 0 if main else 1
        self.fail_save = fail_save
        self.backend = backend
        self.world_size = world_size
        self.accumulation_steps = accumulation_steps
        self.events: list[str] = []

    def wait_for_everyone(self):
        self.events.append("barrier")

    def save_state(self, path: Path):
        self.events.append("save_state")
        if self.fail_save:
            raise RuntimeError("rank save failed")
        path.mkdir(parents=True, exist_ok=True)
        (path / f"rank-{self.rank}.state").write_text("ok\n", encoding="utf-8")

    def load_state(self, path: Path):
        assert path.is_dir()
        self.events.append("load_state")

    def collective_errors(self, local_error):
        self.events.append("collective_errors")
        return [local_error, None]


def _state(**overrides):
    values = {
        "optimizer_step": 3,
        "curriculum_step": 3,
        "world_size": 2,
        "distributed_backend": "accelerate",
        "accumulation_steps": 2,
        "config_fingerprint": "config-hash",
    }
    values.update(overrides)
    return TrainerState(**values)


def test_distributed_checkpoint_publishes_only_after_all_rank_state_and_portable_weights(
    tmp_path: Path,
) -> None:
    runtime = FakeDistributedRuntime()
    module = nn.Linear(2, 1)
    expected = module.weight.detach().clone()
    destination = tmp_path / "step-00000003"

    published = save_distributed_checkpoint(
        runtime,
        {"module": module},
        destination,
        state=_state(),
    )

    assert published == destination.resolve()
    assert runtime.events.index("save_state") < runtime.events.index("collective_errors")
    assert (destination / "COMMITTED").is_file()
    assert (destination / "accelerate" / "rank-0.state").is_file()
    assert (destination / "portable" / "COMMITTED").is_file()
    with torch.no_grad():
        module.weight.zero_()
    from doc_to_atom.training.checkpoint import load_training_checkpoint

    load_training_checkpoint({"module": module}, destination / "portable")
    assert torch.equal(module.weight, expected)


def test_distributed_checkpoint_failure_never_publishes_or_updates_pointer(
    tmp_path: Path,
) -> None:
    runtime = FakeDistributedRuntime(fail_save=True)
    destination = tmp_path / "step-00000003"
    pointer = tmp_path / "last.json"

    with pytest.raises(CheckpointError, match="rank save failed"):
        save_distributed_checkpoint(
            runtime, {"module": nn.Linear(1, 1)}, destination, state=_state()
        )
    assert not destination.exists()
    assert not pointer.exists()

    with pytest.raises(CheckpointError, match="committed"):
        publish_checkpoint_pointer(pointer, destination, optimizer_step=3)


def test_pointer_is_written_only_for_committed_checkpoint(tmp_path: Path) -> None:
    runtime = FakeDistributedRuntime()
    destination = save_distributed_checkpoint(
        runtime, {"module": nn.Linear(1, 1)}, tmp_path / "step", state=_state()
    )
    pointer = tmp_path / "last.json"

    publish_checkpoint_pointer(pointer, destination, optimizer_step=3)

    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload == {"optimizer_step": 3, "path": "step"}


def test_distributed_retention_keeps_resumable_best_and_last_shards(tmp_path: Path):
    runtime = FakeDistributedRuntime(backend="deepspeed_zero2")
    checkpoints = []
    for step in (1, 2, 3):
        target = save_distributed_checkpoint(
            runtime, {"module": nn.Linear(1, 1)}, tmp_path / f"step-{step:08d}",
            state=_state(optimizer_step=step, distributed_backend="deepspeed_zero2"),
        )
        checkpoints.append(target)
    publish_checkpoint_pointer(tmp_path / "best.json", checkpoints[0], optimizer_step=1)
    publish_checkpoint_pointer(tmp_path / "last.json", checkpoints[2], optimizer_step=3)

    removed = prune_unreferenced_checkpoints(tmp_path)

    assert removed == [checkpoints[1]]
    for target in (checkpoints[0], checkpoints[2]):
        assert (target / "accelerate" / "rank-0.state").is_file()
        assert (target / "portable" / "model.safetensors").is_file()
    restored = load_distributed_checkpoint(
        runtime, checkpoints[2], expected_fingerprints={"config": "config-hash"}
    )
    assert restored.optimizer_step == 3 and restored.exact_resume


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (_state(world_size=4), "world_size"),
        (_state(distributed_backend="deepspeed_zero2"), "distributed_backend"),
        (_state(accumulation_steps=8), "accumulation_steps"),
    ],
)
def test_distributed_resume_rejects_topology_before_accelerate_load(
    tmp_path: Path, state: TrainerState, message: str
) -> None:
    writer = FakeDistributedRuntime(
        world_size=state.world_size,
        backend=state.distributed_backend,
        accumulation_steps=state.accumulation_steps,
    )
    destination = save_distributed_checkpoint(
        writer, {"module": nn.Linear(1, 1)}, tmp_path / message, state=state
    )
    reader = FakeDistributedRuntime()

    with pytest.raises(IncompatibleCheckpointError, match=message):
        load_distributed_checkpoint(
            reader,
            destination,
            expected_fingerprints={"config": "config-hash"},
        )
    assert "load_state" not in reader.events


def test_distributed_resume_loads_accelerate_state_before_returning_control_state(
    tmp_path: Path,
) -> None:
    writer = FakeDistributedRuntime()
    destination = save_distributed_checkpoint(
        writer, {"module": nn.Linear(1, 1)}, tmp_path / "step", state=_state()
    )
    reader = FakeDistributedRuntime()

    restored = load_distributed_checkpoint(
        reader,
        destination,
        expected_fingerprints={"config": "config-hash"},
    )

    assert reader.events[0] == "load_state"
    assert restored.optimizer_step == 3
