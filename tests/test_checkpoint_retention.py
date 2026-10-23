import json
from pathlib import Path

import pytest
import torch
from torch import nn

from doc_to_atom.training.checkpoint import (
    CheckpointError,
    prune_unreferenced_checkpoints,
    publish_checkpoint_pointer,
    save_training_checkpoint,
    verify_committed_checkpoint,
)
from doc_to_atom.training.state import TrainerState


def _save(directory: Path, step: int) -> Path:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    destination = directory / f"step-{step:08d}"
    save_training_checkpoint(
        {"model": model}, destination,
        state=TrainerState(optimizer_step=step, curriculum_step=step),
        optimizer=optimizer,
    )
    return destination


def test_prune_keeps_only_distinct_best_and_last(tmp_path: Path):
    first, best, last = (_save(tmp_path, step) for step in (1, 2, 3))
    publish_checkpoint_pointer(tmp_path / "best.json", best, optimizer_step=2)
    publish_checkpoint_pointer(tmp_path / "last.json", last, optimizer_step=3)
    unrelated = tmp_path / "manual-backup"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep")

    removed = prune_unreferenced_checkpoints(tmp_path)

    assert removed == [first.resolve()]
    assert not first.exists()
    assert best.is_dir() and last.is_dir() and unrelated.is_dir()
    assert (best / "training.pt").is_file()
    assert (last / "rng.safetensors").is_file()


def test_prune_keeps_shared_best_last_target(tmp_path: Path):
    old, current = (_save(tmp_path, step) for step in (1, 2))
    for name in ("best", "last"):
        publish_checkpoint_pointer(tmp_path / f"{name}.json", current, optimizer_step=2)

    prune_unreferenced_checkpoints(tmp_path)

    assert not old.exists()
    assert current.is_dir()


@pytest.mark.parametrize("bad_path", ["../outside", "missing", "step-00000001/other"])
def test_prune_refuses_invalid_pointer_without_deleting(tmp_path: Path, bad_path: str):
    stale = _save(tmp_path, 1)
    (tmp_path / "last.json").write_text(json.dumps({"path": bad_path, "optimizer_step": 1}))

    with pytest.raises(CheckpointError):
        prune_unreferenced_checkpoints(tmp_path)

    assert stale.is_dir()


def test_verify_rejects_partial_commit_before_prune(tmp_path: Path):
    target = _save(tmp_path, 2)
    (target / "training.pt").unlink()

    with pytest.raises(CheckpointError):
        verify_committed_checkpoint(target, optimizer_step=2, distributed=False)


def test_verify_distributed_requires_accelerate_and_portable(tmp_path: Path):
    target = tmp_path / "step-00000002"
    target.mkdir()
    (target / "COMMITTED").write_text("distributed-v2")
    (target / "state.json").write_text(json.dumps(TrainerState(optimizer_step=2).to_dict()))

    with pytest.raises(CheckpointError):
        verify_committed_checkpoint(target, optimizer_step=2, distributed=True)


def test_verify_distributed_rejects_accelerate_tree_without_state_file(tmp_path: Path):
    target = _save(tmp_path, 2)
    portable = target / "portable"
    portable.mkdir()
    for name in ("COMMITTED", "model.safetensors", "training.pt", "rng.json", "rng.safetensors"):
        (portable / name).write_bytes(b"x")
    (target / "distributed.json").write_text("{}")
    (target / "accelerate" / "empty").mkdir(parents=True)

    with pytest.raises(CheckpointError, match="Accelerate state"):
        verify_committed_checkpoint(target, optimizer_step=2, distributed=True)
