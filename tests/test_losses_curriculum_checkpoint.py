import json
from pathlib import Path

import pytest
import torch
from torch import nn

from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.losses.objectives import (
    composition_consistency_loss,
    confidence_alignment_loss,
    conflict_listwise_loss,
    distillation_loss,
    irrelevant_suppression_loss,
    language_modeling_loss,
    listwise_rerank_loss,
    protection_loss,
    routing_loss,
    sparse_regularization_loss,
)
import doc_to_atom.training.checkpoint as checkpoint_module
from doc_to_atom.training.curriculum import stage_for_step


def test_every_loss_is_finite():
    logits = torch.randn(2, 4, 7)
    teacher = torch.randn(2, 4, 7)
    labels = torch.randint(0, 7, (2, 4))
    scores = torch.randn(3)
    gold = torch.tensor([True, False, False])
    distractor = torch.tensor([False, True, False])
    bank = MemoryBank(
        keys=torch.randn(3, 2),
        factors_a={"k": torch.randn(3, 2, 3)},
        factors_b={"k": torch.randn(3, 4, 2)},
        mask_logits=torch.randn(3, 1),
        atoms=[],
    )
    composed = MemoryComposer()(bank, torch.tensor([0, 1]), torch.tensor([0.4, 0.6]))
    losses = [
        language_modeling_loss(logits, labels),
        distillation_loss(logits, teacher, torch.ones(2, 4, dtype=torch.bool)),
        routing_loss(scores, gold, distractor),
        listwise_rerank_loss(scores, gold),
        irrelevant_suppression_loss(composed),
        protection_loss(composed),
        sparse_regularization_loss(torch.tensor([0.2, 0.5]), 0.25),
        composition_consistency_loss(logits, teacher),
        conflict_listwise_loss(scores, ["g", "g", None], gold),
        confidence_alignment_loss(torch.tensor([0.4, 0.6]), torch.tensor([0.8, 0.5])),
    ]
    assert all(torch.isfinite(loss) for loss in losses)


def test_curriculum_switches_exact_boundaries():
    assert (stage_for_step(0).top_k, stage_for_step(1999).top_k) == (1, 1)
    assert stage_for_step(2000).top_k == 4
    assert stage_for_step(6000).top_k == 8
    assert stage_for_step(9000).strengthen_sparse


def test_checkpoint_save_load(tmp_path: Path):
    module = nn.Linear(3, 2)
    original = module.weight.detach().clone()
    path = tmp_path / "checkpoint"
    checkpoint_module.save_checkpoint({"module": module}, path, step=17)
    with torch.no_grad():
        module.weight.zero_()
    assert checkpoint_module.load_checkpoint({"module": module}, path) == 17
    assert torch.allclose(module.weight, original)
    assert (path / "COMMITTED").is_file()


def test_checkpoint_v2_round_trips_complete_trainer_state(tmp_path: Path):
    module = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cpu")
    state = checkpoint_module.TrainerState(
        epoch=3,
        record_cursor=11,
        micro_step=21,
        optimizer_step=7,
        curriculum_step=2007,
        sampler_state={"epoch": 3, "seed": 123, "cursor": 11},
        best_validation=0.75,
        bad_validations=2,
        early_stopped=False,
        config_fingerprint="config-hash",
        dataset_fingerprint="dataset-hash",
        model_fingerprint="model-hash",
        prompt_fingerprint="prompt-hash",
        wandb_run_id="wandb-id",
        wandb_mode="offline",
    )
    checkpoint = tmp_path / "step-000007"

    checkpoint_module.save_training_checkpoint(
        {"module": module},
        checkpoint,
        state=state,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    restored = checkpoint_module.load_training_checkpoint(
        {"module": module},
        checkpoint,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_fingerprints={
            "config": "config-hash",
            "dataset": "dataset-hash",
            "model": "model-hash",
            "prompt": "prompt-hash",
        },
    )

    assert restored == state
    metadata = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["optimizer_step"] == 7


def test_checkpoint_rejects_uncommitted_and_incompatible_state(tmp_path: Path):
    module = nn.Linear(2, 1)
    checkpoint = tmp_path / "step"
    state = checkpoint_module.TrainerState(config_fingerprint="original")
    checkpoint_module.save_training_checkpoint({"module": module}, checkpoint, state=state)
    (checkpoint / "COMMITTED").unlink()

    with pytest.raises(checkpoint_module.CheckpointError, match="not committed"):
        checkpoint_module.load_training_checkpoint({"module": module}, checkpoint)

    (checkpoint / "COMMITTED").write_text("\n", encoding="utf-8")
    with pytest.raises(checkpoint_module.IncompatibleCheckpointError, match="config"):
        checkpoint_module.load_training_checkpoint(
            {"module": module}, checkpoint, expected_fingerprints={"config": "changed"}
        )


def test_legacy_checkpoint_requires_explicit_weights_only_migration(tmp_path: Path):
    source = nn.Linear(2, 1)
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {"step": 9, "modules": {"module": source.state_dict()}, "optimizer": {"unsafe": True}},
        legacy,
    )
    destination = nn.Linear(2, 1)

    with pytest.raises(checkpoint_module.CheckpointError, match="explicit weights-only"):
        checkpoint_module.load_training_checkpoint({"module": destination}, legacy)
    restored = checkpoint_module.load_training_checkpoint(
        {"module": destination}, legacy, allow_legacy_weights=True
    )

    assert restored.optimizer_step == 9
    assert restored.exact_resume is False
    assert torch.equal(destination.weight, source.weight)
