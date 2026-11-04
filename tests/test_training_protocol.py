import torch
import pytest

from doc_to_atom.config import config_fingerprint, load_and_resolve_config
from doc_to_atom.inference.answer_format import build_qa_prompt
from doc_to_atom.losses.objectives import distillation_loss, composition_consistency_loss
from doc_to_atom.training.protocol import (
    objective_protocol, training_texts, reduce_question_losses,
)


def test_missing_protocol_preserves_legacy_prompt_and_target_exactly():
    assert objective_protocol({}) == "legacy_v1"
    assert training_texts("Who?", "Context", "Alice", {}) == (
        "Document:\nContext\n\nQuestion: Who?\nAnswer:", "Question: Who?\nAnswer:", "Alice",
    )


def test_aligned_training_matches_generation_and_learns_line_termination():
    config = {"training": {"objective_protocol": "aligned_qa_v2"}}
    teacher, student, target = training_texts("Who?", "Context", "Alice", config)
    assert student == build_qa_prompt("Who?")
    assert teacher == build_qa_prompt("Who?", context="Context")
    assert target == "Alice\n"
    assert "Alice" not in student  # No reference leakage into the prompt.


def test_protocol_is_fingerprinted_and_unknown_protocol_is_refused():
    legacy = {"training": {"objective_protocol": "legacy_v1"}}
    aligned = {"training": {"objective_protocol": "aligned_qa_v2"}}
    assert config_fingerprint(legacy) != config_fingerprint(aligned)
    with pytest.raises(ValueError, match="objective_protocol"):
        objective_protocol({"training": {"objective_protocol": "typo"}})


def test_token_mean_kl_is_invariant_to_repeating_same_token_distribution():
    student = torch.tensor([[[2.0, -1.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 1.0, -1.0]]])
    def loss(length):
        return distillation_loss(student.repeat(1, length, 1), teacher.repeat(1, length, 1),
                                 torch.ones(1, length, dtype=torch.bool))
    assert torch.allclose(loss(1), loss(5))
    loss(5).backward()
    assert torch.isfinite(student.grad).all()
    assert torch.allclose(
        composition_consistency_loss(student.reshape(-1, 3), teacher.reshape(-1, 3)),
        composition_consistency_loss(student.repeat(1, 5, 1).reshape(-1, 3),
                                     teacher.repeat(1, 5, 1).reshape(-1, 3)),
    )


def test_group_balance_is_explicit_and_does_not_depend_on_probe_count():
    losses = [torch.tensor(6.0, requires_grad=True), torch.tensor(1.0), torch.tensor(1.0)]
    assert reduce_question_losses(losses, [False, True, True]).item() == pytest.approx(8 / 3)
    balanced = reduce_question_losses(losses, [False, True, True], "group_balanced")
    assert balanced.item() == 3.5
    balanced.backward()
    assert losses[0].grad.item() == 0.5
    assert reduce_question_losses(losses[:2], [False, True], "group_balanced").item() == 3.5


def test_new_profile_is_separate_and_does_not_change_old_run_config():
    config = load_and_resolve_config("configs/experiments/molab_aligned_qa_v2.yaml", require_runnable=True)
    assert config["training"]["objective_protocol"] == "aligned_qa_v2"
    assert config["training"]["question_loss_reduction"] == "group_balanced"
    assert config["training"]["early_stopping_metric"] == "answerable.losses.lm"
    assert config["training"]["warmup_steps"] == 40


def test_console_summary_preserves_full_saved_examples_without_flooding_stdout():
    from doc_to_atom.training.cli import console_report

    examples = [{"prediction": "tokenwise text", "target": "answer"}] * 300
    full = {"training": {"last_validation": {"overall": {"count": 300, "examples": examples}}}}
    brief = console_report(full)
    assert "examples" not in brief["training"]["last_validation"]["overall"]
    assert brief["training"]["last_validation"]["overall"]["examples_saved_in_report"] == 300
    assert len(full["training"]["last_validation"]["overall"]["examples"]) == 300
    assert console_report(full, full=True) is full
