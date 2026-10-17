import json
from pathlib import Path

import pytest
import torch
from conftest import ToyLM

from doc_to_atom.atomization.models import Atom, QuestionAnnotation, StructuredRecord
from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.config import load_config
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector, discover_target_shapes
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.training.loop import run_training
import doc_to_atom.training.loop as loop_module
from doc_to_atom.training.system import TeacherStudentSystem
import doc_to_atom.training.trainer as trainer_module


class ToyTokenizer:
    def _ids(self, text):
        return [1] + [2 + ord(char) % 25 for char in text]

    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=True,
    ):
        if isinstance(texts, str):
            ids = self._ids(texts)
            if truncation and max_length:
                ids = ids[:max_length]
            if return_tensors == "pt":
                tensor = torch.tensor([ids])
                return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
            return {"input_ids": ids}
        sequences = [
            self._ids(text)[:max_length] if max_length else self._ids(text) for text in texts
        ]
        width = max(map(len, sequences))
        ids = torch.zeros(len(sequences), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, : len(sequence)] = torch.tensor(sequence)
            mask[index, : len(sequence)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _record(split, suffix=""):
    atoms = [
        Atom(
            atom_id="a0",
            semantic_type="fact_claim",
            content="Alpha",
            source_span="Alpha",
            retrieval_text="Alpha",
            is_answer_bearing=True,
            abstraction_level="evidence",
            confidence=0.9,
        )
    ]
    qa = QuestionAnnotation(
        question_id=f"q0{suffix}",
        question="What?",
        answer="Alpha",
        question_type="single_hop",
        gold_atom_ids=["a0"],
        relevant_atom_ids=["a0"],
    )
    return StructuredRecord(
        record_id=f"r-{split}{suffix}", split=split, context="Alpha", atoms=atoms, qa_pairs=[qa]
    )


def test_tiny_teacher_student_training_loop(tmp_path: Path):
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    encoder = SharedAtomQueryEncoder(model, 8, 16, 4)
    compiler = AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2)
    router = TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32)
    composer = MemoryComposer()
    injector = DynamicLoRAInjector(model, targets, rank=2, alpha=4)
    system = TeacherStudentSystem(model, encoder, compiler, router, composer, injector)
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"]["max_epochs"] = 1
    config["training"]["gradient_accumulation_steps"] = 1
    result = run_training(
        system, ToyTokenizer(), [_record("train"), _record("validation")], config, tmp_path
    )
    assert result["steps"] == 1
    assert result["optimizer_steps"] == 1
    assert result["micro_steps"] == 1
    assert (tmp_path / "checkpoints" / "last.json").exists()
    assert torch.isfinite(torch.tensor(result["best_validation_lm"]))


def test_optimizer_scheduler_and_curriculum_advance_at_update_boundaries(tmp_path: Path):
    torch.manual_seed(11)
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    system = TeacherStudentSystem(
        model,
        SharedAtomQueryEncoder(model, 8, 16, 4),
        AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2),
        TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32),
        MemoryComposer(),
        DynamicLoRAInjector(model, targets, rank=2, alpha=4),
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update(
        {
            "max_epochs": 1,
            "max_steps": 2,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 0,
            "initial_curriculum_step": 0,
        }
    )

    result = run_training(
        system,
        ToyTokenizer(),
        [_record("train", "-0"), _record("train", "-1"), _record("train", "-2")],
        config,
        tmp_path,
    )

    events = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert result["micro_steps"] == 3
    assert result["optimizer_steps"] == result["steps"] == 2
    assert [event["optimizer_step"] for event in events] == [1, 2]
    assert [event["micro_step"] for event in events] == [2, 3]
    assert [event["curriculum_step"] for event in events] == [1, 2]
    assert [event["learning_rate"] for event in events] == [0.00005, 0.0]
    assert all(event["cuda_allocated_bytes"] >= 0 for event in events)
    assert all(event["cuda_reserved_bytes"] >= 0 for event in events)


def test_early_stopping_uses_min_delta_patience_and_nonfinite_policy() -> None:
    stopping = trainer_module.EarlyStopping(mode="min", patience=2, min_delta=0.1)

    assert stopping.observe(1.0).improved
    assert not stopping.observe(0.95).improved
    nan_result = stopping.observe(float("nan"))

    assert nan_result.nonfinite
    assert nan_result.should_stop
    assert stopping.best == 1.0
    assert stopping.bad_validations == 2


def test_early_stopping_warmup_observations_do_not_consume_patience() -> None:
    stopping = trainer_module.EarlyStopping(
        mode="min", patience=1, min_delta=0.0, warmup_observations=2
    )

    assert stopping.observe(1.0).improved
    assert not stopping.observe(1.1).should_stop
    assert stopping.bad_validations == 0
    assert stopping.observe(1.2).should_stop
    assert stopping.observations == 3


def test_early_stopping_metric_path_is_explicit() -> None:
    report = {"overall": {"losses": {"lm": 0.75}, "token_accuracy": 0.5}}

    assert loop_module._metric_value(report, "overall.losses.lm") == 0.75
    assert loop_module._metric_value(report, "overall.token_accuracy") == 0.5
    with pytest.raises(ValueError, match="missing early-stopping metric"):
        loop_module._metric_value(report, "overall.missing")


def test_resume_starts_at_next_unconsumed_record(tmp_path: Path) -> None:
    def build_system():
        model = ToyLM(depth=6)
        targets = discover_target_shapes(
            model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
        )
        return TeacherStudentSystem(
            model,
            SharedAtomQueryEncoder(model, 8, 16, 4),
            AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2),
            TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32),
            MemoryComposer(),
            DynamicLoRAInjector(model, targets, rank=2, alpha=4),
        )

    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update(
        {
            "max_epochs": 1,
            "max_steps": 2,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 0,
            "seed": 91,
        }
    )
    records = [_record("train", f"-{index}") for index in range(3)]
    torch.manual_seed(41)
    first_system = build_system()
    first = run_training(
        first_system,
        ToyTokenizer(),
        records,
        config,
        tmp_path,
        stop_after_optimizer_steps=1,
    )
    pointer = tmp_path / "checkpoints" / "last.json"
    state_path = tmp_path / "checkpoints" / json.loads(pointer.read_text())["path"] / "state.json"
    saved_state = json.loads(state_path.read_text())

    torch.manual_seed(41)
    resumed_system = build_system()
    resumed = run_training(
        resumed_system,
        ToyTokenizer(),
        records,
        config,
        tmp_path,
        resume_from=pointer,
    )

    assert first["micro_steps"] == 2
    assert saved_state["record_cursor"] == 2
    assert resumed["micro_steps"] == 3
    assert resumed["optimizer_steps"] == 2
    events = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert [event["micro_step"] for event in events] == [2, 3]


def test_partial_accumulation_window_preserves_gradient_scale(tmp_path: Path) -> None:
    def build(seed: int):
        torch.manual_seed(seed)
        model = ToyLM(depth=6)
        targets = discover_target_shapes(
            model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
        )
        return TeacherStudentSystem(
            model,
            SharedAtomQueryEncoder(model, 8, 16, 4),
            AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2),
            TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32),
            MemoryComposer(),
            DynamicLoRAInjector(model, targets, rank=2, alpha=4),
        )

    base = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    base["training"].update({"max_epochs": 1, "max_steps": 1, "warmup_steps": 0})
    system_one = build(57)
    config_one = {**base, "training": {**base["training"], "gradient_accumulation_steps": 1}}
    run_training(system_one, ToyTokenizer(), [_record("train")], config_one, tmp_path / "one")

    system_two = build(57)
    config_two = {**base, "training": {**base["training"], "gradient_accumulation_steps": 2}}
    run_training(system_two, ToyTokenizer(), [_record("train")], config_two, tmp_path / "two")

    one_event = json.loads((tmp_path / "one" / "train_log.jsonl").read_text().strip())
    two_event = json.loads((tmp_path / "two" / "train_log.jsonl").read_text().strip())
    assert two_event["gradient_norm"] == pytest.approx(one_event["gradient_norm"], rel=1e-6)
    assert all(
        torch.allclose(left, right, atol=1e-7, rtol=1e-6)
        for left, right in zip(
            system_one.encoder.projection.parameters(),
            system_two.encoder.projection.parameters(),
            strict=True,
        )
    )
