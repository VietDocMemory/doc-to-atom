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
from doc_to_atom.training.distributed import LocalTrainingRuntime
from doc_to_atom.training.system import TeacherStudentSystem
import doc_to_atom.training.trainer as trainer_module


def test_validation_evaluates_generated_irrelevant_after_training_question_cap(monkeypatch):
    record = _record("validation")
    record.qa_pairs.append(
        QuestionAnnotation(
            question_id="generated_irrelevant_0",
            question="Unknown?",
            answers=(),
            answer_provenance="generated_irrelevant",
            question_type="irrelevant",
            is_irrelevant=True,
            is_generated_irrelevant=True,
        )
    )
    seen = []
    monkeypatch.setattr(loop_module, "_memory", lambda *args: object())

    def fake_one_question(system, tokenizer, row, qa, bank, config, step):
        seen.append(qa.question_id)
        return None, 0.0, {
            "question_id": qa.question_id,
            "group": "generated_irrelevant" if qa.is_generated_irrelevant else "answerable",
            "losses": {"lm": 0.0},
            "total_loss": 0.0,
            "gold_recall_at_1": None,
            "gold_recall_at_k": None,
            "mrr": None,
            "routing_entropy": 0.0,
            "adapter_norm": 0.0,
            "token_accuracy": 0.0,
            "selected_atom_ids": [],
            "routing_weights": [],
            "prediction": "unanswerable",
            "target": "unanswerable",
        }

    monkeypatch.setattr(loop_module, "_one_question", fake_one_question)
    report = loop_module._validation_report(
        type("System", (), {"eval": lambda self: None})(),
        None,
        [record],
        {"data": {"max_questions_per_sample": 1}},
        1,
    )

    assert seen == ["q0", "generated_irrelevant_0"]
    assert report["generated_irrelevant"]["count"] == 1


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


def _tiny_system():
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


def test_checkpoint_retention_does_not_save_each_nonimproving_validation(
    tmp_path: Path, monkeypatch
):
    class Tracker:
        run_id = "test-run"
        mode = "disabled"

        def __init__(self):
            self.uploads = []

        def log_metrics(self, values, *, step):
            pass

        def log_checkpoint(self, path):
            self.uploads.append(("full", Path(path).name))

        def log_best_weights(self, path, *, optimizer_step):
            self.uploads.append(("best", Path(path).name, optimizer_step))

    tracker = Tracker()
    values = iter((1.0, 1.1, 1.2, 1.3))
    monkeypatch.setattr(
        loop_module, "_validation_report",
        lambda *args: {"overall": {"losses": {"lm": next(values)}}},
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update({
        "max_epochs": 1, "max_steps": 4, "gradient_accumulation_steps": 1,
        "validation_every_steps": 1, "checkpoint_every_steps": 3,
        "early_stopping_patience": 10,
    })

    result = run_training(
        _tiny_system(), ToyTokenizer(),
        [_record("train", f"-{index}") for index in range(4)] + [_record("validation")],
        config, tmp_path, tracker=tracker,
    )

    checkpoints = tmp_path / "checkpoints"
    assert result["optimizer_steps"] == 4
    assert sorted(path.name for path in checkpoints.glob("step-*") if path.is_dir()) == [
        "step-00000001", "step-00000004"
    ]
    assert json.loads((checkpoints / "best.json").read_text())["path"] == "step-00000001"
    assert json.loads((checkpoints / "last.json").read_text())["path"] == "step-00000004"
    from doc_to_atom.training.checkpoint import load_training_checkpoint

    assert load_training_checkpoint({}, checkpoints / "step-00000001", restore_rng=False).optimizer_step == 1
    last = load_training_checkpoint({}, checkpoints / "step-00000004", restore_rng=False)
    assert last.optimizer_step == 4
    assert last.bad_validations == 3
    assert tracker.uploads == [("best", "step-00000001", 1)]


def test_early_stopping_keeps_best_and_final_resumable_last(tmp_path: Path, monkeypatch):
    class Tracker:
        run_id = "test-run"
        mode = "disabled"

        def __init__(self):
            self.uploads = 0

        def log_metrics(self, values, *, step):
            pass

        def log_best_weights(self, path, *, optimizer_step):
            self.uploads += 1

    tracker = Tracker()
    values = iter((1.0, 2.0, 3.0))
    monkeypatch.setattr(
        loop_module, "_validation_report",
        lambda *args: {"overall": {"losses": {"lm": next(values)}}},
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update({
        "max_epochs": 1, "max_steps": 3, "gradient_accumulation_steps": 1,
        "validation_every_steps": 1, "checkpoint_every_steps": 100,
        "early_stopping_patience": 1,
    })

    result = run_training(
        _tiny_system(), ToyTokenizer(),
        [_record("train", f"-{index}") for index in range(3)] + [_record("validation")],
        config, tmp_path, tracker=tracker,
    )

    checkpoints = tmp_path / "checkpoints"
    assert result["early_stopped"] and result["optimizer_steps"] == 2
    assert json.loads((checkpoints / "best.json").read_text())["path"] == "step-00000001"
    last_path = checkpoints / json.loads((checkpoints / "last.json").read_text())["path"]
    from doc_to_atom.training.checkpoint import load_training_checkpoint

    state = load_training_checkpoint({}, last_path, restore_rng=False)
    assert state.exact_resume and state.early_stopped
    assert state.optimizer_step == 2 and state.bad_validations == 1

    resumed = run_training(
        _tiny_system(), ToyTokenizer(),
        [_record("train", f"-{index}") for index in range(3)] + [_record("validation")],
        config, tmp_path, resume_from=checkpoints / "last.json", tracker=tracker,
    )
    assert resumed["early_stopped"] and resumed["optimizer_steps"] == 2
    assert json.loads((checkpoints / "last.json").read_text())["path"] == "step-00000002"
    assert tracker.uploads == 1


def test_intentional_mid_run_stop_keeps_last_but_defers_best_upload(tmp_path: Path, monkeypatch):
    class Tracker:
        run_id = "test-run"
        mode = "disabled"

        def __init__(self):
            self.uploads = 0

        def log_metrics(self, values, *, step):
            pass

        def log_best_weights(self, path, *, optimizer_step):
            self.uploads += 1

    tracker = Tracker()
    monkeypatch.setattr(
        loop_module, "_validation_report",
        lambda *args: {"overall": {"losses": {"lm": 1.0}}},
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update({
        "max_epochs": 1, "max_steps": 3, "gradient_accumulation_steps": 1,
        "validation_every_steps": 1, "early_stopping_patience": 10,
    })

    result = run_training(
        _tiny_system(), ToyTokenizer(),
        [_record("train", f"-{index}") for index in range(3)] + [_record("validation")],
        config, tmp_path, tracker=tracker, stop_after_optimizer_steps=1,
    )

    assert result["optimizer_steps"] == 1
    assert (tmp_path / "checkpoints" / "last.json").is_file()
    assert tracker.uploads == 0


def test_terminal_policy_uploads_best_and_last_once_without_touching_resume_state(tmp_path: Path, monkeypatch):
    class Tracker:
        run_id = "test-run"
        mode = "offline"

        def __init__(self):
            self.uploads = []

        def log_metrics(self, values, *, step):
            pass

        def log_best_weights(self, path, *, optimizer_step):
            self.uploads.append(("best", optimizer_step))
            return {"status": "offline_pending", "checkpoint": Path(path).name, "optimizer_step": optimizer_step}

        def log_last_checkpoint(self, path, *, optimizer_step):
            self.uploads.append(("last", optimizer_step))
            assert (Path(path) / "state.json").is_file()
            return {"status": "offline_pending", "checkpoint": Path(path).name, "optimizer_step": optimizer_step}

    tracker = Tracker()
    monkeypatch.setattr(
        loop_module, "_validation_report",
        lambda *args: {"overall": {"losses": {"lm": 1.0}}},
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["runtime"]["checkpoint_upload"] = "best_weights_and_last_full"
    config["training"].update({
        "max_epochs": 1, "max_steps": 1, "gradient_accumulation_steps": 1,
        "validation_every_steps": 1,
    })
    records = [_record("train"), _record("validation")]

    run_training(_tiny_system(), ToyTokenizer(), records, config, tmp_path, tracker=tracker)
    pointer = tmp_path / "checkpoints" / "last.json"
    from doc_to_atom.training.checkpoint import load_training_checkpoint
    state = load_training_checkpoint(
        {}, tmp_path / "checkpoints" / json.loads(pointer.read_text())["path"],
        restore_rng=False,
    )
    assert state.exact_resume and state.optimizer_step == 1
    assert tracker.uploads == [("best", 1), ("last", 1)]
    assert json.loads((tmp_path / "manifests" / "last_checkpoint_upload.json").read_text())["status"] == "offline_pending"

    run_training(
        _tiny_system(), ToyTokenizer(), records, config, tmp_path,
        resume_from=pointer, tracker=tracker,
    )
    assert tracker.uploads == [("best", 1), ("last", 1)]


def test_new_upload_policy_early_stop_preserves_resume_after_upload_error(tmp_path: Path, monkeypatch):
    class Tracker:
        run_id = "test-run"
        mode = "online"

        def __init__(self):
            self.last_uploads = []

        def log_metrics(self, values, *, step):
            pass

        def log_best_weights(self, path, *, optimizer_step):
            raise ConnectionError("simulated upload failure")

        def log_last_checkpoint(self, path, *, optimizer_step):
            self.last_uploads.append(optimizer_step)
            return {"status": "uploaded", "checkpoint": Path(path).name, "optimizer_step": optimizer_step}

    values = iter((1.0, 2.0))
    monkeypatch.setattr(
        loop_module, "_validation_report",
        lambda *args: {"overall": {"losses": {"lm": next(values)}}},
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["runtime"]["checkpoint_upload"] = "best_weights_and_last_full"
    config["training"].update({
        "max_epochs": 1, "max_steps": 3, "gradient_accumulation_steps": 1,
        "validation_every_steps": 1, "checkpoint_every_steps": 100,
        "early_stopping_patience": 1,
    })
    tracker = Tracker()

    result = run_training(
        _tiny_system(), ToyTokenizer(),
        [_record("train", f"-{i}") for i in range(3)] + [_record("validation")],
        config, tmp_path, tracker=tracker,
    )

    checkpoint_dir = tmp_path / "checkpoints"
    last_pointer = json.loads((checkpoint_dir / "last.json").read_text())
    best_pointer = json.loads((checkpoint_dir / "best.json").read_text())
    from doc_to_atom.training.checkpoint import load_training_checkpoint
    state = load_training_checkpoint({}, checkpoint_dir / last_pointer["path"], restore_rng=False)
    assert result["early_stopped"] and state.exact_resume and state.early_stopped
    assert last_pointer["optimizer_step"] == 2 and best_pointer["optimizer_step"] == 1
    assert tracker.last_uploads == [2]
    assert not (tmp_path / "manifests" / "best_model_upload.json").exists()
    assert json.loads((tmp_path / "manifests" / "last_checkpoint_upload.json").read_text())["status"] == "uploaded"


def test_distributed_main_rank_retains_best_last_after_collective_save(
    tmp_path: Path, monkeypatch
):
    class SingleRankZeROHarness(LocalTrainingRuntime):
        backend = "deepspeed_zero2"
        world_size = 2
        rank = 0
        is_main_process = True
        is_distributed = True

        def __init__(self):
            super().__init__(1)
            self.barriers = 0

        def wait_for_everyone(self):
            self.barriers += 1

        def collective_errors(self, error):
            return [error, None]

        def save_state(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "rank-0.state").write_bytes(b"accelerate state")

    values = iter((1.0, 2.0, 3.0))
    monkeypatch.setattr(
        loop_module, "_validation_report",
        lambda *args: {"overall": {"losses": {"lm": next(values)}}},
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update({
        "max_epochs": 1, "max_steps": 3, "gradient_accumulation_steps": 1,
        "validation_every_steps": 1, "checkpoint_every_steps": 2,
        "early_stopping_patience": 10,
    })
    harness = SingleRankZeROHarness()

    result = run_training(
        _tiny_system(), ToyTokenizer(),
        [_record("train", f"-{index}") for index in range(6)] + [_record("validation")],
        config, tmp_path, runtime=harness,
    )

    checkpoints = tmp_path / "checkpoints"
    assert result["optimizer_steps"] == 3
    assert sorted(path.name for path in checkpoints.glob("step-*") if path.is_dir()) == [
        "step-00000001", "step-00000003"
    ]
    assert (checkpoints / "step-00000003" / "accelerate" / "rank-0.state").is_file()
    assert harness.barriers >= 3 * 3


def test_checkpoint_upload_none_keeps_local_best_without_tracker_upload(
    tmp_path: Path, monkeypatch
):
    class Tracker:
        run_id = "test-run"
        mode = "disabled"

        def __init__(self):
            self.uploads = 0

        def log_metrics(self, values, *, step):
            pass

        def log_best_weights(self, path, *, optimizer_step):
            self.uploads += 1

    tracker = Tracker()
    monkeypatch.setattr(
        loop_module, "_validation_report",
        lambda *args: {"overall": {"losses": {"lm": 1.0}}},
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update({
        "max_epochs": 1, "max_steps": 1, "gradient_accumulation_steps": 1,
        "early_stopping_patience": 2,
    })
    config["runtime"]["checkpoint_upload"] = "none"

    run_training(
        _tiny_system(), ToyTokenizer(), [_record("train"), _record("validation")],
        config, tmp_path, tracker=tracker,
    )

    assert tracker.uploads == 0
    assert (tmp_path / "checkpoints" / "best.json").is_file()


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
