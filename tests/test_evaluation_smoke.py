"""Bounded CPU train -> committed checkpoint -> evaluation artifact smoke."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_training_loop import ToyTokenizer, _record, _tiny_system
from doc_to_atom.atomization.models import QuestionAnnotation
from doc_to_atom.config import config_fingerprint, load_config
from doc_to_atom.training.checkpoint import IncompatibleCheckpointError, load_training_checkpoint
from doc_to_atom.training.loop import run_training
import doc_to_atom.evaluation.cli as cli


def test_local_train_checkpoint_evaluate_smoke(tmp_path: Path, monkeypatch):
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["runtime"]["checkpoint_upload"] = "none"
    config["training"].update({
        "max_epochs": 1, "max_steps": 1, "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
    })
    train = _record("train")
    validation = _record("validation")
    validation.qa_pairs.append(QuestionAnnotation(
        question_id="irrelevant-0", question="Unknown?", answers=(),
        answer_provenance="generated_irrelevant", question_type="irrelevant",
        is_irrelevant=True, is_generated_irrelevant=True,
    ))
    run_training(_tiny_system(), ToyTokenizer(), [train, validation], config, tmp_path)
    pointer = tmp_path / "checkpoints" / "last.json"
    path = cli._checkpoint_directory(pointer)
    state = load_training_checkpoint({}, path, expected_fingerprints={"config": config_fingerprint(config)}, restore_rng=False)
    assert state.exact_resume and state.optimizer_step == 1
    with pytest.raises(IncompatibleCheckpointError, match="config"):
        load_training_checkpoint({}, path, expected_fingerprints={"config": "wrong"}, restore_rng=False)

    class Engine:
        def __init__(self, *args):
            pass

        def answer(self, question, bank, **kwargs):
            return {
                "prediction": "unanswerable" if question == "Unknown?" else "Alpha",
                "raw_generation": "unanswerable" if question == "Unknown?" else "Alpha",
                "stop_reason": "eos", "selected_atom_ids": ["a0"],
                "ranked_atom_ids": ["a0"], "routing_weights": [1.0],
                "composed_adapter_norm": 0.0,
            }

    original_generate = cli.generate_rows
    monkeypatch.setattr(cli, "generate_rows", lambda system, tokenizer, records, config: original_generate(
        system, tokenizer, records, config,
        engine_factory=Engine,
        memory_builder=lambda system, tokenizer, record, config: SimpleNamespace(atoms=record.atoms),
    ))
    monkeypatch.setattr(cli, "validate_jsonl", lambda *args, **kwargs: {"errors": 0, "invalid_spans": 0})
    monkeypatch.setattr(cli, "read_jsonl", lambda path: [train, validation])
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "build_system", lambda *args, **kwargs: (_tiny_system(), ToyTokenizer(), []))
    config_path = tmp_path / "resolved.yaml"
    source_path = tmp_path / "source.jsonl"
    data_path = tmp_path / "atomized.jsonl"
    for file in (config_path, source_path, data_path):
        file.write_text("{}", encoding="utf-8")
    output = tmp_path / "predictions" / "validation.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "doc2atom-evaluate", "--config", str(config_path), "--data", str(data_path),
        "--source", str(source_path), "--checkpoint", str(pointer),
        "--output", str(output), "--split", "validation",
    ])

    cli.main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    metrics = json.loads((tmp_path / "metrics" / "evaluation_validation.json").read_text())
    assert len(rows) == 2 and {row["is_irrelevant"] for row in rows} == {False, True}
    assert metrics["checkpoint_step"] == 1
    assert metrics["f1"] == 1.0 and metrics["refusal_f1"] == 1.0
    assert (tmp_path / "reports" / "evaluation_validation.json").is_file()
