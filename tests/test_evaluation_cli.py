import json
import sys
from types import SimpleNamespace

import pytest

from doc_to_atom.atomization.models import Atom, QuestionAnnotation, StructuredRecord
from doc_to_atom.evaluation import cli


def _record():
    atom = Atom(
        atom_id="a1",
        semantic_type="fact_claim",
        content="Copper statue",
        source_span="Copper statue",
        retrieval_text="Copper statue",
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1,
    )
    return StructuredRecord(
        record_id="r1",
        dataset="squad",
        split="validation",
        context="Copper statue",
        atoms=[atom],
        qa_pairs=[
            QuestionAnnotation(
                question_id="q1",
                question="What?",
                answers=("a copper statue", "Copper statue"),
                answer_provenance="source",
                question_type="single_hop",
                gold_atom_ids=["a1"],
                relevant_atom_ids=["a1"],
            ),
            QuestionAnnotation(
                question_id="generated_irrelevant_0",
                question="Who built it?",
                answers=(),
                answer_provenance="generated_irrelevant",
                question_type="irrelevant",
                is_irrelevant=True,
                is_generated_irrelevant=True,
            ),
        ],
    )


def test_score_rows_uses_all_source_references_and_separate_refusal_denominator():
    rows = [
        {"prediction": "Copper statue", "gold": ["wrong", "Copper statue"], "is_irrelevant": False},
        {"prediction": "unanswerable", "gold": [], "is_irrelevant": True},
    ]

    summary = cli.score_rows(rows)

    assert summary["answerable_count"] == 1
    assert summary["irrelevant_count"] == 1
    assert summary["f1"] == pytest.approx(1.0)
    assert summary["rouge_l"] == pytest.approx(1.0)
    assert summary["refusal_f1"] == pytest.approx(1.0)


def test_generate_rows_covers_answerable_and_irrelevant_with_source_gold():
    class FakeEngine:
        def __init__(self, *args):
            pass

        def answer(self, question, bank, top_k, max_new_tokens):
            return {
                "prediction": "unanswerable" if question.startswith("Who") else "Copper statue",
                "selected_atom_ids": ["a1"],
                "ranked_atom_ids": ["a1"],
                "routing_weights": [1.0],
                "composed_adapter_norm": 0.5,
            }

    system = SimpleNamespace(
        base_model=object(), encoder=object(), router=object(), composer=object(), injector=object()
    )
    rows = cli.generate_rows(
        system,
        object(),
        [_record()],
        {"evaluation": {"max_new_tokens": 8, "routing_recall_k": 1}},
        engine_factory=FakeEngine,
        memory_builder=lambda *args: object(),
    )

    assert [row["question_id"] for row in rows] == ["q1", "generated_irrelevant_0"]
    assert rows[0]["gold"] == ["a copper statue", "Copper statue"]
    assert rows[1]["gold"] == []
    assert rows[1]["is_irrelevant"] is True
    assert cli.score_rows(rows)["refusal_f1"] == pytest.approx(1.0)


def test_evaluation_cli_generates_predictions_and_metrics_from_checkpoint(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "resolved.yaml"
    source = tmp_path / "source.jsonl"
    data = tmp_path / "atoms.jsonl"
    checkpoint = tmp_path / "last.json"
    output = tmp_path / "predictions.jsonl"
    for path in (config_path, source, data, checkpoint):
        path.write_text("{}", encoding="utf-8")
    record = _record()
    fake_module = object()
    system = SimpleNamespace(
        encoder=SimpleNamespace(projection=fake_module),
        compiler=fake_module,
        router=fake_module,
        eval=lambda: None,
    )
    monkeypatch.setattr(cli, "validate_jsonl", lambda *args, **kwargs: {"errors": 0, "invalid_spans": 0})
    monkeypatch.setattr(cli, "read_jsonl", lambda path: [record])
    monkeypatch.setattr(cli, "load_config", lambda path: {"evaluation": {"max_new_tokens": 8}})
    monkeypatch.setattr(cli, "build_system", lambda *args, **kwargs: (system, object(), []))
    monkeypatch.setattr(cli, "_checkpoint_directory", lambda path: path)
    monkeypatch.setattr(cli, "load_training_checkpoint", lambda *args, **kwargs: SimpleNamespace(optimizer_step=36))
    monkeypatch.setattr(
        cli,
        "generate_rows",
        lambda *args: [
            {"prediction": "Copper statue", "gold": ["Copper statue"], "is_irrelevant": False},
            {"prediction": "unanswerable", "gold": [], "is_irrelevant": True},
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["doc2atom-evaluate", "--config", str(config_path), "--data", str(data),
         "--source", str(source), "--checkpoint", str(checkpoint), "--output", str(output)],
    )

    cli.main()

    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    metrics = json.loads((tmp_path / "metrics" / "evaluation_validation.json").read_text())
    assert metrics["checkpoint_step"] == 36
    assert metrics["irrelevant_count"] == 1
    assert json.loads(capsys.readouterr().out)["answerable_count"] == 1


def test_checkpoint_pointer_resolves_portable_distributed_weights(tmp_path):
    checkpoint = tmp_path / "step-00000036"
    checkpoint.mkdir()
    (checkpoint / "distributed.json").write_text("{}", encoding="utf-8")
    pointer = tmp_path / "best.json"
    pointer.write_text(json.dumps({"path": checkpoint.name}), encoding="utf-8")

    assert cli._checkpoint_directory(pointer) == checkpoint / "portable"
