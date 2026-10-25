import json
import hashlib
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


def test_score_rows_keeps_full_prediction_as_primary_and_first_line_as_diagnostic():
    rows = [{
        "question_id": "q1",
        "prediction": "Paris\nExplanation: Paris is the city.",
        "gold": ["Lyon", "Paris"],
        "is_irrelevant": False,
    }]

    summary = cli.score_rows(rows)

    assert summary["f1"] == summary["raw_f1"]
    assert 0 < summary["f1"] < 1
    assert summary["first_line_f1"] == 1.0
    assert summary["exact_match"] == 0.0
    assert summary["scoring_protocol"] == "squad_v1_multiref"


def test_score_rows_refusal_confusion_matrix_and_false_rates():
    rows = [
        {"question_id": "a1", "prediction": "Paris", "gold": ["Paris"], "is_irrelevant": False},
        {"question_id": "a2", "prediction": "unanswerable.", "gold": ["Lyon"], "is_irrelevant": False},
        {"question_id": "i1", "prediction": "unanswerable", "gold": [], "is_irrelevant": True},
        {"question_id": "i2", "prediction": "Lyon", "gold": [], "is_irrelevant": True},
    ]

    summary = cli.score_rows(rows)

    assert summary["refusal_confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert summary["refusal_precision"] == 0.5
    assert summary["refusal_recall"] == 0.5
    assert summary["refusal_f1"] == 0.5
    assert summary["false_refusal_rate"] == 0.5
    assert summary["false_answer_rate"] == 0.5


def test_score_rows_refusal_zero_denominators_are_explicit():
    summary = cli.score_rows([
        {"question_id": "a", "prediction": "Paris", "gold": ["Paris"], "is_irrelevant": False},
    ])
    assert summary["refusal_precision"] is None
    assert summary["refusal_recall"] is None
    assert summary["refusal_f1"] is None
    assert summary["false_refusal_rate"] == 0.0
    assert summary["false_answer_rate"] is None


def test_routing_scores_expose_rank_cutoffs_and_saturation():
    scores = cli.routing_scores(
        ["other", "gold-1", "other-2", "gold-2", "other-3"],
        ["gold-1", "gold-2"],
        candidate_count=5,
    )

    assert scores["recall_at_1"] == 0.0
    assert scores["recall_at_4"] == 1.0
    assert scores["recall_at_8"] == 1.0
    assert scores["mrr"] == 0.5
    assert scores["routing_saturation_at_8"] == 1.0


def test_score_rows_does_not_infer_new_recall_from_legacy_recall_at_k():
    rows = [{
        "question_id": "legacy",
        "prediction": "Copper statue",
        "gold": ["Copper statue"],
        "is_irrelevant": False,
        "ranked_atom_ids": ["a1"],
        "recall_at_k": 1.0,
        "mrr": 1.0,
    }]

    summary = cli.score_rows(rows)

    assert summary["recall_at_1"] is None
    assert summary["recall_at_4"] is None
    assert summary["recall_at_8"] is None
    assert summary["mrr"] is None


def test_legacy_enrichment_requires_exact_source_fields_and_adds_gold_ids():
    record = _record()
    rows = [
        {"record_id": "r1", "split": "validation", "question_id": "q1", "question": "What?",
         "prediction": "Copper statue", "gold": ["a copper statue", "Copper statue"],
         "is_irrelevant": False, "ranked_atom_ids": ["a1"]},
        {"record_id": "r1", "split": "validation", "question_id": "generated_irrelevant_0",
         "question": "Who built it?", "prediction": "unanswerable", "gold": [],
         "is_irrelevant": True, "ranked_atom_ids": ["a1"]},
    ]

    cli.enrich_rows(rows, [record])

    assert rows[0]["gold_atom_ids"] == ["a1"]
    assert rows[0]["atom_count"] == 1
    assert rows[0]["candidate_count"] == 1
    assert cli.score_rows(rows)["recall_at_1"] == 1.0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "changed_gold"])
def test_legacy_enrichment_rejects_incomplete_or_mutated_questions(mutation):
    record = _record()
    rows = [
        {"record_id": "r1", "split": "validation", "question_id": qa.question_id,
         "question": qa.question, "gold": list(qa.answers), "is_irrelevant": qa.is_irrelevant,
         "prediction": "answer", "ranked_atom_ids": ["a1"]}
        for qa in record.qa_pairs
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "unknown":
        rows[0]["question_id"] = "not-in-source"
    else:
        rows[0]["gold"] = ["Changed"]

    with pytest.raises(ValueError):
        cli.enrich_rows(rows, [record])


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
    output = tmp_path / "predictions" / "validation.jsonl"
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
    config = {
        "model": {"flash_attention_2": True},
        "evaluation": {"max_new_tokens": 8},
    }
    checkpoint_call = {}

    def build_system_with_sdpa_fallback(runtime_config, **kwargs):
        runtime_config["model"]["flash_attention_2"] = False
        runtime_config["model"]["attn_implementation"] = "sdpa"
        return system, object(), []

    def load_checkpoint(*args, **kwargs):
        checkpoint_call.update(kwargs)
        return SimpleNamespace(optimizer_step=36)

    monkeypatch.setattr(cli, "validate_jsonl", lambda *args, **kwargs: {"errors": 0, "invalid_spans": 0})
    monkeypatch.setattr(cli, "read_jsonl", lambda path: [record])
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "build_system", build_system_with_sdpa_fallback)
    monkeypatch.setattr(cli, "_checkpoint_directory", lambda path: path)
    monkeypatch.setattr(cli, "load_training_checkpoint", load_checkpoint)
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
    report = json.loads((tmp_path / "reports" / "evaluation_validation.json").read_text())
    assert metrics["checkpoint_step"] == 36
    assert metrics["irrelevant_count"] == 1
    assert metrics["source_fingerprint"] == hashlib.sha256(b"{}").hexdigest()
    assert metrics["atomized_fingerprint"] == hashlib.sha256(b"{}").hexdigest()
    assert metrics["generation_protocol"] == "short_answer_v1"
    assert report["settings"]["max_new_tokens"] == 8
    assert report["predictions_path"] == str(output)
    prediction = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert prediction["checkpoint_step"] == 36
    assert prediction["source_fingerprint"] == metrics["source_fingerprint"]
    assert json.loads(capsys.readouterr().out)["answerable_count"] == 1
    expected = checkpoint_call["expected_fingerprints"]["config"]
    assert cli.legacy_config_fingerprint(
        {"model": {"flash_attention_2": True}, "evaluation": {"max_new_tokens": 8}}
    ) in expected


def test_evaluation_cli_does_not_log_wandb_when_disabled(tmp_path, monkeypatch):
    class ForbiddenWandb:
        def init(self, **kwargs):
            raise AssertionError("W&B must remain disabled for local score-only")

    monkeypatch.setitem(sys.modules, "wandb", ForbiddenWandb())
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "question_id": "q1", "prediction": "Paris", "gold": ["Paris"],
        "is_irrelevant": False,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["doc2atom-evaluate", "--predictions", str(predictions)])

    cli.main()


def test_evaluation_cli_opt_in_offline_logs_metrics_and_files(tmp_path, monkeypatch):
    class FakeRun:
        def __init__(self):
            self.logged = []
            self.saved = []

        def log(self, metrics):
            self.logged.append(metrics)

        def save(self, path, **kwargs):
            self.saved.append(path)

        def finish(self):
            pass

    run = FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "question_id": "q1", "prediction": "Paris", "gold": ["Paris"],
        "is_irrelevant": False,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "doc2atom-evaluate", "--predictions", str(predictions),
        "--wandb-mode", "offline",
    ])

    cli.main()

    assert run.logged[0]["eval/f1"] == 1.0
    assert str(predictions) in run.saved


def test_checkpoint_pointer_resolves_portable_distributed_weights(tmp_path):
    checkpoint = tmp_path / "step-00000036"
    checkpoint.mkdir()
    (checkpoint / "distributed.json").write_text("{}", encoding="utf-8")
    pointer = tmp_path / "best.json"
    pointer.write_text(json.dumps({"path": checkpoint.name}), encoding="utf-8")

    assert cli._checkpoint_directory(pointer) == checkpoint / "portable"
