import importlib.util
import json
import sys
from pathlib import Path

import pytest

from doc_to_atom.atomization.prompts import DECOMPOSITION_SYSTEM, IRRELEVANT_SYSTEM, QUESTION_SYSTEM
from doc_to_atom.atomization.provenance import RawProviderResponse, write_raw_response_jsonl
from doc_to_atom.atomization.client import MiniMaxConfig
from doc_to_atom.atomization.models import AnnotationMeta, Atom, StructuredRecord

spec = importlib.util.spec_from_file_location(
    "annotation_experiment", Path(__file__).parents[1] / "scripts/run_annotation_experiment.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_prompt_explicitly_requires_disjoint_minimal_roles():
    assert "pairwise disjoint" in QUESTION_SYSTEM
    assert "An empty list is normal" in QUESTION_SYSTEM
    assert "bridge/comparison" in QUESTION_SYSTEM
    assert "must NOT also appear in supporting_atom_ids" in QUESTION_SYSTEM


def test_annotation_prompts_start_with_task_instructions():
    assert DECOMPOSITION_SYSTEM.startswith("Decompose the supplied document")
    assert IRRELEVANT_SYSTEM.startswith("Generate plausible questions")
    assert QUESTION_SYSTEM.startswith("Annotate each question")
    assert AnnotationMeta().prompt_provenance == ""


def test_metrics_detect_overlap_in_uncanonicalized_provider_output(tmp_path):
    path = tmp_path / "raw.jsonl"
    raw = RawProviderResponse.capture(
        record_id="r", stage="question_annotation", attempt=1, provider="minimax", model="test",
        content="""<question_annotations><question_annotation>
        <question_id>q</question_id><question_type>single_hop</question_type>
        <gold_atom_ids><atom_id>a</atom_id></gold_atom_ids>
        <supporting_atom_ids><atom_id>a</atom_id></supporting_atom_ids>
        <distractor_atom_ids/></question_annotation></question_annotations>""",
        raw_body=json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 20,
                             "completion_tokens_details": {"reasoning_tokens": 15}}}),
        latency_ms=123,
    )
    write_raw_response_jsonl(raw, path)
    metrics = experiment.raw_metrics(path)
    assert metrics["provider_overlap_questions"] == 1
    assert metrics["provider_overlap_rate"] == 1
    assert metrics["total_tokens"] == 30
    assert metrics["reasoning_tokens"] == 15
    assert metrics["summed_request_latency_ms"] == 123


def result(effort, *, accepted=True, tokens=10, overlaps=0):
    return {"reasoning_effort": effort, "validation": {"accepted": accepted}, "failures": [],
            "provider_overlap_questions": overlaps, "provider_role_parse_notices": 0,
            "total_tokens": tokens, "summed_request_latency_ms": 100}


def test_selection_requires_quality_before_token_savings():
    selected = experiment.choose_effort([
        result("low", accepted=False, tokens=1),
        result("medium", tokens=2, overlaps=1), result("max", tokens=100),
    ])
    assert selected["reasoning_effort"] == "max"


def test_selection_blocks_production_if_no_effort_passes():
    with pytest.raises(ValueError, match="production is blocked"):
        experiment.choose_effort([result("low", accepted=False)])


def test_source_gate_rejects_missing_records(tmp_path):
    (tmp_path / "atoms.jsonl").touch()
    (tmp_path / "raw_source.jsonl").write_text(json.dumps({
        "id": "r", "context": "A", "split": "train",
        "questions": [{"id": "q", "question": "What?", "answers": ["A"]}],
    }) + "\n", encoding="utf-8")
    report = experiment.source_report(tmp_path)
    assert not report["accepted"]
    assert report["errors"] > 0


def test_full_workflow_keeps_validation_out_of_pilot_and_reuses_paid_records(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    rows = [{"id": record_id, "context": "Paris", "dataset": "squad", "split": split,
             "questions": [{"id": "q", "question": "Which city?", "answers": ["Paris"]}]}
            for record_id, split in (("t1", "train"), ("t2", "train"), ("v1", "validation"))]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    calls = []

    class FakeAtomizer:
        def __init__(self, config):
            self.config = config

        def atomize(self, record_id, context, questions, dataset, split):
            calls.append((self.config.reasoning_effort, record_id))
            self.response_sink(RawProviderResponse.capture(
                record_id=record_id, stage="decomposition", attempt=1,
                provider="minimax", model="test", content="<atoms/>",
                raw_body=json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 10}}),
                latency_ms=100,
            ))
            return StructuredRecord(
                record_id=record_id, context=context, dataset=dataset, split=split,
                atoms=[Atom(atom_id="a", semantic_type="fact_claim", content=context,
                            source_span=context, retrieval_text=context, is_answer_bearing=True,
                            abstraction_level="evidence", confidence=1)],
                qa_pairs=[q.model_copy(update={"gold_atom_ids": ["a"], "relevant_atom_ids": ["a"]})
                          for q in questions],
            )

    monkeypatch.setattr(experiment, "MiniMaxAtomizer", FakeAtomizer)
    monkeypatch.setattr(experiment, "load_project_env", lambda: None)
    monkeypatch.setattr(experiment.MiniMaxConfig, "from_env", lambda: MiniMaxConfig(api_key="test"))
    root = tmp_path / "release"
    monkeypatch.setattr(sys, "argv", ["experiment", "--source", str(source),
                                    "--output-dir", str(root), "--workers", "1"])
    experiment.main()
    assert len(calls) == 11  # 5 efforts x 2 train + 1 validation; no duplicate pilot calls.
    assert sum(record_id == "v1" for _, record_id in calls) == 1
    assert json.loads((root / "release.json").read_text())["training_ready"]
    assert (root / "raw_source.jsonl").read_bytes() == source.read_bytes()
    assert json.loads((root / "selection.json").read_text())["validation_used_for_selection"] is False
    for split, expected in (("train", {"t1", "t2"}), ("validation", {"v1"})):
        folder = root / "production" / split
        atom_ids = {r.record_id for r in experiment.read_jsonl(folder / "atoms.jsonl")}
        raw_ids = {json.loads(s)["record_id"] for s in
                   (folder / "raw_responses.jsonl").read_text().splitlines()}
        source_ids = set(experiment.load_source_records(folder / "raw_source.jsonl"))
        assert atom_ids == raw_ids == source_ids == expected
        assert json.loads((folder / "validation.json").read_text())["accepted"]
    with pytest.raises(FileExistsError):
        experiment.main()
    assert len(calls) == 11  # Refuse existing destinations before any new paid calls.
