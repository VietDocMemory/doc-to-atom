import json

import pytest

from doc_to_atom.atomization.models import (
    AnnotationMeta,
    Atom,
    QuestionAnnotation,
    StructuredRecord,
    TransformationEvent,
)
from doc_to_atom.data import io, preprocessors, semantic_validation


def _atom(atom_id="a1", text="Exact Gold"):
    return Atom(
        atom_id=atom_id,
        semantic_type="fact_claim",
        content=text,
        source_span=text,
        retrieval_text=text,
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1,
    )


def _record(dataset="synthetic_all", answers=("Exact Gold",), roles=None):
    roles = roles or {}
    qa = QuestionAnnotation(
        question_id="q1",
        question="Exact question?",
        answers=answers,
        answer_provenance="source",
        question_type="single_hop",
        gold_atom_ids=roles.get("gold", ["a1"]),
        supporting_atom_ids=roles.get("supporting", []),
        distractor_atom_ids=roles.get("distractor", []),
        relevant_atom_ids=roles.get("relevant", ["a1"]),
    )
    return StructuredRecord(
        record_id="r1",
        dataset=dataset,
        split="train",
        context="Exact Gold",
        atoms=[_atom()],
        qa_pairs=[qa],
    )


def _source(answers):
    return {
        "id": "r1",
        "dataset": "synthetic_all",
        "split": "train",
        "context": "Exact Gold",
        "questions": [
            {"id": "q1", "question": "Exact question?", "answers": list(answers)}
        ],
    }


def test_grounding_policy_is_owned_by_dataset_adapter(monkeypatch):
    all_adapter = preprocessors.DatasetAdapter(
        preprocess=lambda row: row,
        grounding_policy="all_answers_verbatim",
        grounding_note="test-only exact policy",
    )
    none_adapter = preprocessors.DatasetAdapter(
        preprocess=lambda row: row,
        grounding_policy="not_required",
        grounding_note="test-only no grounding",
    )
    monkeypatch.setitem(preprocessors.DATASET_ADAPTERS, "synthetic_all", all_adapter)
    monkeypatch.setitem(preprocessors.DATASET_ADAPTERS, "synthetic_none", none_adapter)
    not_grounded = _record(dataset="synthetic_all", answers=("Missing",))
    skipped = _record(dataset="synthetic_none", answers=("Missing",))

    assert "source_answer_not_grounded" in {
        issue["code"] for issue in semantic_validation.audit_record(not_grounded)
    }
    assert "source_answer_not_grounded" not in {
        issue["code"] for issue in semantic_validation.audit_record(skipped)
    }


def test_semantic_audit_reports_all_overlaps_without_mutating_input(monkeypatch):
    monkeypatch.setitem(
        preprocessors.DATASET_ADAPTERS,
        "synthetic_all",
        preprocessors.DatasetAdapter(
            preprocess=lambda row: row,
            grounding_policy="all_answers_verbatim",
            grounding_note="test",
        ),
    )
    record = _record(
        roles={
            "gold": ["a1"],
            "supporting": ["a1"],
            "distractor": ["a1"],
            "relevant": ["a1"],
        }
    )
    before = record.model_dump_json()

    issues = semantic_validation.audit_record(record)

    assert record.model_dump_json() == before
    codes = {issue["code"] for issue in issues}
    assert "gold_supporting_overlap" in codes
    assert "gold_distractor_overlap" in codes
    assert "supporting_distractor_overlap" in codes


def test_validation_separates_provider_and_canonical_unknown_counts(tmp_path, monkeypatch):
    monkeypatch.setitem(
        preprocessors.DATASET_ADAPTERS,
        "synthetic_all",
        preprocessors.DatasetAdapter(
            preprocess=lambda row: row,
            grounding_policy="all_answers_verbatim",
            grounding_note="test",
        ),
    )
    record = _record()
    record.annotation_meta = AnnotationMeta(
        transformations=[
            TransformationEvent(
                code="DROP_UNKNOWN_ATOM_REFERENCE",
                stage="canonicalization",
                message="q1: dropped unknown gold atoms",
                question_id="q1",
                details={"role": "gold", "removed_atom_ids": ["missing-1", "missing-2"]},
            )
        ]
    )
    atomized = tmp_path / "atoms.jsonl"
    source = tmp_path / "source.jsonl"
    atomized.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    source.write_text(json.dumps(_source(["Exact Gold"])) + "\n", encoding="utf-8")

    report = io.validate_jsonl(atomized, source_path=source, reject_span_mismatch=True)

    assert report["provider_unknown_atom_references"] == 2
    assert report["canonical_unknown_atom_references"] == 0
    assert report["source_answer_matches"] == 1
    assert report["source_answer_mismatches"] == 0
    assert report["answer_preservation_rate"] == 1.0
    assert report["answerable_empty_gold"] == 0
    assert report["grounding_failures"] == 0
    assert report["gold_supporting_overlap"] == 0
    assert report["gold_distractor_overlap"] == 0
    assert report["supporting_distractor_overlap"] == 0
    assert report["invalid_spans"] == 0
    assert report["parse_errors"] == 0
    assert report["transformation_counts"]["DROP_UNKNOWN_ATOM_REFERENCE"] == 1


def test_exact_source_answer_comparison_rejects_case_order_and_duplicate_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        preprocessors.DATASET_ADAPTERS,
        "synthetic_all",
        preprocessors.DatasetAdapter(
            preprocess=lambda row: row,
            grounding_policy="all_answers_verbatim",
            grounding_note="test",
        ),
    )
    atomized = tmp_path / "atoms.jsonl"
    source = tmp_path / "source.jsonl"
    atomized.write_text(
        _record(answers=("Exact Gold",)).model_dump_json() + "\n", encoding="utf-8"
    )
    source.write_text(
        json.dumps(_source(["Exact Gold", "Exact Gold", "exact gold"])) + "\n",
        encoding="utf-8",
    )

    report = io.validate_jsonl(atomized, source_path=source)

    assert report["source_answer_matches"] == 0
    assert report["source_answer_mismatches"] == 1
    assert report["answer_preservation_rate"] == 0.0


def test_irrelevant_contract_and_failed_parse_status_are_errors(monkeypatch):
    monkeypatch.setitem(
        preprocessors.DATASET_ADAPTERS,
        "synthetic_none",
        preprocessors.DatasetAdapter(
            preprocess=lambda row: row,
            grounding_policy="not_required",
            grounding_note="test",
        ),
    )
    record = _record(dataset="synthetic_none", answers=("unanswerable",))
    qa = record.qa_pairs[0]
    qa.is_irrelevant = True
    qa.question_type = "irrelevant"
    qa.supporting_atom_ids = ["a1"]
    record.annotation_meta.question_parse_status = "failed"

    report = semantic_validation.audit_records([record])
    codes = {issue["code"] for issue in report["issues"]}

    assert "irrelevant_has_answers" in codes
    assert "irrelevant_has_gold" in codes
    assert "irrelevant_has_supporting" in codes
    assert report["parse_errors"] == 1


def test_semantic_audit_rejects_duplicate_and_unknown_question_ids(monkeypatch):
    monkeypatch.setitem(
        preprocessors.DATASET_ADAPTERS,
        "synthetic_all",
        preprocessors.DatasetAdapter(
            preprocess=lambda row: row,
            grounding_policy="all_answers_verbatim",
            grounding_note="test",
        ),
    )
    record = _record()
    record.qa_pairs.extend(
        [
            record.qa_pairs[0].model_copy(deep=True),
            QuestionAnnotation(
                question_id="unknown-q",
                question="Unknown?",
                answers=("Exact Gold",),
                answer_provenance="source",
                question_type="single_hop",
                gold_atom_ids=["a1"],
                relevant_atom_ids=["a1"],
            ),
        ]
    )

    report = semantic_validation.audit_records([record], {"r1": _source(["Exact Gold"])})
    codes = [issue["code"] for issue in report["issues"]]

    assert "duplicate_question_id" in codes
    assert "unknown_question_id" in codes


def test_semantic_audit_hard_fails_missing_source_question_id(monkeypatch):
    monkeypatch.setitem(
        preprocessors.DATASET_ADAPTERS,
        "synthetic_all",
        preprocessors.DatasetAdapter(
            preprocess=lambda row: row,
            grounding_policy="all_answers_verbatim",
            grounding_note="test",
        ),
    )
    source = _source(["Exact Gold"])
    del source["questions"][0]["id"]

    with pytest.raises(ValueError, match="missing source question_id"):
        semantic_validation.audit_records([_record()], {"r1": source})


def test_semantic_audit_requires_exact_source_record_set(monkeypatch):
    monkeypatch.setitem(
        preprocessors.DATASET_ADAPTERS,
        "synthetic_all",
        preprocessors.DatasetAdapter(
            preprocess=lambda row: row,
            grounding_policy="all_answers_verbatim",
            grounding_note="test",
        ),
    )
    record = _record()
    unknown = record.model_copy(update={"record_id": "unknown-record"}, deep=True)
    sources = {
        "r1": _source(["Exact Gold"]),
        "missing-record": {**_source(["Exact Gold"]), "id": "missing-record"},
    }

    report = semantic_validation.audit_records([record, record.model_copy(deep=True), unknown], sources)
    codes = [issue["code"] for issue in report["issues"]]

    assert "duplicate_record_id" in codes
    assert "unknown_record_id" in codes
    assert "source_record_missing_in_atomized" in codes
    assert report["source_questions"] == 2
    assert report["source_answer_matches"] == 1
    assert report["source_answer_mismatches"] == 1
