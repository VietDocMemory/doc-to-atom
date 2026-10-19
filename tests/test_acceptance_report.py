import importlib.util
import json
from pathlib import Path

from doc_to_atom.atomization.models import Atom, QuestionAnnotation, StructuredRecord

_SCRIPT = Path(__file__).parents[1] / "scripts" / "report_atomization_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("acceptance_report", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_summary = _MODULE._summary
build_report = _MODULE.build_report


def _qa(question_id: str, answer: str, *, generated: bool = False):
    return QuestionAnnotation(
        question_id=question_id,
        question=f"Question {question_id}?",
        answers=() if generated else (answer,),
        answer_provenance="generated_irrelevant" if generated else "source",
        question_type="irrelevant" if generated else "single_hop",
        gold_atom_ids=[] if generated else ["a1"],
        relevant_atom_ids=[] if generated else ["a1"],
        is_irrelevant=generated,
        is_generated_irrelevant=generated,
    )


def test_acceptance_summary_uses_source_questions_plus_generated_probes(tmp_path):
    atom = Atom(
        atom_id="a1",
        semantic_type="fact_claim",
        content="Gold",
        source_span="Gold",
        retrieval_text="Gold",
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1,
    )
    record = StructuredRecord(
        record_id="r1",
        dataset="squad",
        context="Gold",
        atoms=[atom],
        qa_pairs=[
            _qa("in-source", "Gold"),
            _qa("outside-source", "Missing"),
            _qa("generated_irrelevant_0", "", generated=True),
        ],
    )
    atomized = tmp_path / "atoms.jsonl"
    atomized.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    sources = {
        "r1": {
            "id": "r1",
            "dataset": "squad",
            "context": "Gold",
            "questions": [
                {"id": "in-source", "question": "Question in-source?", "answers": ["Gold"]}
            ],
        }
    }

    report = _summary(str(atomized), sources)

    assert report["questions"] == 2
    assert report["grounding_failures"] == 0
    assert report["source_questions"] == 1


def test_acceptance_report_rejects_any_semantic_error(tmp_path):
    atom = Atom(
        atom_id="a1",
        semantic_type="fact_claim",
        content="Gold",
        source_span="Gold",
        retrieval_text="Gold",
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1,
    )
    record = StructuredRecord(
        record_id="r1",
        dataset="squad",
        context="Gold",
        atoms=[atom],
        qa_pairs=[_qa("q1", "Gold")],
    )
    atomized = tmp_path / "atoms.jsonl"
    source = tmp_path / "source.jsonl"
    atomized.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    source.write_text(
        json.dumps(
            {
                "id": "r1",
                "dataset": "squad",
                "context": "Gold",
                "questions": [{"id": "q1", "question": "Different?", "answers": ["Gold"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_report(str(atomized), str(atomized), str(source))

    assert report["after"]["errors"] == 1
    assert report["accepted"] is False
    assert "errors" in report["failed_gates"]


def test_acceptance_report_does_not_filter_unknown_ids_from_after_artifact(tmp_path):
    atom = Atom(
        atom_id="a1",
        semantic_type="fact_claim",
        content="Gold",
        source_span="Gold",
        retrieval_text="Gold",
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1,
    )
    record = StructuredRecord(
        record_id="r1",
        dataset="squad",
        context="Gold",
        atoms=[atom],
        qa_pairs=[_qa("q1", "Gold"), _qa("provider-invented", "Gold")],
    )
    atomized = tmp_path / "atoms.jsonl"
    source = tmp_path / "source.jsonl"
    atomized.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    source.write_text(
        json.dumps(
            {
                "id": "r1",
                "dataset": "squad",
                "context": "Gold",
                "questions": [{"id": "q1", "question": "Question q1?", "answers": ["Gold"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_report(str(atomized), str(atomized), str(source))

    assert "unknown_question_id" in {
        issue["code"] for issue in report["after"]["issues"]
    }
    assert report["accepted"] is False
