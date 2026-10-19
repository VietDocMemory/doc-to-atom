import importlib.util
from pathlib import Path

from doc_to_atom.atomization.models import Atom, QuestionAnnotation, StructuredRecord

_SCRIPT = Path(__file__).parents[1] / "scripts" / "report_atomization_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("acceptance_report", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_summary = _MODULE._summary


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
