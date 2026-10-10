from doc_to_atom.atomization.models import Atom, QuestionAnnotation, StructuredRecord
from doc_to_atom.data.semantic_validation import audit_record


def _record(qa: QuestionAnnotation) -> StructuredRecord:
    atom = Atom(
        atom_id="a1",
        semantic_type="fact_claim",
        content="The championship is called the Super Bowl.",
        source_span="The championship is called the Super Bowl.",
        retrieval_text="championship Super Bowl",
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1.0,
    )
    return StructuredRecord(record_id="r", context=atom.source_span, atoms=[atom], qa_pairs=[qa])


def test_semantic_validator_reports_missing_gold_without_repairing():
    qa = QuestionAnnotation(
        question_id="q",
        question="Name?",
        answer="Super Bowl",
        question_type="single_hop",
    )
    issues = audit_record(_record(qa))
    assert [issue["code"] for issue in issues] == ["answerable_empty_gold"]
    assert qa.gold_atom_ids == []


def test_semantic_validator_accepts_grounded_mapping():
    qa = QuestionAnnotation(
        question_id="q",
        question="Name?",
        answer="Super Bowl",
        question_type="single_hop",
        gold_atom_ids=["a1"],
        relevant_atom_ids=["a1"],
    )
    assert audit_record(_record(qa)) == []


def test_semantic_validator_reports_irrelevant_references_and_union_mismatch():
    qa = QuestionAnnotation(
        question_id="q",
        question="Unrelated?",
        answer="unanswerable",
        question_type="irrelevant",
        is_irrelevant=True,
        is_generated_irrelevant=True,
        gold_atom_ids=["missing"],
        supporting_atom_ids=["a1"],
        relevant_atom_ids=["a1"],
    )
    issues = audit_record(_record(qa))
    codes = [issue["code"] for issue in issues]
    assert "irrelevant_has_gold" in codes
    assert "unknown_atom_reference" in codes
    assert "relevant_union_mismatch" in codes
    assert qa.gold_atom_ids == ["missing"]
