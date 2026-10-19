from copy import deepcopy

from doc_to_atom.atomization.models import Atom, QuestionAnnotation, StructuredRecord
from doc_to_atom.atomization import validation


def _atom(atom_id: str) -> Atom:
    return Atom(
        atom_id=atom_id,
        semantic_type="fact_claim",
        content=f"Content {atom_id}",
        source_span=f"Span {atom_id}",
        retrieval_text=f"Retrieval {atom_id}",
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1.0,
    )


def _record(qa: QuestionAnnotation) -> StructuredRecord:
    return StructuredRecord(
        record_id="r1",
        context="Span a1 Span a2 Span a3",
        atoms=[_atom("a1"), _atom("a2"), _atom("a3")],
        qa_pairs=[qa],
    )


def test_role_repair_enforces_priority_and_records_every_transformation():
    qa = QuestionAnnotation(
        question_id="q1",
        question="Exact?",
        answers=["Exact"],
        answer_provenance="source",
        question_type="multi_hop",
        gold_atom_ids=["a1", "a1", "missing-g"],
        supporting_atom_ids=["a1", "a2", "a2", "missing-s"],
        distractor_atom_ids=["a1", "a2", "a3", "a3"],
        relevant_atom_ids=["wrong"],
    )
    record = _record(qa)

    validation.canonicalize_record(record)

    repaired = record.qa_pairs[0]
    assert repaired.gold_atom_ids == ["a1"]
    assert repaired.supporting_atom_ids == ["a2"]
    assert repaired.distractor_atom_ids == ["a3"]
    assert repaired.relevant_atom_ids == ["a1", "a2"]
    assert repaired.question_id == "q1"
    assert repaired.question == "Exact?"
    assert repaired.answers == ("Exact",)
    codes = [event.code for event in record.annotation_meta.transformations]
    assert codes.count("DROP_UNKNOWN_ATOM_REFERENCE") == 2
    assert codes.count("DEDUPLICATE_ROLE_ATOM") == 3
    assert "REMOVE_GOLD_FROM_SUPPORTING" in codes
    assert "REMOVE_GOLD_FROM_DISTRACTOR" in codes
    assert "REMOVE_SUPPORTING_FROM_DISTRACTOR" in codes
    assert "RECOMPUTE_RELEVANT_UNION" in codes
    assert validation.provider_unknown_atom_reference_count(record) == 2
    assert all(event.message in record.annotation_meta.warnings for event in record.annotation_meta.transformations)


def test_role_repair_is_idempotent():
    qa = QuestionAnnotation(
        question_id="q1",
        question="Exact?",
        answers=["Exact"],
        answer_provenance="source",
        question_type="single_hop",
        gold_atom_ids=["a1", "a1"],
        supporting_atom_ids=["a1", "a2"],
        relevant_atom_ids=["stale"],
    )
    record = _record(qa)
    validation.canonicalize_record(record)
    first = deepcopy(record.model_dump())

    validation.canonicalize_record(record)

    assert record.model_dump() == first


def test_irrelevant_role_repair_clears_gold_and_supporting_with_codes():
    qa = QuestionAnnotation(
        question_id="iq",
        question="Unknown?",
        answers=[],
        answer_provenance="generated_irrelevant",
        question_type="irrelevant",
        gold_atom_ids=["a1"],
        supporting_atom_ids=["a2"],
        relevant_atom_ids=["a1", "a2"],
        is_irrelevant=True,
        is_generated_irrelevant=True,
    )
    record = _record(qa)

    validation.canonicalize_record(record)

    repaired = record.qa_pairs[0]
    assert repaired.answers == ()
    assert repaired.gold_atom_ids == []
    assert repaired.supporting_atom_ids == []
    assert repaired.relevant_atom_ids == []
    codes = [event.code for event in record.annotation_meta.transformations]
    assert "CLEAR_IRRELEVANT_GOLD" in codes
    assert "CLEAR_IRRELEVANT_SUPPORTING" in codes
    assert "RECOMPUTE_RELEVANT_UNION" in codes
