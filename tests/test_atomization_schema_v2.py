import json

import pytest
from pydantic import ValidationError

import doc_to_atom.atomization.models as models


def _question(**overrides):
    values = {
        "question_id": "q1",
        "question": "Exact question?",
        "answers": ["Gold", "Gold", " gold "],
        "answer_provenance": "source",
        "question_type": "single_hop",
    }
    values.update(overrides)
    return models.QuestionAnnotation(**values)


def test_v2_answers_round_trip_exactly_without_singular_answer():
    qa = _question()

    payload = json.loads(qa.model_dump_json())

    assert payload["answers"] == ["Gold", "Gold", " gold "]
    assert "answer" not in payload
    assert models.QuestionAnnotation.model_validate(payload).answers == (
        "Gold",
        "Gold",
        " gold ",
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("question_id", "q2"),
        ("question", "Changed?"),
        ("answers", ("Changed",)),
    ],
)
def test_source_owned_fields_are_frozen(field, replacement):
    qa = _question()

    with pytest.raises(ValidationError, match="Field is frozen"):
        setattr(qa, field, replacement)


def test_answers_collection_is_immutable():
    qa = _question()

    assert isinstance(qa.answers, tuple)
    with pytest.raises(AttributeError):
        qa.answers.append("Changed")


def test_irrelevant_question_has_empty_answers_without_source_refusal_gold():
    qa = _question(
        answers=[],
        answer_provenance="generated_irrelevant",
        question_type="irrelevant",
        is_irrelevant=True,
    )

    payload = json.loads(qa.model_dump_json())
    assert payload["answers"] == []
    assert payload["refusal_target"] is None
    assert payload["refusal_target_derived"] is False


def test_legacy_singular_answer_is_loaded_as_unverified():
    qa = models.QuestionAnnotation.model_validate(
        {
            "question_id": "legacy-q",
            "question": "Legacy?",
            "answer": ["Mutated", "Mutated"],
            "question_type": "single_hop",
        }
    )

    assert qa.answers == ("Mutated", "Mutated")
    assert qa.answer_provenance == "legacy_unverified"
    assert "answer" not in json.loads(qa.model_dump_json())


def test_structured_record_declares_schema_v2_and_structured_transformations():
    event = models.TransformationEvent(
        code="REMOVE_GOLD_FROM_SUPPORTING",
        stage="canonicalization",
        message="removed overlap",
        question_id="q1",
        details={"removed_atom_ids": ["a1"]},
    )
    record = models.StructuredRecord(
        record_id="r1",
        context="Context",
        atoms=[],
        qa_pairs=[_question()],
        annotation_meta=models.AnnotationMeta(transformations=[event]),
    )

    payload = json.loads(record.model_dump_json())
    assert payload["schema_version"] == models.ATOMIZATION_SCHEMA_VERSION == 2
    assert payload["annotation_meta"]["transformations"][0]["code"] == (
        "REMOVE_GOLD_FROM_SUPPORTING"
    )
