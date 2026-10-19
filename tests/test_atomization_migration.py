import json
from types import SimpleNamespace
import importlib

import pytest

from doc_to_atom.atomization.models import Atom, QuestionAnnotation, StructuredRecord

try:
    migration = importlib.import_module("doc_to_atom.atomization.migration")
except ModuleNotFoundError:
    migration = SimpleNamespace()


def _atom():
    return Atom(
        atom_id="a1",
        semantic_type="fact_claim",
        content="Gold & exact.",
        source_span="Gold & exact.",
        retrieval_text="Gold exact",
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1,
    )


def _legacy(*questions):
    return StructuredRecord(
        record_id="r1",
        dataset="squad",
        split="train",
        context="Gold & exact.",
        atoms=[_atom()],
        qa_pairs=list(questions),
    )


def _legacy_question(question_id="q1", **overrides):
    values = {
        "question_id": question_id,
        "question": "Provider rewrite?",
        "answer": "Mutated answer",
        "question_type": "single_hop",
        "gold_atom_ids": ["a1"],
        "supporting_atom_ids": ["a1"],
    }
    values.update(overrides)
    return QuestionAnnotation(**values)


def _source(questions=None):
    return {
        "id": "r1",
        "dataset": "squad",
        "split": "train",
        "context": "Gold & exact.",
        "questions": questions
        or [
            {
                "id": "q1",
                "question": "Exact source question?",
                "answers": ["Gold & exact", "Gold & exact", " gold "],
                "question_type": "single_hop",
            }
        ],
    }


def test_migration_uses_exact_source_fields_and_never_legacy_answer():
    migrated = migration.migrate_record(_legacy(_legacy_question()), _source())

    qa = migrated.qa_pairs[0]
    assert qa.question_id == "q1"
    assert qa.question == "Exact source question?"
    assert qa.answers == ("Gold & exact", "Gold & exact", " gold ")
    assert qa.answer_provenance == "source"
    assert qa.supporting_atom_ids == []
    assert any(
        event.code == "REMOVE_GOLD_FROM_SUPPORTING"
        for event in migrated.annotation_meta.transformations
    )


def test_migration_retains_only_explicit_generated_irrelevant_extras():
    probe = _legacy_question(
        "generated_irrelevant_0",
        question="Generated probe?",
        answer="unanswerable",
        question_type="irrelevant",
        is_irrelevant=True,
        is_generated_irrelevant=True,
        gold_atom_ids=[],
        supporting_atom_ids=[],
    )

    migrated = migration.migrate_record(_legacy(_legacy_question(), probe), _source())

    assert [qa.question_id for qa in migrated.qa_pairs] == ["q1", "generated_irrelevant_0"]
    assert migrated.qa_pairs[1].answers == ()
    assert migrated.qa_pairs[1].answer_provenance == "generated_irrelevant"


@pytest.mark.parametrize(
    ("legacy_questions", "source_questions", "message"),
    [
        ([_legacy_question(), _legacy_question()], None, "duplicate legacy question_id: q1"),
        (
            [_legacy_question()],
            [
                {"id": "q1", "question": "One?", "answers": ["One"]},
                {"id": "q1", "question": "Two?", "answers": ["Two"]},
            ],
            "duplicate source question_id: q1",
        ),
        (
            [_legacy_question("unknown")],
            None,
            "unknown legacy question_id: unknown",
        ),
        ([], None, "missing legacy question_id: q1"),
    ],
)
def test_migration_rejects_question_id_contract_errors(
    legacy_questions, source_questions, message
):
    source = _source(source_questions) if source_questions is not None else _source()
    with pytest.raises(migration.MigrationError, match=message):
        migration.migrate_record(_legacy(*legacy_questions), source)


def test_migration_rejects_non_string_source_answer_shape():
    source = _source(
        [{"id": "q1", "question": "Exact?", "answers": ["Gold", 7]}]
    )

    with pytest.raises(migration.MigrationError, match="ordered list of strings"):
        migration.migrate_record(_legacy(_legacy_question()), source)


def test_migrate_jsonl_requires_source_record_and_writes_v2(tmp_path):
    legacy_path = tmp_path / "legacy.jsonl"
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "v2.jsonl"
    legacy_path.write_text(_legacy(_legacy_question()).model_dump_json() + "\n", encoding="utf-8")
    source_path.write_text(json.dumps(_source(), ensure_ascii=False) + "\n", encoding="utf-8")

    report = migration.migrate_jsonl(legacy_path, source_path, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert report == {"records": 1, "questions": 1, "source_answer_matches": 1}
    assert payload["schema_version"] == 2
    assert payload["qa_pairs"][0]["answers"] == ["Gold & exact", "Gold & exact", " gold "]

    source_path.write_text(
        json.dumps({**_source(), "id": "different"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(migration.MigrationError, match="missing source record: r1"):
        migration.migrate_jsonl(legacy_path, source_path, output_path)
