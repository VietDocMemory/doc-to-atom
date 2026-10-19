from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AnnotationMeta, QuestionAnnotation, StructuredRecord
from .validation import canonicalize_record


class MigrationError(ValueError):
    pass


def _duplicate(values: list[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _source_answers(question: dict[str, Any]) -> tuple[str, ...]:
    value = question["answers"] if "answers" in question else question.get("answer", [])
    if value is None:
        value = []
    elif isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MigrationError("source answers must be an ordered list of strings")
    return tuple(value)


def migrate_record(
    legacy_record: StructuredRecord, source_record: dict[str, Any]
) -> StructuredRecord:
    source_id = str(source_record.get("id", source_record.get("record_id", "")))
    if source_id != legacy_record.record_id:
        raise MigrationError(
            f"source record_id mismatch: expected {legacy_record.record_id}, got {source_id}"
        )

    source_questions = source_record.get("questions")
    if not isinstance(source_questions, list):
        raise MigrationError("source questions must be an ordered list")
    source_ids = [str(item.get("id", item.get("question_id", ""))) for item in source_questions]
    duplicate = _duplicate(source_ids)
    if duplicate is not None:
        raise MigrationError(f"duplicate source question_id: {duplicate}")

    legacy_ids = [qa.question_id for qa in legacy_record.qa_pairs]
    duplicate = _duplicate(legacy_ids)
    if duplicate is not None:
        raise MigrationError(f"duplicate legacy question_id: {duplicate}")
    legacy_by_id = {qa.question_id: qa for qa in legacy_record.qa_pairs}
    source_id_set = set(source_ids)

    unknown = next(
        (
            qa.question_id
            for qa in legacy_record.qa_pairs
            if qa.question_id not in source_id_set and not qa.is_generated_irrelevant
        ),
        None,
    )
    if unknown is not None:
        raise MigrationError(f"unknown legacy question_id: {unknown}")
    missing = next((question_id for question_id in source_ids if question_id not in legacy_by_id), None)
    if missing is not None:
        raise MigrationError(f"missing legacy question_id: {missing}")

    migrated_questions: list[QuestionAnnotation] = []
    for source_question, question_id in zip(source_questions, source_ids, strict=True):
        legacy = legacy_by_id[question_id]
        answers = _source_answers(source_question)
        is_irrelevant = bool(source_question.get("is_irrelevant", False))
        if is_irrelevant and answers:
            raise MigrationError(f"irrelevant source question has answers: {question_id}")
        migrated_questions.append(
            QuestionAnnotation(
                question_id=question_id,
                question=str(source_question["question"]),
                answers=answers,
                answer_provenance="source",
                question_type=(
                    "irrelevant"
                    if is_irrelevant
                    else source_question.get("question_type", legacy.question_type)
                ),
                gold_atom_ids=list(legacy.gold_atom_ids),
                supporting_atom_ids=list(legacy.supporting_atom_ids),
                distractor_atom_ids=list(legacy.distractor_atom_ids),
                relevant_atom_ids=list(legacy.relevant_atom_ids),
                is_irrelevant=is_irrelevant,
                has_conflict=legacy.has_conflict,
            )
        )

    for legacy in legacy_record.qa_pairs:
        if legacy.question_id in source_id_set:
            continue
        migrated_questions.append(
            QuestionAnnotation(
                question_id=legacy.question_id,
                question=legacy.question,
                answers=[],
                answer_provenance="generated_irrelevant",
                question_type="irrelevant",
                gold_atom_ids=list(legacy.gold_atom_ids),
                supporting_atom_ids=list(legacy.supporting_atom_ids),
                distractor_atom_ids=list(legacy.distractor_atom_ids),
                relevant_atom_ids=list(legacy.relevant_atom_ids),
                is_irrelevant=True,
                has_conflict=legacy.has_conflict,
                is_generated_irrelevant=True,
            )
        )

    meta = legacy_record.annotation_meta.model_copy(deep=True)
    if not isinstance(meta, AnnotationMeta):
        meta = AnnotationMeta.model_validate(meta)
    migrated = StructuredRecord(
        record_id=legacy_record.record_id,
        dataset=source_record.get("dataset", legacy_record.dataset),
        split=source_record.get("split", legacy_record.split),
        context=str(source_record["context"]),
        atoms=[atom.model_copy(deep=True) for atom in legacy_record.atoms],
        qa_pairs=migrated_questions,
        annotation_meta=meta,
    )
    return canonicalize_record(migrated)


def _read_source(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            record_id = str(value.get("id", value.get("record_id", "")))
            if not record_id:
                raise MigrationError(f"source line {line_number} has no record id")
            if record_id in records:
                raise MigrationError(f"duplicate source record_id: {record_id}")
            records[record_id] = value
    return records


def migrate_jsonl(
    legacy_path: str | Path, source_path: str | Path, output_path: str | Path
) -> dict[str, int | float]:
    sources = _read_source(source_path)
    migrated: list[StructuredRecord] = []
    seen_legacy: set[str] = set()
    with Path(legacy_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            legacy = StructuredRecord.model_validate_json(line)
            if legacy.record_id in seen_legacy:
                raise MigrationError(f"duplicate legacy record_id: {legacy.record_id}")
            seen_legacy.add(legacy.record_id)
            source = sources.get(legacy.record_id)
            if source is None:
                raise MigrationError(f"missing source record: {legacy.record_id}")
            migrated.append(migrate_record(legacy, source))

    missing_legacy = next((record_id for record_id in sources if record_id not in seen_legacy), None)
    if missing_legacy is not None:
        raise MigrationError(f"missing legacy record: {missing_legacy}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in migrated:
            handle.write(record.model_dump_json() + "\n")
    source_questions = sum(
        qa.answer_provenance == "source" for record in migrated for qa in record.qa_pairs
    )
    return {
        "records": len(migrated),
        "questions": sum(len(record.qa_pairs) for record in migrated),
        "source_answer_matches": source_questions,
    }
