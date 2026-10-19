from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from doc_to_atom.atomization.models import StructuredRecord
from doc_to_atom.atomization.validation import provider_unknown_atom_reference_count
from doc_to_atom.data.io import read_jsonl
from doc_to_atom.data.preprocessors import get_dataset_adapter


def _source_answers(question: dict[str, Any]) -> tuple[str, ...]:
    value = question["answers"] if "answers" in question else question.get("answer", [])
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or any(not isinstance(answer, str) for answer in value):
        raise ValueError("source answers must be an ordered list of strings")
    return tuple(value)


def load_source_records(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            record_id = str(value.get("id", value.get("record_id", "")))
            if not record_id:
                raise ValueError(f"{path}:{line_number}: missing record id")
            if record_id in records:
                raise ValueError(f"duplicate source record_id: {record_id}")
            records[record_id] = value
    return records


def _base(record: StructuredRecord, qa) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "question_id": qa.question_id,
        "question": qa.question,
    }


def audit_record(
    record: StructuredRecord, source_record: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    atom_by_id = {atom.atom_id: atom for atom in record.atoms}
    known_ids = set(atom_by_id)
    source_questions: dict[str, dict[str, Any]] = {}
    if source_record is not None:
        for item in source_record.get("questions", []):
            question_id = str(item.get("id", item.get("question_id", "")))
            if not question_id.strip():
                raise ValueError("missing source question_id")
            if question_id in source_questions:
                raise ValueError(f"duplicate source question_id: {question_id}")
            source_questions[question_id] = item

    adapter = get_dataset_adapter(record.dataset)
    seen_source_ids: set[str] = set()
    seen_question_ids: set[str] = set()
    for qa in record.qa_pairs:
        base = _base(record, qa)
        if qa.question_id in seen_question_ids:
            issues.append({**base, "severity": "error", "code": "duplicate_question_id"})
        seen_question_ids.add(qa.question_id)
        if (
            source_record is not None
            and qa.question_id not in source_questions
            and not qa.is_generated_irrelevant
        ):
            issues.append({**base, "severity": "error", "code": "unknown_question_id"})
        if not qa.is_irrelevant and not qa.gold_atom_ids:
            issues.append({**base, "severity": "error", "code": "answerable_empty_gold"})
        if qa.is_irrelevant and qa.answers:
            issues.append({**base, "severity": "error", "code": "irrelevant_has_answers"})
        if qa.is_irrelevant and qa.gold_atom_ids:
            issues.append({**base, "severity": "error", "code": "irrelevant_has_gold"})
        if qa.is_irrelevant and qa.supporting_atom_ids:
            issues.append({**base, "severity": "error", "code": "irrelevant_has_supporting"})

        overlaps = (
            ("gold_supporting_overlap", set(qa.gold_atom_ids) & set(qa.supporting_atom_ids)),
            ("gold_distractor_overlap", set(qa.gold_atom_ids) & set(qa.distractor_atom_ids)),
            (
                "supporting_distractor_overlap",
                set(qa.supporting_atom_ids) & set(qa.distractor_atom_ids),
            ),
        )
        for code, atom_ids in overlaps:
            if atom_ids:
                issues.append(
                    {**base, "severity": "error", "code": code, "atom_ids": sorted(atom_ids)}
                )

        roles = {
            "gold": qa.gold_atom_ids,
            "supporting": qa.supporting_atom_ids,
            "distractor": qa.distractor_atom_ids,
            "relevant": qa.relevant_atom_ids,
        }
        for role, values in roles.items():
            for atom_id in values:
                if atom_id not in known_ids:
                    issues.append(
                        {
                            **base,
                            "severity": "error",
                            "code": "unknown_atom_reference",
                            "role": role,
                            "atom_id": atom_id,
                        }
                    )

        expected_relevant = list(dict.fromkeys(qa.gold_atom_ids + qa.supporting_atom_ids))
        if qa.relevant_atom_ids != expected_relevant:
            issues.append(
                {
                    **base,
                    "severity": "error",
                    "code": "relevant_union_mismatch",
                    "expected": expected_relevant,
                    "actual": qa.relevant_atom_ids,
                }
            )

        source_question = source_questions.get(qa.question_id)
        if source_question is not None:
            seen_source_ids.add(qa.question_id)
            expected_answers = _source_answers(source_question)
            if qa.answers != expected_answers:
                issues.append(
                    {
                        **base,
                        "severity": "error",
                        "code": "source_answer_mismatch",
                        "expected": list(expected_answers),
                        "actual": list(qa.answers),
                    }
                )
            expected_question = str(source_question["question"])
            if qa.question != expected_question:
                issues.append(
                    {
                        **base,
                        "severity": "error",
                        "code": "source_question_mismatch",
                        "expected": expected_question,
                        "actual": qa.question,
                    }
                )

        if not qa.is_irrelevant and qa.gold_atom_ids and qa.answers:
            evidence = [
                f"{atom_by_id[atom_id].content} {atom_by_id[atom_id].source_span}"
                for atom_id in qa.gold_atom_ids
                if atom_id in atom_by_id
            ]
            grounded = [any(answer in text for text in evidence) for answer in qa.answers]
            policy = adapter.grounding_policy
            failed = (policy == "all_answers_verbatim" and not all(grounded)) or (
                policy == "any_answer_verbatim" and not any(grounded)
            )
            if failed:
                issues.append(
                    {
                        **base,
                        "severity": "error",
                        "code": "source_answer_not_grounded",
                        "answers": list(qa.answers),
                        "grounded": grounded,
                        "policy": policy,
                    }
                )

    for question_id in source_questions.keys() - seen_source_ids:
        source_question = source_questions[question_id]
        issues.append(
            {
                "record_id": record.record_id,
                "question_id": question_id,
                "question": source_question.get("question", ""),
                "severity": "error",
                "code": "source_question_missing_in_atomized",
            }
        )

    statuses = {
        "decomposition": record.annotation_meta.decomposition_parse_status,
        "irrelevant": record.annotation_meta.irrelevant_parse_status,
        "question": record.annotation_meta.question_parse_status,
    }
    for stage, status in statuses.items():
        if status != "ok":
            issues.append(
                {
                    "record_id": record.record_id,
                    "severity": "error",
                    "code": "parse_error",
                    "stage": stage,
                    "status": status,
                }
            )
    return issues


def audit_records(
    records: list[StructuredRecord],
    source_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_records = source_records or {}
    issues = [
        issue
        for record in records
        for issue in audit_record(record, source_records.get(record.record_id))
    ]
    codes = Counter(issue["code"] for issue in issues)
    transformations = Counter(
        event.code for record in records for event in record.annotation_meta.transformations
    )
    source_question_count = sum(
        len(source_records.get(record.record_id, {}).get("questions", [])) for record in records
    )
    answer_mismatches = codes["source_answer_mismatch"] + codes[
        "source_question_missing_in_atomized"
    ]
    answer_matches = max(0, source_question_count - answer_mismatches)
    return {
        "records": len(records),
        "questions": sum(len(record.qa_pairs) for record in records),
        "errors": sum(issue["severity"] == "error" for issue in issues),
        "warnings": sum(issue["severity"] == "warning" for issue in issues),
        "source_questions": source_question_count,
        "source_answer_matches": answer_matches,
        "source_answer_mismatches": answer_mismatches,
        "answer_preservation_rate": (
            None if source_question_count == 0 else answer_matches / source_question_count
        ),
        "answerable_empty_gold": codes["answerable_empty_gold"],
        "grounding_failures": codes["source_answer_not_grounded"],
        "gold_supporting_overlap": codes["gold_supporting_overlap"],
        "gold_distractor_overlap": codes["gold_distractor_overlap"],
        "supporting_distractor_overlap": codes["supporting_distractor_overlap"],
        "irrelevant_with_answers": codes["irrelevant_has_answers"],
        "irrelevant_with_gold": codes["irrelevant_has_gold"],
        "irrelevant_with_supporting": codes["irrelevant_has_supporting"],
        "provider_unknown_atom_references": sum(
            provider_unknown_atom_reference_count(record) for record in records
        ),
        "canonical_unknown_atom_references": codes["unknown_atom_reference"],
        "parse_errors": codes["parse_error"],
        "transformation_counts": dict(sorted(transformations.items())),
        "issues": issues,
    }


def audit_jsonl(path: str | Path, source_path: str | Path | None = None) -> dict[str, Any]:
    sources = None if source_path is None else load_source_records(source_path)
    return audit_records(read_jsonl(path), sources)
