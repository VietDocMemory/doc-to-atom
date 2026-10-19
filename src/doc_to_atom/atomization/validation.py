from __future__ import annotations

import re
from typing import Any

from .models import StructuredRecord, TransformationEvent


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _record_transformation(
    record: StructuredRecord,
    *,
    code: str,
    message: str,
    question_id: str | None = None,
    atom_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    event = TransformationEvent(
        code=code,
        stage="canonicalization",
        message=message,
        question_id=question_id,
        atom_id=atom_id,
        details=details or {},
    )
    record.annotation_meta.transformations.append(event)
    record.annotation_meta.warnings.append(message)


def _normalise_role(
    record: StructuredRecord,
    values: list[str],
    *,
    known_ids: set[str],
    question_id: str,
    role: str,
) -> list[str]:
    unknown = [atom_id for atom_id in values if atom_id not in known_ids]
    if unknown:
        _record_transformation(
            record,
            code="DROP_UNKNOWN_ATOM_REFERENCE",
            message=f"{question_id}: dropped unknown {role} atoms {unknown}",
            question_id=question_id,
            details={"role": role, "removed_atom_ids": unknown},
        )

    result: list[str] = []
    duplicates: list[str] = []
    for atom_id in values:
        if atom_id not in known_ids:
            continue
        if atom_id in result:
            duplicates.append(atom_id)
        else:
            result.append(atom_id)
    if duplicates:
        _record_transformation(
            record,
            code="DEDUPLICATE_ROLE_ATOM",
            message=f"{question_id}: deduplicated {role} atoms {duplicates}",
            question_id=question_id,
            details={"role": role, "removed_atom_ids": duplicates},
        )
    return result


def _remove_overlap(
    record: StructuredRecord,
    values: list[str],
    excluded: set[str],
    *,
    code: str,
    message_role: str,
    question_id: str,
) -> list[str]:
    removed = [atom_id for atom_id in values if atom_id in excluded]
    if removed:
        _record_transformation(
            record,
            code=code,
            message=f"{question_id}: removed {message_role} overlap {removed}",
            question_id=question_id,
            details={"removed_atom_ids": removed},
        )
    return [atom_id for atom_id in values if atom_id not in excluded]


def canonicalize_record(
    record: StructuredRecord, fuzzy_whitespace: bool = True
) -> StructuredRecord:
    """Apply deterministic repairs while retaining structured provenance for every mutation."""
    context_match = _normalise_ws(record.context) if fuzzy_whitespace else record.context
    ids = {atom.atom_id for atom in record.atoms}
    conflict_by_id = {atom.atom_id: atom.conflict_group for atom in record.atoms}

    for atom in record.atoms:
        span = _normalise_ws(atom.source_span) if fuzzy_whitespace else atom.source_span
        span_valid = span in context_match
        if atom.span_valid != span_valid:
            _record_transformation(
                record,
                code="MARK_INVALID_SOURCE_SPAN",
                message=f"{atom.atom_id}: source_span mismatch",
                atom_id=atom.atom_id,
                details={"source_span": atom.source_span, "span_valid": span_valid},
            )
            atom.span_valid = span_valid

        valid_relations = []
        removed_targets = []
        for relation in atom.relations:
            if relation.target_atom_id in ids:
                valid_relations.append(relation)
            else:
                removed_targets.append(relation.target_atom_id)
        if removed_targets:
            _record_transformation(
                record,
                code="DROP_UNKNOWN_RELATION_TARGET",
                message=f"{atom.atom_id}: dropped relations to unknown atoms {removed_targets}",
                atom_id=atom.atom_id,
                details={"removed_atom_ids": removed_targets},
            )
            atom.relations = valid_relations

    for qa in record.qa_pairs:
        qa_id = qa.question_id
        gold = _normalise_role(
            record, qa.gold_atom_ids, known_ids=ids, question_id=qa_id, role="gold"
        )
        supporting = _normalise_role(
            record,
            qa.supporting_atom_ids,
            known_ids=ids,
            question_id=qa_id,
            role="supporting",
        )
        distractor = _normalise_role(
            record,
            qa.distractor_atom_ids,
            known_ids=ids,
            question_id=qa_id,
            role="distractor",
        )

        if qa.question_type == "irrelevant" and not qa.is_irrelevant:
            _record_transformation(
                record,
                code="NORMALIZE_IRRELEVANT_FLAG",
                message=f"{qa_id}: set is_irrelevant from question_type",
                question_id=qa_id,
            )
            qa.is_irrelevant = True
        if qa.is_irrelevant and qa.question_type != "irrelevant":
            _record_transformation(
                record,
                code="NORMALIZE_IRRELEVANT_QUESTION_TYPE",
                message=f"{qa_id}: set question_type to irrelevant",
                question_id=qa_id,
            )
            qa.question_type = "irrelevant"

        if qa.is_irrelevant:
            if gold:
                _record_transformation(
                    record,
                    code="CLEAR_IRRELEVANT_GOLD",
                    message=f"{qa_id}: cleared gold atoms for irrelevant question",
                    question_id=qa_id,
                    details={"removed_atom_ids": list(gold)},
                )
                gold = []
            if supporting:
                _record_transformation(
                    record,
                    code="CLEAR_IRRELEVANT_SUPPORTING",
                    message=f"{qa_id}: cleared supporting atoms for irrelevant question",
                    question_id=qa_id,
                    details={"removed_atom_ids": list(supporting)},
                )
                supporting = []
        else:
            supporting = _remove_overlap(
                record,
                supporting,
                set(gold),
                code="REMOVE_GOLD_FROM_SUPPORTING",
                message_role="gold/supporting",
                question_id=qa_id,
            )

        distractor = _remove_overlap(
            record,
            distractor,
            set(gold),
            code="REMOVE_GOLD_FROM_DISTRACTOR",
            message_role="gold/distractor",
            question_id=qa_id,
        )
        distractor = _remove_overlap(
            record,
            distractor,
            set(supporting),
            code="REMOVE_SUPPORTING_FROM_DISTRACTOR",
            message_role="supporting/distractor",
            question_id=qa_id,
        )

        qa.gold_atom_ids = gold
        qa.supporting_atom_ids = supporting
        qa.distractor_atom_ids = distractor
        relevant = list(dict.fromkeys(gold + supporting))
        if qa.relevant_atom_ids != relevant:
            _record_transformation(
                record,
                code="RECOMPUTE_RELEVANT_UNION",
                message=f"{qa_id}: recomputed relevant atoms",
                question_id=qa_id,
                details={"before": list(qa.relevant_atom_ids), "after": relevant},
            )
            qa.relevant_atom_ids = relevant

        groups = [conflict_by_id[item] for item in relevant if conflict_by_id[item]]
        has_conflict = any(
            sum(atom.conflict_group == group for atom in record.atoms) > 1 for group in groups
        )
        if qa.has_conflict != has_conflict:
            _record_transformation(
                record,
                code="RECOMPUTE_CONFLICT_FLAG",
                message=f"{qa_id}: recomputed conflict flag",
                question_id=qa_id,
                details={"before": qa.has_conflict, "after": has_conflict},
            )
            qa.has_conflict = has_conflict
        qa.annotation_warnings = [
            warning
            for warning in record.annotation_meta.warnings
            if warning.startswith(f"{qa_id}:")
        ]

    record.annotation_meta.num_atoms = len(record.atoms)
    record.annotation_meta.num_questions = len(record.qa_pairs)
    record.annotation_meta.num_irrelevant_generated = sum(
        qa.is_generated_irrelevant for qa in record.qa_pairs
    )
    return record


def provider_unknown_atom_reference_count(record: StructuredRecord) -> int:
    return sum(
        len(event.details.get("removed_atom_ids", []))
        for event in record.annotation_meta.transformations
        if event.code == "DROP_UNKNOWN_ATOM_REFERENCE"
    )


def validate_record(record: StructuredRecord, fuzzy_whitespace: bool = True) -> StructuredRecord:
    """Compatibility wrapper for deterministic schema-v2 canonicalization."""
    return canonicalize_record(record, fuzzy_whitespace=fuzzy_whitespace)
