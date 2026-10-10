from __future__ import annotations

import re

from .models import StructuredRecord


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_record(record: StructuredRecord, fuzzy_whitespace: bool = True) -> StructuredRecord:
    """Apply Appendix B's deterministic post-processing and warning logging."""
    context_match = _normalise_ws(record.context) if fuzzy_whitespace else record.context
    ids = {atom.atom_id for atom in record.atoms}
    conflict_by_id = {atom.atom_id: atom.conflict_group for atom in record.atoms}

    for atom in record.atoms:
        span = _normalise_ws(atom.source_span) if fuzzy_whitespace else atom.source_span
        atom.span_valid = span in context_match
        if not atom.span_valid:
            record.annotation_meta.warnings.append(f"{atom.atom_id}: source_span mismatch")
        valid_relations = []
        for relation in atom.relations:
            if relation.target_atom_id in ids:
                valid_relations.append(relation)
            else:
                record.annotation_meta.warnings.append(
                    f"{atom.atom_id}: dropped relation to unknown {relation.target_atom_id}"
                )
        atom.relations = valid_relations

    def known(values: list[str], qa_id: str, role: str) -> list[str]:
        result = []
        for atom_id in values:
            if atom_id in ids and atom_id not in result:
                result.append(atom_id)
            elif atom_id not in ids:
                warning = f"{qa_id}: dropped unknown {role} atom {atom_id}"
                record.annotation_meta.warnings.append(warning)
        return result

    for qa in record.qa_pairs:
        qa.gold_atom_ids = known(qa.gold_atom_ids, qa.question_id, "gold")
        qa.supporting_atom_ids = known(qa.supporting_atom_ids, qa.question_id, "supporting")
        qa.distractor_atom_ids = known(qa.distractor_atom_ids, qa.question_id, "distractor")
        if qa.is_irrelevant:
            if qa.gold_atom_ids:
                record.annotation_meta.warnings.append(
                    f"{qa.question_id}: cleared gold atoms because query is irrelevant"
                )
            qa.gold_atom_ids = []
            qa.question_type = "irrelevant"
        qa.relevant_atom_ids = list(dict.fromkeys(qa.gold_atom_ids + qa.supporting_atom_ids))
        groups = [conflict_by_id[item] for item in qa.relevant_atom_ids if conflict_by_id[item]]
        qa.has_conflict = any(
            sum(atom.conflict_group == group for atom in record.atoms) > 1 for group in groups
        )
        qa.annotation_warnings = [
            warning
            for warning in record.annotation_meta.warnings
            if warning.startswith(f"{qa.question_id}:")
        ]

    record.annotation_meta.num_atoms = len(record.atoms)
    record.annotation_meta.num_questions = len(record.qa_pairs)
    record.annotation_meta.num_irrelevant_generated = sum(
        qa.is_generated_irrelevant for qa in record.qa_pairs
    )
    return record

