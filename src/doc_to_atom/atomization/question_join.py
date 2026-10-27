from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import QuestionAnnotation, QuestionRoleAnnotation


class QuestionJoinError(ValueError):
    """Provider annotations do not form a bijection with authoritative questions."""


def _first_duplicate(values: Iterable[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def build_authoritative_questions(
    source_questions: Sequence[QuestionAnnotation],
    generated_probes: Sequence[QuestionAnnotation],
) -> tuple[QuestionAnnotation, ...]:
    authoritative = (*source_questions, *generated_probes)
    duplicate = _first_duplicate(qa.question_id for qa in authoritative)
    if duplicate is not None:
        raise QuestionJoinError(f"duplicate authoritative question_id: {duplicate}")
    return authoritative


def join_question_roles(
    authoritative_questions: Sequence[QuestionAnnotation],
    role_annotations: Sequence[QuestionRoleAnnotation],
) -> list[QuestionAnnotation]:
    authoritative = build_authoritative_questions(authoritative_questions, ())
    authoritative_by_id = {qa.question_id: qa for qa in authoritative}

    duplicate = _first_duplicate(role.question_id for role in role_annotations)
    if duplicate is not None:
        raise QuestionJoinError(f"duplicate provider question_id: {duplicate}")

    role_by_id = {role.question_id: role for role in role_annotations}
    unknown = next((item for item in role_by_id if item not in authoritative_by_id), None)
    if unknown is not None:
        raise QuestionJoinError(f"unknown provider question_id: {unknown}")
    missing = next((item for item in authoritative_by_id if item not in role_by_id), None)
    if missing is not None:
        raise QuestionJoinError(f"missing provider question_id: {missing}")

    joined: list[QuestionAnnotation] = []
    for source in authoritative:
        role = role_by_id[source.question_id]
        if role.question_type == "irrelevant" and not source.is_irrelevant:
            raise QuestionJoinError(
                f"provider relevance conflicts with authoritative question_id: {source.question_id}"
            )
        # Generation is provenance, not a guaranteed relevance label. Only a
        # generated probe may be rescued by the annotation pass.
        rescued = source.is_generated_irrelevant and role.question_type not in (None, "irrelevant")
        if rescued and not role.gold_atom_ids:
            raise QuestionJoinError(f"rescued probe has no gold atoms: {source.question_id}")
        question_type = (
            "irrelevant"
            if source.is_irrelevant and not rescued
            else (role.question_type or source.question_type)
        )
        joined.append(
            source.model_copy(
                update={
                    "question_type": question_type,
                    "is_irrelevant": source.is_irrelevant and not rescued,
                    "gold_atom_ids": list(role.gold_atom_ids),
                    "supporting_atom_ids": list(role.supporting_atom_ids),
                    "distractor_atom_ids": list(role.distractor_atom_ids),
                }
            )
        )
    return joined
