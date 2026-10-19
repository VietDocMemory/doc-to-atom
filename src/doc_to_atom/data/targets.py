from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from doc_to_atom.atomization.models import QuestionAnnotation


class TargetPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class TargetValue:
    text: str
    provenance: Literal["source_answer", "derived_refusal"]


def resolve_target(
    question: QuestionAnnotation, refusal_literal: str = "unanswerable"
) -> TargetValue:
    if question.is_irrelevant:
        return TargetValue(refusal_literal, "derived_refusal")
    if not question.answers:
        raise TargetPolicyError(f"{question.question_id}: missing source answer")
    return TargetValue(question.answers[0], "source_answer")
