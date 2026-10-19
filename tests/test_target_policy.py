import importlib
from types import SimpleNamespace

import pytest

from doc_to_atom.atomization.models import QuestionAnnotation

try:
    targets = importlib.import_module("doc_to_atom.data.targets")
except ModuleNotFoundError:
    targets = SimpleNamespace()


def _qa(**overrides):
    values = {
        "question_id": "q1",
        "question": "Exact?",
        "answers": [" Exact answer ", "Alternative"],
        "answer_provenance": "source",
        "question_type": "single_hop",
    }
    values.update(overrides)
    return QuestionAnnotation(**values)


def test_answerable_target_uses_exact_first_source_answer():
    target = targets.resolve_target(_qa())

    assert target.text == " Exact answer "
    assert target.provenance == "source_answer"


def test_irrelevant_target_is_derived_without_mutating_answers():
    qa = _qa(
        answers=[],
        answer_provenance="generated_irrelevant",
        question_type="irrelevant",
        is_irrelevant=True,
    )

    target = targets.resolve_target(qa, refusal_literal="cannot_answer")

    assert target.text == "cannot_answer"
    assert target.provenance == "derived_refusal"
    assert qa.answers == ()
    assert qa.refusal_target is None


def test_answerable_question_without_source_answers_is_rejected():
    with pytest.raises(targets.TargetPolicyError, match="q1: missing source answer"):
        targets.resolve_target(_qa(answers=[]))
