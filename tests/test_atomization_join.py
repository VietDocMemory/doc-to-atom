import importlib

import pytest

from doc_to_atom.atomization.cli import _source_question_id
from doc_to_atom.atomization.client import OpenAIConfig, ThreeStageXMLAtomizer
from doc_to_atom.atomization.models import QuestionAnnotation, QuestionRoleAnnotation


def _join_api():
    try:
        return importlib.import_module("doc_to_atom.atomization.question_join")
    except ModuleNotFoundError:
        pytest.fail("question_join production module is missing")


def _source(question_id="source-q", question="Original question?", answers=None):
    return QuestionAnnotation(
        question_id=question_id,
        question=question,
        answers=["Gold", "Gold", " gold "] if answers is None else answers,
        answer_provenance="source",
        question_type="single_hop",
    )


def _probe(question_id="generated_irrelevant_0", question="Unknown?"):
    return QuestionAnnotation(
        question_id=question_id,
        question=question,
        answers=[],
        answer_provenance="generated_irrelevant",
        question_type="irrelevant",
        is_irrelevant=True,
        is_generated_irrelevant=True,
    )


def test_authoritative_set_is_source_then_generated_and_keeps_exact_answers():
    authoritative = _join_api().build_authoritative_questions([_source()], [_probe()])

    assert isinstance(authoritative, tuple)
    assert [qa.question_id for qa in authoritative] == ["source-q", "generated_irrelevant_0"]
    assert authoritative[0].answers == ("Gold", "Gold", " gold ")
    assert authoritative[1].answers == ()


def test_authoritative_set_rejects_duplicate_ids_across_source_and_probes():
    api = _join_api()
    with pytest.raises(api.QuestionJoinError, match="duplicate authoritative question_id: source-q"):
        api.build_authoritative_questions([_source()], [_probe(question_id="source-q")])


@pytest.mark.parametrize("source", [{}, {"id": ""}, {"question_id": "   "}])
def test_source_question_id_is_required_instead_of_synthesized(source):
    with pytest.raises(ValueError, match="missing question_id"):
        _source_question_id(source)


def test_join_preserves_source_owned_fields_and_applies_only_roles():
    source = _source(question="Case & punctuation?", answers=["A & B", "A & B"])
    roles = QuestionRoleAnnotation(
        question_id="source-q",
        question_type="multi_hop",
        gold_atom_ids=["a1"],
        supporting_atom_ids=["a2"],
        distractor_atom_ids=["a3"],
    )

    [joined] = _join_api().join_question_roles((source,), [roles])

    assert joined.question_id == "source-q"
    assert joined.question == "Case & punctuation?"
    assert joined.answers == ("A & B", "A & B")
    assert joined.answer_provenance == "source"
    assert joined.question_type == "multi_hop"
    assert joined.gold_atom_ids == ["a1"]
    assert joined.supporting_atom_ids == ["a2"]
    assert joined.distractor_atom_ids == ["a3"]


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        ([], "missing provider question_id: source-q"),
        (
            [
                QuestionRoleAnnotation(question_id="source-q"),
                QuestionRoleAnnotation(question_id="source-q"),
            ],
            "duplicate provider question_id: source-q",
        ),
        (
            [QuestionRoleAnnotation(question_id="unknown-q")],
            "unknown provider question_id: unknown-q",
        ),
    ],
)
def test_join_rejects_non_bijective_provider_ids(roles, message):
    api = _join_api()
    with pytest.raises(api.QuestionJoinError, match=message):
        api.join_question_roles((_source(),), roles)


def test_join_rejects_repeated_known_id_even_when_another_id_is_missing():
    api = _join_api()
    authoritative = (_source("q1"), _source("q2"))
    roles = [
        QuestionRoleAnnotation(question_id="q1"),
        QuestionRoleAnnotation(question_id="q1"),
    ]

    with pytest.raises(api.QuestionJoinError, match="duplicate provider question_id: q1"):
        api.join_question_roles(authoritative, roles)


def test_join_rejects_provider_attempt_to_flip_source_relevance():
    api = _join_api()
    roles = [QuestionRoleAnnotation(question_id="source-q", question_type="irrelevant")]

    with pytest.raises(api.QuestionJoinError, match="relevance conflicts with authoritative"):
        api.join_question_roles((_source(),), roles)


class _FakeAtomizer(ThreeStageXMLAtomizer):
    def __init__(self, responses):
        super().__init__(OpenAIConfig(api_key="test", irrelevant_probe_count=1), "openai")
        self.responses = iter(responses)
        self.users = []

    def _request(self, system: str, user: str) -> str:
        self.users.append(user)
        return next(self.responses)


def test_three_stage_atomizer_joins_roles_into_authoritative_questions():
    atomizer = _FakeAtomizer(
        [
            """<atoms><atom id="a1"><semantic_type>fact_claim</semantic_type>
            <content>A &amp; B is exact.</content><source_span>A &amp; B is exact.</source_span>
            <retrieval_text>A B exact</retrieval_text><is_answer_bearing>true</is_answer_bearing>
            <abstraction_level>evidence</abstraction_level><confidence>1</confidence>
            <relations/></atom></atoms>""",
            """<irrelevant_questions><question>Unknown &amp; probe?</question>
            </irrelevant_questions>""",
            """<question_annotations>
            <question_annotation id="source-q"><question_type>single_hop</question_type>
            <gold_atom_ids><atom_id>a1</atom_id></gold_atom_ids></question_annotation>
            <question_annotation id="generated_irrelevant_0"><question_type>irrelevant</question_type>
            <gold_atom_ids/></question_annotation>
            </question_annotations>""",
        ]
    )
    source = _source(question="What is A & B?", answers=["A & B", "A & B"])

    record = atomizer.atomize("r1", "A & B is exact.", [source], dataset="squad")

    assert record is not None
    assert [qa.question_id for qa in record.qa_pairs] == [
        "source-q",
        "generated_irrelevant_0",
    ]
    assert record.qa_pairs[0].question == "What is A & B?"
    assert record.qa_pairs[0].answers == ("A & B", "A & B")
    assert record.qa_pairs[1].answers == ()
    assert record.qa_pairs[1].is_irrelevant
    assert "A &amp; B" in atomizer.users[2]
    assert atomizer.users[2].count("<atoms") == 1
    assert atomizer.users[2].count("<questions") == 1
