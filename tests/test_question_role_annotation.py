import pytest
from pydantic import ValidationError

import doc_to_atom.atomization.models as models
import doc_to_atom.atomization.parser as parser


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("answer", "provider answer"),
        ("answers", ["provider answer"]),
        ("question", "provider rewrite?"),
        ("refusal_target", "unanswerable"),
    ],
)
def test_role_dto_forbids_source_owned_and_refusal_fields(field, value):
    payload = {"question_id": "q1", "gold_atom_ids": ["a1"], field: value}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        models.QuestionRoleAnnotation(**payload)


def test_role_dto_exposes_only_annotation_owned_fields():
    assert set(models.QuestionRoleAnnotation.model_fields) == {
        "question_id",
        "question_type",
        "gold_atom_ids",
        "supporting_atom_ids",
        "distractor_atom_ids",
    }


def test_role_xml_preserves_order_and_duplicates_for_canonical_repair():
    xml = """<question_annotations><question_annotation id="q1">
    <question_type>multi_hop</question_type>
    <gold_atom_ids><atom_id>a2</atom_id><atom_id>a1</atom_id><atom_id>a2</atom_id></gold_atom_ids>
    <supporting_atom_ids><atom_id>a1</atom_id></supporting_atom_ids>
    <distractor_atom_ids>a3,a1</distractor_atom_ids>
    </question_annotation></question_annotations>"""

    roles, notices = parser.parse_question_roles_xml(xml)

    assert notices == []
    assert len(roles) == 1
    assert roles[0].question_id == "q1"
    assert roles[0].question_type == "multi_hop"
    assert roles[0].gold_atom_ids == ("a2", "a1", "a2")
    assert roles[0].supporting_atom_ids == ("a1",)
    assert roles[0].distractor_atom_ids == ("a3", "a1")


@pytest.mark.parametrize(
    "forbidden_xml",
    [
        "<answer>provider answer</answer>",
        "<answers><answer>provider answer</answer></answers>",
        "<question>provider rewrite?</question>",
        "<refusal_target>unanswerable</refusal_target>",
    ],
)
def test_role_xml_rejects_forbidden_provider_fields(forbidden_xml):
    xml = f"""<question_annotations><question_annotation id="q1">
    {forbidden_xml}<question_type>single_hop</question_type>
    <gold_atom_ids><atom_id>a1</atom_id></gold_atom_ids>
    </question_annotation></question_annotations>"""

    roles, notices = parser.parse_question_roles_xml(xml)

    assert roles == []
    assert [notice.code for notice in notices] == ["FORBIDDEN_ANNOTATION_FIELD"]
    assert notices[0].severity == "error"
