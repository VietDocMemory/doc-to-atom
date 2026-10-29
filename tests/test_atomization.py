from doc_to_atom.atomization.models import (
    AnnotationMeta,
    Atom,
    QuestionAnnotation,
    StructuredRecord,
)
from doc_to_atom.atomization.parser import parse_atoms_xml, parse_questions_xml
from doc_to_atom.atomization.validation import validate_record
from doc_to_atom.atomization.client import MiniMaxConfig, ThreeStageXMLAtomizer
from doc_to_atom.atomization.question_join import join_question_roles
from doc_to_atom.atomization.models import QuestionRoleAnnotation


def test_atom_xml_parser_recovers_partial_response():
    xml = """<atoms><atom id="atom_0"><semantic_type>fact_claim</semantic_type>
    <content>Paris is in France.</content><source_span>Paris is in France.</source_span>
    <retrieval_text>Paris country</retrieval_text><is_answer_bearing>true</is_answer_bearing>
    <abstraction_level>evidence</abstraction_level><confidence>0.9</confidence>
    <relations><relation type="supports" target="atom_1"/></relations></atom><atom"""
    atoms, warnings = parse_atoms_xml(xml)
    assert len(atoms) == 1
    assert atoms[0].semantic_type == "fact_claim"
    assert not warnings


def test_question_xml_parser():
    xml = """<root><question_annotation id="q0">
    <question_type>single_hop</question_type><gold_atom_ids><atom_id>atom_0</atom_id></gold_atom_ids>
    <supporting_atom_ids></supporting_atom_ids><distractor_atom_ids>atom_1</distractor_atom_ids>
    </question_annotation></root>"""
    questions, warnings = parse_questions_xml(xml)
    assert questions[0].gold_atom_ids == ("atom_0",)
    assert questions[0].distractor_atom_ids == ("atom_1",)
    assert not warnings


def test_attribute_xml_and_abstraction_repair():
    atom_xml = """<atoms><atom><atom_id>a1</atom_id><semantic_type>fact_claim</semantic_type>
    <content>A fact.</content><source_span>A fact.</source_span><retrieval_text>A fact.</retrieval_text>
    <is_answer_bearing>true</is_answer_bearing><abstraction_level>fact_claim</abstraction_level>
    <confidence>0.9</confidence><relations/></atom></atoms>"""
    atoms, atom_warnings = parse_atoms_xml(atom_xml)
    assert atoms[0].abstraction_level == "evidence"
    assert "normalized abstraction_level" in atom_warnings[0]

    question_xml = """<question_annotations><question_annotation question_id="q1"
    question_type="single_hop" gold_atom_ids="a1"/>
    </question_annotations>"""
    questions, warnings = parse_questions_xml(question_xml)
    assert not warnings
    assert questions[0].question_id == "q1"
    assert questions[0].gold_atom_ids == ("a1",)


def test_span_and_question_cross_validation():
    atoms = [
        Atom(
            atom_id="atom_0",
            semantic_type="fact_claim",
            content="A",
            source_span="exact span",
            retrieval_text="A",
            is_answer_bearing=True,
            abstraction_level="evidence",
            conflict_group="c",
            confidence=1,
        ),
        Atom(
            atom_id="atom_1",
            semantic_type="fact_claim",
            content="B",
            source_span="fabricated",
            retrieval_text="B",
            is_answer_bearing=True,
            abstraction_level="evidence",
            conflict_group="c",
            confidence=1,
        ),
    ]
    qa = QuestionAnnotation(
        question_id="q0",
        question="?",
        question_type="single_hop",
        gold_atom_ids=["atom_0", "missing"],
        supporting_atom_ids=["atom_0"],
        relevant_atom_ids=["wrong"],
        is_irrelevant=False,
    )
    record = StructuredRecord(
        record_id="r",
        context="An exact   span is here.",
        atoms=atoms,
        qa_pairs=[qa],
        annotation_meta=AnnotationMeta(),
    )
    validate_record(record)
    assert record.atoms[0].span_valid
    assert not record.atoms[1].span_valid
    assert record.qa_pairs[0].gold_atom_ids == ["atom_0"]
    assert record.qa_pairs[0].relevant_atom_ids == ["atom_0"]
    assert record.qa_pairs[0].has_conflict
    assert record.annotation_meta.warnings


def test_irrelevant_clears_gold():
    atom = Atom(
        atom_id="atom_0",
        semantic_type="fact_claim",
        content="A",
        source_span="A",
        retrieval_text="A",
        is_answer_bearing=True,
        abstraction_level="evidence",
        confidence=1,
    )
    qa = QuestionAnnotation(
        question_id="q",
        question="?",
        question_type="single_hop",
        gold_atom_ids=["atom_0"],
        is_irrelevant=True,
    )
    record = validate_record(
        StructuredRecord(record_id="r", context="A", atoms=[atom], qa_pairs=[qa])
    )
    assert record.qa_pairs[0].gold_atom_ids == []
    assert record.qa_pairs[0].question_type == "irrelevant"


def test_generated_probe_can_be_rescued_as_in_appendix_f():
    probe = QuestionAnnotation(
        question_id="generated_irrelevant_0", question="What is Ascension Island's code?",
        question_type="irrelevant", is_irrelevant=True, is_generated_irrelevant=True,
        answers=(), answer_provenance="generated_irrelevant",
    )
    role = QuestionRoleAnnotation(
        question_id=probe.question_id, question_type="field_lookup", gold_atom_ids=("a0",)
    )
    [rescued] = join_question_roles((probe,), (role,))
    assert rescued.is_generated_irrelevant and not rescued.is_irrelevant
    assert rescued.question_type == "field_lookup" and rescued.gold_atom_ids == ["a0"]
    assert rescued.answers == ()  # Do not invent an answer absent from the source.


def test_question_annotation_failure_retains_validated_atom_only_record():
    class FakeAtomizer(ThreeStageXMLAtomizer):
        def __init__(self):
            super().__init__(MiniMaxConfig(api_key="test"))

        def _request(self, system, user):
            if "Decompose" in system:
                return ("<atoms><atom id='a0'><semantic_type>fact_claim</semantic_type>"
                        "<content>A</content><source_span>A</source_span>"
                        "<retrieval_text>A</retrieval_text><is_answer_bearing>true</is_answer_bearing>"
                        "<abstraction_level>evidence</abstraction_level><confidence>1</confidence>"
                        "</atom></atoms>")
            if "Generate plausible" in system:
                return "<irrelevant_questions/>"
            raise RuntimeError("annotation unavailable")

    result = FakeAtomizer().atomize("r", "A", [QuestionAnnotation(
        question_id="q", question="A?", answers=("A",), question_type="single_hop"
    )])
    assert result is not None and len(result.atoms) == 1 and result.atoms[0].span_valid
    assert result.qa_pairs == []
    assert result.annotation_meta.question_parse_status == "failed"
    assert result.annotation_meta.errors
