import importlib.util
import json
from pathlib import Path

import pytest

from doc_to_atom.atomization.models import AnnotationMeta, Atom, QuestionAnnotation, StructuredRecord
from doc_to_atom.atomization.provenance import RawProviderResponse

spec = importlib.util.spec_from_file_location(
    "pilot_repair", Path(__file__).parents[1] / "scripts/repair_annotation_pilot.py"
)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

PROBES = ("irrelevant_questions><question>Which player was named Most Valuable Player of Super Bowl 50?"
          "</question><question>Who performed the halftime show at Super Bowl 50?"
          "</question></irrelevant_questions>")


def test_probe_recovery_changes_only_one_character():
    recovered, questions = repair.recover_probe_xml(PROBES)
    assert recovered[1:] == PROBES
    assert len(questions) == 2


@pytest.mark.parametrize("text", [PROBES[:-1], "<" + PROBES, PROBES.replace("halftime", "other"),
                                  "prefix" + PROBES])
def test_probe_recovery_refuses_unreviewed_or_incomplete_payload(text):
    with pytest.raises((ValueError, repair.ET.ParseError)):
        repair.recover_probe_xml(text)


def pair():
    source = {"id": "r", "dataset": "squad", "split": "validation", "context": "Paris.",
              "questions": [{"id": "q", "question": "Where?", "answers": ["Paris", "Paris."]}]}
    record = StructuredRecord(record_id="r", dataset="squad", split="validation", context="Paris.",
        atoms=[Atom(atom_id="a", semantic_type="entity_attribute", content="Paris", source_span="Paris.",
                    retrieval_text="Paris", is_answer_bearing=True, abstraction_level="evidence", confidence=1)],
        qa_pairs=[QuestionAnnotation(question_id="q", question="Where?", answers=["Paris", "Paris."],
                                    answer_provenance="source", question_type="single_hop",
                                    gold_atom_ids=["a"], relevant_atom_ids=["a"])],
        annotation_meta=AnnotationMeta(num_atoms=1, num_questions=1))
    return record, source


def test_strict_gate_does_not_canonicalize_overlap_or_answers():
    record, source = pair()
    assert repair.strict_report([record], {"r": source})["accepted"]
    record.qa_pairs[0].supporting_atom_ids = ["a"]
    before = record.model_dump_json()
    assert not repair.strict_report([record], {"r": source})["accepted"]
    assert before == record.model_dump_json()
    record.qa_pairs[0].supporting_atom_ids = []
    record.qa_pairs[0] = record.qa_pairs[0].model_copy(update={"answers": ("Paris",)})
    report = repair.strict_report([record], {"r": source})
    assert not report["accepted"]
    assert report["source_answer_mismatches"] == 1


def test_strict_gate_requires_completeness_and_exact_span():
    record, source = pair()
    assert not repair.strict_report([], {"r": source})["accepted"]
    record.atoms[0].source_span = "Paris ."
    report = repair.strict_report([record], {"r": source})
    assert not report["accepted"]
    assert report["invalid_spans"] == 1


def test_prepare_preserves_originals_and_drops_edges_without_semantic_mapping():
    original, source = pair()
    original.record_id = repair.TARGET
    nodes = []
    for i in range(1, 13):
        atom = original.atoms[0].model_copy(update={"atom_id": f"atom_{i}"})
        if i == 8:
            atom.source_span = "Paris"
        node = repair.ET.Element("atom")
        for key, value in atom.model_dump(exclude={"relations", "span_valid", "conflict_group"}).items():
            repair.ET.SubElement(node, key).text = str(value).lower() if isinstance(value, bool) else str(value)
        relations = repair.ET.SubElement(node, "relations")
        if i in (5, 6):
            repair.ET.SubElement(relations, "relation", type="participates", target="atom_3")
        nodes.append(repair.ET.tostring(node, encoding="unicode"))
    nodes[-1] = nodes[-1].replace("</semantic_type>", "</emantic_type>")

    def raw(stage, text):
        return RawProviderResponse.capture(record_id=repair.TARGET, stage=stage, attempt=1,
                                           provider="minimax", model="MiniMax-M3.1-Flash", content=text)

    responses = {"irrelevant_probe": raw("irrelevant_probe", PROBES),
                 "decomposition": raw("decomposition", "<atoms>" + "".join(nodes) + "</atoms>")}
    before = original.model_dump_json()
    raw_before = json.dumps({k: v.model_dump() for k, v in responses.items()})
    record, probes, events, _, _ = repair.prepare_record(original, source, responses)
    assert len(record.atoms) == 12
    assert len(probes) == 2
    assert next(a for a in record.atoms if a.atom_id == "atom_8").source_span == "Paris."
    assert not next(a for a in record.atoms if a.atom_id == "atom_5").relations
    assert original.model_dump_json() == before
    assert json.dumps({k: v.model_dump() for k, v in responses.items()}) == raw_before
    assert "DROP_UNSUPPORTED_RELATION" in {event["code"] for event in events}
