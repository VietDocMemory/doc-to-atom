"""Appendix B behavior using fixed local responses, with no model/API calls."""
import json
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from doc_to_atom.atomization.client import AnnotationResponseError, MiniMaxAPIError
from doc_to_atom.atomization.local import LocalQwenAtomizer, LocalQwenConfig
from doc_to_atom.atomization.models import QuestionAnnotation, StructuredRecord
from doc_to_atom.atomization.parser import parse_atoms_xml, parse_question_roles_xml
from doc_to_atom.atomization.validation import validate_record


def atom_xml(atom_id="a0", span="Paris is in France.", relations=""):
    return (f'<atom id="{atom_id}"><semantic_type>fact_claim</semantic_type>'
            f'<content>Paris is in France.</content><source_span>{span}</source_span>'
            '<retrieval_text>Paris France</retrieval_text><is_answer_bearing>true</is_answer_bearing>'
            '<abstraction_level>evidence</abstraction_level><confidence>0.9</confidence>'
            f'<relations>{relations}</relations></atom>')


def role_xml(qid="q0", gold="a0", kind="single_hop", extra=""):
    ids = f'<atom_id>{gold}</atom_id>' if gold else ''
    return (f'<question_annotation id="{qid}"><question_type>{kind}</question_type>'
            f'<gold_atom_ids>{ids}</gold_atom_ids>{extra}</question_annotation>')


class FixedLocal(LocalQwenAtomizer):
    def __init__(self, responses):
        super().__init__(LocalQwenConfig(irrelevant_probe_count=1))
        self.responses = iter(responses)
        self.requests = []

    def _load_model(self):
        pass

    def _request(self, system, user):
        self.requests.append((system, user))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, tuple):
            result, self._response_partial = result
            self._response_error = 'incomplete completion (length)'
        return result


def source():
    return QuestionAnnotation(question_id="q0", question="Where is Paris?", answers=["France"],
                              answer_provenance="source", question_type="single_hop")


def run(atomizer):
    return atomizer.atomize("r0", "Paris is in France.", [source()], dataset="squad")


def test_span_validation_precedes_qa_and_invalid_evidence_is_not_presented():
    agent = FixedLocal([
        '<atoms>'+atom_xml()+atom_xml('a_bad','Fabricated quotation.')+'</atoms>',
        '<irrelevant_questions/>', role_xml(extra='<supporting_atom_ids>a_bad</supporting_atom_ids>'),
    ])
    record = run(agent)
    assert [a.span_valid for a in record.atoms] == [True, False]
    bank = ET.fromstring('<root>'+agent.requests[2][1].split('\n',1)[1]+'</root>').find('atoms')
    assert [node.attrib['id'] for node in bank] == ['a0']
    assert bank[0].findtext('source_span') == 'Paris is in France.'
    assert record.qa_pairs[0].supporting_atom_ids == []
    codes = [t.code for t in record.annotation_meta.transformations]
    assert 'MARK_INVALID_SOURCE_SPAN' in codes
    assert 'DROP_INVALID_SPAN_ATOM_REFERENCE' in codes


def test_decomposition_failure_discards_without_calling_other_stages():
    agent = FixedLocal(['<atoms/>'])
    assert run(agent) is None
    assert len(agent.requests) == 1
    assert agent.last_annotation_meta.decomposition_parse_status == 'failed'
    assert agent.last_errors


def test_all_unanchored_atoms_discard_sample():
    agent = FixedLocal(['<atoms>'+atom_xml(span='invented')+'</atoms>'])
    assert run(agent) is None
    assert agent.last_annotation_meta.decomposition_parse_status == 'failed'
    assert len(agent.requests) == 1


def test_probe_failure_is_logged_and_qa_still_runs():
    agent = FixedLocal([atom_xml(), AnnotationResponseError('bad annotation'), role_xml()])
    record = run(agent)
    assert record.qa_pairs[0].gold_atom_ids == ['a0']
    assert record.annotation_meta.irrelevant_parse_status == 'failed_ignored'
    assert any('probe generation failed' in x for x in record.annotation_meta.warnings)
    from doc_to_atom.data.semantic_validation import audit_records
    report = audit_records([record])
    assert report['errors'] == report['parse_errors'] == 0
    assert any(i['code'] == 'auxiliary_probe_failure' and i['severity'] == 'warning'
               for i in report['issues'])


def test_question_failure_retains_atom_only_record():
    agent = FixedLocal([atom_xml(), '<irrelevant_questions/>', AnnotationResponseError('bad annotation')])
    record = run(agent)
    assert len(record.atoms) == 1 and record.qa_pairs == []
    assert record.annotation_meta.question_parse_status == 'failed'
    assert 'provider=local_qwen' in record.annotation_meta.prompt_provenance


def test_partial_decomposition_recovers_complete_atoms_without_retry():
    agent = FixedLocal([('<atoms>'+atom_xml()+'<atom>', True), '<irrelevant_questions/>', role_xml()])
    record = run(agent)
    assert len(record.atoms) == 1 and len(agent.requests) == 3
    assert any('partial output' in x for x in record.annotation_meta.warnings)
    assert 'incomplete' not in record.annotation_meta.errors


def test_partial_qa_requires_all_authoritative_question_ids():
    agent = FixedLocal([atom_xml(), '<irrelevant_questions/>', ('<question_annotations><question_annotation>', True)])
    record = run(agent)
    assert record.qa_pairs == []
    assert record.annotation_meta.question_parse_status == 'failed'


def test_partial_qa_can_recover_a_complete_authoritative_set():
    agent = FixedLocal([atom_xml(), '<irrelevant_questions/>',
                        ('<question_annotations>'+role_xml(), True)])
    record = run(agent)
    assert record.annotation_meta.question_parse_status == 'ok'
    assert record.qa_pairs[0].gold_atom_ids == ['a0']
    assert len(agent.requests) == 3


def test_generated_probes_cannot_be_rescued_by_annotation():
    agent = FixedLocal([atom_xml(), '<irrelevant_questions><question>Unknown?</question></irrelevant_questions>',
                        '<question_annotations>'+role_xml()+role_xml('generated_irrelevant_0')+'</question_annotations>'])
    record = run(agent)
    probe = record.qa_pairs[1]
    assert probe.is_irrelevant and probe.is_generated_irrelevant
    assert probe.question_type == 'irrelevant'
    assert probe.gold_atom_ids == probe.supporting_atom_ids == probe.relevant_atom_ids == []
    assert any('forced irrelevant' in x for x in record.annotation_meta.warnings)


def test_invalid_relation_does_not_drop_valid_atom_and_duplicate_ids_are_logged():
    atoms, warnings = parse_atoms_xml('<atoms>'+atom_xml(relations='<relation type="bad" target="a1"/>')+atom_xml()+'</atoms>')
    assert len(atoms) == 1 and atoms[0].relations == []
    assert any('relation dropped' in x for x in warnings)
    assert any('duplicate atom_id' in x for x in warnings)


@pytest.mark.parametrize('stage', [0, 1, 2])
def test_fatal_infrastructure_failure_never_continues(stage):
    responses = [atom_xml(), '<irrelevant_questions/>', role_xml()]
    responses[stage] = MiniMaxAPIError('CUDA OOM / transport failure')
    agent = FixedLocal(responses)
    with pytest.raises(MiniMaxAPIError):
        run(agent)
    assert len(agent.requests) == stage+1


def test_optional_irrelevance_flag_is_parsed_strictly():
    roles, notices = parse_question_roles_xml(role_xml(extra='<is_irrelevant>true</is_irrelevant>'))
    assert roles[0].is_irrelevant is True and not notices
    roles, notices = parse_question_roles_xml(role_xml(extra='<is_irrelevant>maybe</is_irrelevant>'))
    assert roles == [] and notices[0].severity == 'error'


def test_postprocessing_is_deterministic_and_idempotent():
    responses = [atom_xml(), '<irrelevant_questions/>', role_xml(gold='missing')]
    first, second = run(FixedLocal(responses)), run(FixedLocal(responses))
    assert first.model_dump() == second.model_dump()
    before = first.model_dump()
    assert validate_record(first).model_dump() == before


def test_cli_logs_discarded_sample_metadata_and_keeps_later_samples(tmp_path, monkeypatch):
    from doc_to_atom.atomization import cli, local

    agent = FixedLocal(['<atoms/>', atom_xml(), '<irrelevant_questions/>', role_xml()])
    monkeypatch.setattr(local, 'LocalQwenAtomizer', Mock(return_value=agent))
    monkeypatch.setattr(cli.MiniMaxConfig, 'from_env', Mock(side_effect=AssertionError('API forbidden')))
    raw = tmp_path/'raw.jsonl'
    record = {'context':'Paris is in France.', 'questions':[{'id':'q0','question':'Where is Paris?','answers':['France']}]}
    raw.write_text('\n'.join(json.dumps({'id':rid,**record}) for rid in ['bad','good'])+'\n')
    output = tmp_path/'atoms.jsonl'
    with pytest.raises(RuntimeError, match='discarded 1'):
        cli.atomize_file(raw, output, provider='local_qwen')
    quality = [json.loads(l) for l in (tmp_path/'atoms.jsonl.annotation-meta.jsonl').read_text().splitlines()]
    assert [r['status'] for r in quality] == ['discarded','annotated']
    assert quality[0]['annotation_meta']['decomposition_parse_status'] == 'failed'
    assert json.loads(output.read_text())['record_id'] == 'good'
