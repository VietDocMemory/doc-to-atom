from doc_to_atom.atomization.models import Atom,QuestionAnnotation,StructuredRecord
from doc_to_atom.data.io import read_jsonl,read_canonical_jsonl,write_jsonl

def test_training_reader_normalizes_legacy_probes_without_rewriting_input(tmp_path):
    record=StructuredRecord(record_id="r",context="A",atoms=[
        Atom(atom_id="a",semantic_type="fact_claim",content="A",source_span="A",
             retrieval_text="A",is_answer_bearing=True,abstraction_level="evidence",confidence=1)
    ],qa_pairs=[
        QuestionAnnotation(question_id="probe",question="Unknown?",question_type="field_lookup",
            gold_atom_ids=["a"],is_generated_irrelevant=True,is_irrelevant=False,
            answer_provenance="generated_irrelevant")
    ])
    path=tmp_path/"atoms.jsonl"
    write_jsonl([record],path)
    before=path.read_bytes()
    assert not read_jsonl(path)[0].qa_pairs[0].is_irrelevant
    qa=read_canonical_jsonl(path)[0].qa_pairs[0]
    assert qa.is_irrelevant and qa.question_type=="irrelevant"
    assert qa.gold_atom_ids==qa.supporting_atom_ids==qa.relevant_atom_ids==[]
    assert path.read_bytes()==before
