import importlib.util
from pathlib import Path

import pytest

from doc_to_atom.atomization.models import Atom, StructuredRecord

spec = importlib.util.spec_from_file_location(
    "source_repairs", Path(__file__).parents[1] / "scripts/repair_annotations.py"
)
repairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repairs)


def sample(span="After procuring necessary gifts"):
    return StructuredRecord(record_id="r", context="After procurring necessary gifts", atoms=[Atom(
        atom_id="a", semantic_type="fact_claim", content="Gifts were prepared.",
        source_span=span, retrieval_text="gifts", is_answer_bearing=True,
        abstraction_level="evidence", confidence=1,
    )])


def test_span_repair_copies_source_typo_and_logs_change():
    record = sample()
    replacements = [["After procuring necessary gifts", "After procurring necessary gifts"]]
    repairs.repair_spans(record, replacements)
    assert record.atoms[0].source_span == record.context
    assert record.annotation_meta.transformations[0].code == "RESTORE_VERBATIM_SOURCE_SPAN"
    assert record.annotation_meta.transformations[0].details["before"] == "After procuring necessary gifts"


def test_unreviewed_span_cannot_be_fuzzy_repaired():
    record = sample("fabricated quotation")
    with pytest.raises(ValueError, match="Unreviewed source span"):
        repairs.repair_spans(record, [])


def test_reviewed_plan_cannot_invent_source_text():
    record = sample(record_context := "After procurring necessary gifts")
    with pytest.raises(ValueError, match="Plan span not in authoritative context"):
        repairs.apply_plan(record, {"atoms": [{"record_id": "r", "atom_id": "a",
            "source_span": "Not present", "reason": "test"}], "new_atoms": []})
    assert record.context == record_context


def test_existing_valid_span_is_never_changed_by_replacements():
    record = sample("After procurring necessary gifts")
    repairs.repair_spans(record, [["procurring", "procuring"]])
    assert record.atoms[0].source_span == record.context
    assert not record.annotation_meta.transformations
