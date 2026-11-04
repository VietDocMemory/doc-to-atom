import json

import pytest

from doc_to_atom.atomization import cli
from doc_to_atom.atomization.models import Atom, StructuredRecord
from doc_to_atom.atomization.validation import source_span_matches
from doc_to_atom.data.io import validate_jsonl


def record(span="A"):
    return StructuredRecord(record_id="cached", context="A", atoms=[Atom(
        atom_id="a", semantic_type="fact_claim", content="A", source_span=span,
        retrieval_text="A", is_answer_bearing=True, abstraction_level="evidence", confidence=1,
    )])


def test_diagnostics_show_actual_bad_span_and_leave_files_unchanged(tmp_path):
    path = tmp_path / "atoms.jsonl"
    path.write_text(record("fabricated quote").model_dump_json() + "\n", encoding="utf-8")
    before = path.read_bytes()
    report = validate_jsonl(path)
    assert report["invalid_spans"] == 1
    assert report["invalid_span_details"] == [{
        "record_id": "cached", "atom_id": "a", "source_span": "fabricated quote", "context": "A",
    }]
    with pytest.raises(ValueError, match="cached/a"):
        validate_jsonl(path, reject_span_mismatch=True)
    assert path.read_bytes() == before


def test_span_match_supports_whitespace_but_never_empty_span():
    assert source_span_matches("New  York", "New York")
    assert not source_span_matches("New  York", "New York", fuzzy_whitespace=False)
    assert not source_span_matches("", "A")
    assert not source_span_matches("   ", "A")


def test_retry_rechecks_actual_spans_even_if_serialized_flag_is_true(tmp_path, monkeypatch):
    source, output = tmp_path / "raw.jsonl", tmp_path / "atoms.jsonl"
    source.write_text(json.dumps({"id": "cached", "context": "A", "questions": []}) + "\n")
    bad = record("B")
    assert bad.atoms[0].span_valid  # Stale historical flag previously prevented retry.
    output.write_text(bad.model_dump_json() + "\n", encoding="utf-8")
    original = output.read_bytes()
    calls = []

    class FakeAtomizer:
        def __init__(self, config):
            pass

        def atomize(self, *args):
            calls.append(args[0])
            return record()

    monkeypatch.setattr(cli, "MiniMaxAtomizer", FakeAtomizer)
    monkeypatch.setenv("MINIMAX_API_KEY", "test")
    stats = cli.atomize_file(source, output, resume=True, retry_invalid=True)
    assert stats["records_cached"] == 0
    assert calls == ["cached"]
    [backup] = tmp_path.glob("*.before-retry-*.jsonl")
    assert backup.read_bytes() == original
    assert validate_jsonl(output)["invalid_spans"] == 0


def test_new_bad_spans_are_saved_but_not_reported_as_success(tmp_path, monkeypatch):
    source, output = tmp_path / "raw.jsonl", tmp_path / "atoms.jsonl"
    source.write_text(json.dumps({"id": "cached", "context": "A", "questions": []}) + "\n")
    calls = []

    class FakeAtomizer:
        def __init__(self, config):
            pass

        def atomize(self, *args):
            calls.append(args[0])
            return record("B")

    monkeypatch.setattr(cli, "MiniMaxAtomizer", FakeAtomizer)
    monkeypatch.setenv("MINIMAX_API_KEY", "test")
    with pytest.raises(RuntimeError, match="cached/a"):
        cli.atomize_file(source, output, resume=False)
    assert calls == ["cached"]  # Never loop over paid retries automatically.
    assert output.is_file()
    assert (tmp_path / "atoms.jsonl.raw-responses.jsonl").is_file()
    assert validate_jsonl(output)["invalid_spans"] == 1
