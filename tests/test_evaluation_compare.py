from __future__ import annotations

import json

import pytest

from doc_to_atom.evaluation.compare import compare_prediction_files


def _write(path, *, prediction="Paris", source="source-hash", protocol="short_answer_v1"):
    row = {
        "question_id": "q1", "question": "Where?", "gold": ["Paris"],
        "prediction": prediction, "is_irrelevant": False, "split": "validation",
        "source_fingerprint": source, "atomized_fingerprint": "atoms-hash",
        "generation_protocol": protocol, "scoring_protocol": "squad_v1_multiref",
        "evaluation_settings": {"do_sample": False, "max_new_tokens": 8, "stop": "first_answer_line"},
        "prompt_id": "model-specific-prompt",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_compare_accepts_different_prompt_ids_on_same_dataset_and_protocol(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write(first)
    _write(second, prediction="Lyon")

    report = compare_prediction_files({"doc2atom": first, "baseline": second})

    assert report["comparable"] is True
    assert report["models"]["doc2atom"]["f1"] == 1.0
    assert report["models"]["baseline"]["f1"] == 0.0


def test_compare_rejects_mismatched_dataset_fingerprint(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write(first)
    _write(second, source="other-source")

    with pytest.raises(ValueError, match="fingerprint"):
        compare_prediction_files({"doc2atom": first, "baseline": second})


def test_compare_labels_legacy_protocol_as_noncomparable(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    _write(first)
    second.write_text(json.dumps({
        "question_id": "q1", "question": "Where?", "gold": ["Paris"],
        "prediction": "Paris", "is_irrelevant": False, "split": "validation",
    }) + "\n", encoding="utf-8")

    report = compare_prediction_files({"new": first, "legacy": second})

    assert report["comparable"] is False
    assert "legacy" in report["reasons"][0]
