"""Contracts for a native, button-gated Molab notebook (no shell execution)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from doc_to_atom.atomization import cli as annotation
from doc_to_atom.evaluation import cli as evaluation


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook.py"


def _cells():
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and any(isinstance(dec, ast.Attribute) and dec.attr == "cell"
                    for dec in node.decorator_list)]


def test_notebook_uses_native_apis_and_requires_explicit_buttons():
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert len(_cells()) == 6
    assert "subprocess" not in source and "uv run" not in source
    assert "atomize_file(" in source
    assert "train_execute(" in source and "run_training(" in source
    assert "evaluate_checkpoint(" in source and "summarize(" in source
    assert "if prepare_button.value:" in source
    assert "if train_button.value or resume_button.value:" in source
    assert "if evaluate_button.value:" in source
    assert "checkpoints/best.json" in source
    assert "predictions/val.jsonl" in source


def test_native_atomization_resume_preserves_cached_data(tmp_path, monkeypatch):
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "atoms.jsonl"
    source.write_text('{"id":"a","context":"hello","questions":[]}\n', encoding="utf-8")
    output.write_text('{"record_id":"a"}\n', encoding="utf-8")
    original = output.read_bytes()
    calls = []

    class FakeAtomizer:
        def __init__(self, config):
            calls.append("constructed")

    monkeypatch.setattr(annotation, "OpenAIAtomizer", FakeAtomizer)
    monkeypatch.setattr(annotation.OpenAIConfig, "from_env", lambda: object())
    # Resume parses valid schema records; an existing malformed cached row must
    # fail rather than silently overwrite the only copy.
    with pytest.raises(Exception):
        annotation.atomize_file(source, output, resume=True)
    assert output.read_bytes() == original


def test_native_evaluator_rejects_missing_generation_inputs():
    import argparse

    args = argparse.Namespace(predictions=None, config=None, data=None, source=None,
                              checkpoint=None, output=None)
    with pytest.raises(ValueError, match="generation requires"):
        evaluation.evaluate_checkpoint(args)
