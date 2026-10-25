from __future__ import annotations

import ast
import os
import shutil
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook.py"


def _cell_with_parameter(parameter: str):
    """Load a real Marimo cell without executing the notebook's setup cells."""
    module = ast.parse(NOTEBOOK.read_text(encoding="utf-8"), filename=str(NOTEBOOK))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and any(
            argument.arg == parameter for argument in node.args.args
        ):
            cell = ast.FunctionDef(
                name="cell_under_test",
                args=node.args,
                body=node.body,
                decorator_list=[],
                returns=node.returns,
                type_comment=node.type_comment,
            )
            namespace: dict[str, object] = {}
            executable = ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[]))
            exec(compile(executable, str(NOTEBOOK), "exec"), namespace)
            return namespace["cell_under_test"]
    raise AssertionError(f"no notebook cell with parameter {parameter}")


def _cell_with_parameters(*parameters: str):
    module = ast.parse(NOTEBOOK.read_text(encoding="utf-8"), filename=str(NOTEBOOK))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and set(parameters).issubset(
            {argument.arg for argument in node.args.args}
        ):
            cell = ast.FunctionDef(
                name="cell_under_test", args=node.args, body=node.body,
                decorator_list=[], returns=node.returns, type_comment=node.type_comment,
            )
            namespace = {"__name__": "__main__"}
            executable = ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[]))
            exec(compile(executable, str(NOTEBOOK), "exec"), namespace)
            return namespace["cell_under_test"]
    raise AssertionError(f"no notebook cell with parameters {parameters}")


def test_notebook_defaults_predictions_to_predictions_directory(tmp_path: Path) -> None:
    values = _cell_with_parameters("REPO_DIR", "subprocess")(
        Path=Path, REPO_DIR=tmp_path, os=os, shutil=shutil, subprocess=subprocess,
    )

    assert values[3] == tmp_path / "outputs" / "molab-single-gpu" / "predictions" / "validation.jsonl"


def test_notebook_open_does_not_run_repo_update_or_dependency_install() -> None:
    class NoSubprocess:
        def run(self, *args, **kwargs):
            raise AssertionError("opening notebook must not run shell commands")

    _cell_with_parameters("Path", "shutil", "subprocess", "sys")(
        Path=Path, os=os, shutil=shutil, subprocess=NoSubprocess(), sys=sys,
    )


def test_evaluation_generates_and_scores_from_best_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    (output / "checkpoints" / "best.json").write_text('{"path":"best"}', encoding="utf-8")
    (output / "config").mkdir()
    (output / "config" / "resolved.yaml").write_text("run: test\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    atoms = tmp_path / "atoms.jsonl"
    source.touch()
    atoms.touch()
    calls: list[list[str]] = []

    _cell_with_parameter("score_button")(
        OUTPUT_DIR=output,
        PRED_EVAL_PATH=output / "predictions.jsonl",
        RAW_DATA_PATH=source,
        run_cmd=lambda command: calls.append(command),
        score_button=SimpleNamespace(value=True),
        selected_atom_data=lambda: atoms,
    )

    assert calls == [
        [
            "uv", "run", "doc2atom-evaluate",
            "--config", str(output / "config" / "resolved.yaml"),
            "--data", str(atoms),
            "--source", str(source),
            "--checkpoint", str(output / "checkpoints" / "best.json"),
            "--output", str(output / "predictions.jsonl"),
            "--split", "validation",
        ]
    ]


def test_evaluation_uses_last_when_best_is_absent(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "checkpoints").mkdir(parents=True)
    (output / "checkpoints" / "last.json").write_text('{"path":"last"}', encoding="utf-8")
    (output / "config").mkdir()
    (output / "config" / "resolved.yaml").touch()
    source = tmp_path / "source.jsonl"
    atoms = tmp_path / "atoms.jsonl"
    source.touch()
    atoms.touch()
    calls: list[list[str]] = []

    _cell_with_parameter("score_button")(
        OUTPUT_DIR=output,
        PRED_EVAL_PATH=output / "predictions.jsonl",
        RAW_DATA_PATH=source,
        run_cmd=lambda command: calls.append(command),
        score_button=SimpleNamespace(value=True),
        selected_atom_data=lambda: atoms,
    )

    assert calls[0][calls[0].index("--checkpoint") + 1] == str(
        output / "checkpoints" / "last.json"
    )


def test_evaluation_scores_existing_predictions_without_regeneration(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.touch()
    calls: list[list[str]] = []

    _cell_with_parameter("score_button")(
        OUTPUT_DIR=tmp_path,
        PRED_EVAL_PATH=predictions,
        RAW_DATA_PATH=tmp_path / "unused.jsonl",
        run_cmd=lambda command: calls.append(command),
        score_button=SimpleNamespace(value=True),
        selected_atom_data=lambda: pytest.fail("must not load atom data"),
    )

    assert calls == [["uv", "run", "doc2atom-evaluate", "--predictions", str(predictions)]]


def test_prepare_reuses_existing_atoms_without_openai_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = tmp_path / "source.jsonl"
    atoms = tmp_path / "atoms.jsonl"
    source.touch()
    atoms.touch()
    calls: list[list[str]] = []

    _cell_with_parameter("prepare_button")(
        ATOM_DATA_PATH=atoms,
        RAW_DATA_PATH=source,
        os=os,
        prepare_button=SimpleNamespace(value=True),
        run_cmd=lambda command: calls.append(command),
    )

    assert calls[0] == [
        "uv", "run", "doc2atom-validate", "--input", str(atoms),
        "--source", str(source), "--reject-span-mismatch",
    ]
