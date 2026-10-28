"""Exercise notebook helpers without Molab, paid APIs, GPU, or a marimo import."""

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from doc_to_atom.config import load_and_resolve_config


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook_molab_merged.py"


def run_cell(definition, **values):
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(child, ast.FunctionDef) and child.name == definition
                for child in ast.walk(node))
    )
    cell.decorator_list = []
    namespace = {"__file__": str(NOTEBOOK)}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["_"](**values)


def test_command_dedents_scripts_and_isolates_project_environment(tmp_path, monkeypatch):
    uv = tmp_path / "uv"
    uv.touch()
    repo = tmp_path / "doc-to-atom"
    repo.mkdir()
    (tmp_path / ".env").write_text("PLACEHOLDER=not-a-real-secret\n")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        compile(cmd[cmd.index("-c") + 1], "<embedded>", "exec")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_PROJECT", "UV_WORKING_DIRECTORY", "UV_PYTHON"):
        monkeypatch.setenv(name, "notebook-sandbox")
    run_cmd, _ = run_cell(
        "run_cmd", NOTEBOOK_DIR=repo, Path=Path, REPO_DIR=repo,
        UV=str(uv), hashlib=hashlib, os=os,
    )
    run_cmd(["uv", "run", "python", "-c", "\n    print('ok')\n"], capture=True)
    assert seen["cmd"][:4] == [str(uv), "run", "--env-file", str(tmp_path / ".env")]
    assert seen["cwd"] == repo
    assert not any(name in seen["env"] for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_PYTHON"))


@pytest.mark.parametrize("mode", ["smoke", "standard", "four_stage"])
@pytest.mark.parametrize("tracking", ["disabled", "offline", "online"])
def test_config_builder_round_trip_and_reuse(mode, tracking, tmp_path, monkeypatch):
    destination = tmp_path / "notebook.yaml"

    def execute_builder(args):
        monkeypatch.setattr(sys, "argv", ["-c", *map(str, args[5:])])
        exec(compile(textwrap.dedent(args[4]), "<config-builder>", "exec"), {})

    (prepare_config,) = run_cell(
        "prepare_config", REPO_DIR=ROOT,
        config_button=SimpleNamespace(value=False), ensure_repo_ready=lambda: None,
        mode_choice=SimpleNamespace(value=mode), notebook_config=destination,
        run_cmd=execute_builder, tracking_choice=SimpleNamespace(value=tracking),
    )
    assert prepare_config() == destination
    before = destination.read_bytes()
    prepare_config()
    assert destination.read_bytes() == before
    config = load_and_resolve_config(destination, require_runnable=True)
    assert config["runtime"]["wandb_mode"] == tracking
    if mode == "smoke":
        assert config["training"]["max_steps"] == 1
        assert not config["training"]["stage4_only_best"]
        assert config["training"]["validation_every_steps"] == 1
    if mode == "four_stage":
        assert config["training"]["curriculum_boundaries"] == [200, 400, 600]
        assert config["training"]["max_steps"] == 800
    config["training"]["max_steps"] = 999
    # JSON is valid YAML; a conflicting existing run must never be overwritten.
    destination.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="Existing run config differs"):
        prepare_config()


def test_embedded_scripts_compile_and_validation_receives_source_paths():
    calls = []
    # Compile subprocess scripts without API calls or GPU work.
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(target, ast.Name) and target.id in {"check", "builder"}
                   for target in node.targets):
                compile(textwrap.dedent(node.value.value), "<notebook-script>", "exec")
    _, validate_pair = run_cell(
        "validate_pair", REPO_DIR=ROOT, atoms_path=NOTEBOOK,
        raw_path=NOTEBOOK, run_cmd=lambda args: calls.append(args),
    )
    assert validate_pair()
    assert len(calls) == 1
    assert calls[0][-2:] == [str(NOTEBOOK), str(NOTEBOOK)]
