from __future__ import annotations

import os
from pathlib import Path

from doc_to_atom.runtime.molab_notebook import (
    find_molab_env_file,
    find_offline_wandb_runs,
    find_repo_dir,
    load_molab_environment,
    resolve_latest_checkpoint,
)


def test_finds_repo_and_sibling_molab_env(tmp_path: Path) -> None:
    notebook = tmp_path / "notebook.py"
    notebook.touch()
    repo = tmp_path / "doc-to-atom"
    (repo / "src" / "doc_to_atom").mkdir(parents=True)
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=molab-token\n", encoding="utf-8")

    assert find_repo_dir(notebook) == repo.resolve()
    assert find_molab_env_file(notebook, repo) == env_file.resolve()


def test_loads_molab_env_without_overriding_injected_secret(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "HF_TOKEN=file-token\n"
        "MINIMAX_API_KEY=minimax-token\n"
        "WANDB_API_KEY=wandb-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "injected-token")
    for name in ("MINIMAX_API_KEY", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    status = load_molab_environment(env_file)

    assert os.environ["HF_TOKEN"] == "injected-token"
    assert os.environ["MINIMAX_API_KEY"] == "minimax-token"
    assert os.environ["WANDB_API_KEY"] == "wandb-token"
    assert status == {
        "HF_TOKEN": True,
        "MINIMAX_API_KEY": True,
        "WANDB_API_KEY": True,
    }


def test_resume_requires_last_checkpoint_pointer(tmp_path: Path) -> None:
    output = tmp_path / "run"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step-00000001").mkdir()

    assert resolve_latest_checkpoint(output) is None

    pointer = checkpoints / "last.json"
    pointer.write_text('{"path":"step-00000001"}', encoding="utf-8")
    assert resolve_latest_checkpoint(output) == pointer.resolve()


def test_returns_only_wandb_offline_run_directories(tmp_path: Path) -> None:
    wandb = tmp_path / "wandb"
    first = wandb / "offline-run-20261019_100000-a"
    second = wandb / "offline-run-20261019_110000-b"
    first.mkdir(parents=True)
    second.mkdir()
    (wandb / "latest-run").mkdir()
    (wandb / "offline-run-file").write_text("not a directory", encoding="utf-8")

    assert find_offline_wandb_runs(wandb) == [first.resolve(), second.resolve()]
