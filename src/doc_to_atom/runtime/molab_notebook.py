from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


SECRET_NAMES = (
    "HF_TOKEN",
    "MINIMAX_API_KEY",
    "WANDB_API_KEY",
)


def find_repo_dir(notebook_file: str | Path) -> Path:
    """Locate the checkout placed next to the Molab notebook."""
    notebook_dir = Path(notebook_file).resolve().parent
    candidates = (notebook_dir / "doc-to-atom", notebook_dir)
    for candidate in candidates:
        if (candidate / "src" / "doc_to_atom").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find the doc-to-atom checkout next to notebook.py. "
        "Clone it as '<notebook directory>/doc-to-atom'."
    )


def find_molab_env_file(notebook_file: str | Path, repo_dir: str | Path) -> Path | None:
    """Prefer Molab's sibling .env, then fall back to a repo-local .env."""
    notebook_env = Path(notebook_file).resolve().parent / ".env"
    repo_env = Path(repo_dir).resolve() / ".env"
    for candidate in (notebook_env, repo_env):
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_molab_environment(env_file: str | Path | None) -> dict[str, bool]:
    """Load Molab secrets without overriding variables injected by the runtime."""
    if env_file is not None:
        load_dotenv(Path(env_file), override=False)
    return {name: bool(os.getenv(name)) for name in SECRET_NAMES}


def resolve_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the exact-resume pointer only when it exists as a file."""
    pointer = Path(output_dir).resolve() / "checkpoints" / "last.json"
    return pointer if pointer.is_file() else None


def find_offline_wandb_runs(wandb_dir: str | Path) -> list[Path]:
    """List concrete W&B offline run directories in deterministic order."""
    root = Path(wandb_dir).resolve()
    if not root.is_dir():
        return []
    return sorted(path.resolve() for path in root.glob("offline-run-*") if path.is_dir())
