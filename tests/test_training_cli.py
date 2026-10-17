from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import doc_to_atom.training.cli as cli_module
import doc_to_atom.training.factory as factory_module


def test_parser_accepts_resume_latest_and_offline_tracking() -> None:
    args = cli_module.build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--data",
            "data.jsonl",
            "--output",
            "run",
            "--resume",
            "latest",
            "--wandb-mode",
            "offline",
        ]
    )

    assert args.resume == "latest"
    assert args.wandb_mode == "offline"
    assert args.output_dir == "run"


def test_factory_rejects_unsafe_gradient_checkpointing_before_model_allocation() -> None:
    config = {
        "runtime": {"gradient_checkpointing": True},
        "compiler": {"micro_kv": {"enabled": False}},
    }

    with pytest.raises(factory_module.UnsupportedRuntimeFeature, match="recomputation"):
        factory_module.validate_runtime_features(config)


def test_factory_rejects_incomplete_micro_kv() -> None:
    config = {
        "runtime": {"gradient_checkpointing": False},
        "compiler": {"micro_kv": {"enabled": True}},
    }

    with pytest.raises(NotImplementedError, match="PAPER_UNSPECIFIED"):
        factory_module.validate_runtime_features(config)


def test_factory_rejects_unwired_distributed_backend() -> None:
    config = {
        "runtime": {"gradient_checkpointing": False, "distributed_backend": "deepspeed_zero2"},
        "compiler": {"micro_kv": {"enabled": False}},
    }

    with pytest.raises(factory_module.UnsupportedRuntimeFeature, match="distributed checkpoint"):
        factory_module.validate_runtime_features(config)


def test_cli_fresh_run_snapshots_config_and_resume_uses_latest_pointer(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"
    calls: list[dict] = []

    def system_factory(config, enable_reranker=True):
        return nn.Linear(2, 1), object(), ["target"]

    def training_runner(system, tokenizer, records, config, output_dir, **kwargs):
        calls.append(kwargs)
        checkpoint = output_dir / "checkpoints" / "step-00000001"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "COMMITTED").write_text("v2\n", encoding="utf-8")
        (output_dir / "checkpoints" / "last.json").write_text(
            json.dumps({"path": checkpoint.name, "optimizer_step": 1}), encoding="utf-8"
        )
        return {"optimizer_steps": 1}

    common = [
        "--config",
        str(root / "configs/experiments/small_overfit.yaml"),
        "--data",
        str(root / "data/squad_small_atoms_openai.jsonl"),
        "--output",
        str(output),
        "--wandb-mode",
        "disabled",
    ]
    fresh = cli_module.execute(
        cli_module.build_parser().parse_args(common),
        system_factory=system_factory,
        training_runner=training_runner,
    )
    resumed = cli_module.execute(
        cli_module.build_parser().parse_args([*common, "--resume", "latest"]),
        system_factory=system_factory,
        training_runner=training_runner,
    )

    assert fresh["training"]["optimizer_steps"] == 1
    assert resumed["training"]["optimizer_steps"] == 1
    assert calls[0]["resume_from"] is None
    assert calls[1]["resume_from"] == output / "checkpoints" / "last.json"
    assert (output / "config" / "requested.yaml").read_text(encoding="utf-8")
    assert (output / "config" / "resolved.yaml").is_file()
    assert (output / "manifests" / "run.json").is_file()
    assert (output / "reports" / "initialization.json").is_file()


def test_cli_resume_requires_existing_latest_pointer(tmp_path: Path) -> None:
    args = argparse.Namespace(
        resume="latest",
        output_dir=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="last.json"):
        cli_module.resolve_resume_argument(args)
