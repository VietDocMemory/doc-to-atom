from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import ToyLM

from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.config import load_config
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector, discover_target_shapes
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.training.loop import run_training
from doc_to_atom.training.system import TeacherStudentSystem
from test_training_loop import ToyTokenizer, _record


@pytest.mark.integration
def test_one_example_loss_decreases(tmp_path: Path) -> None:
    torch.manual_seed(29)
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    system = TeacherStudentSystem(
        model,
        SharedAtomQueryEncoder(model, 8, 16, 4),
        AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2),
        TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32),
        MemoryComposer(),
        DynamicLoRAInjector(model, targets, rank=2, alpha=4),
    )
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update(
        {
            "learning_rate": 0.003,
            "max_epochs": 8,
            "max_steps": 8,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 0,
        }
    )

    run_training(system, ToyTokenizer(), [_record("train")], config, tmp_path)

    events = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    losses = [event["train"]["overall"]["total_loss"] for event in events]
    assert min(losses[-3:]) < losses[0]
