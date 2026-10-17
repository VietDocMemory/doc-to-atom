from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

import doc_to_atom.training.checkpoint as checkpoint_module


def _seed() -> None:
    random.seed(314)
    np.random.seed(314)
    torch.manual_seed(314)


def _build():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler


def _step(model, optimizer, scheduler) -> None:
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(3, 4)
    scale = random.random() + float(np.random.random())
    loss = model(inputs).square().mean() * scale
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_interrupted_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    _seed()
    uninterrupted, full_optimizer, full_scheduler = _build()
    for _ in range(4):
        _step(uninterrupted, full_optimizer, full_scheduler)
    expected_parameters = [parameter.detach().clone() for parameter in uninterrupted.parameters()]
    expected_lr = full_scheduler.get_last_lr()
    expected_next_rng = (random.random(), float(np.random.random()), torch.rand(3))

    _seed()
    interrupted, optimizer, scheduler = _build()
    for _ in range(2):
        _step(interrupted, optimizer, scheduler)
    checkpoint = tmp_path / "step-2"
    checkpoint_module.save_training_checkpoint(
        {"model": interrupted},
        checkpoint,
        state=checkpoint_module.TrainerState(
            epoch=0,
            record_cursor=2,
            micro_step=2,
            optimizer_step=2,
            curriculum_step=2,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
    )

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed, resumed_optimizer, resumed_scheduler = _build()
    restored = checkpoint_module.load_training_checkpoint(
        {"model": resumed},
        checkpoint,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        restore_rng=True,
    )
    for _ in range(restored.record_cursor, 4):
        _step(resumed, resumed_optimizer, resumed_scheduler)

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(resumed.parameters(), expected_parameters, strict=True)
    )
    assert resumed_scheduler.get_last_lr() == expected_lr
    actual_next_rng = (random.random(), float(np.random.random()), torch.rand(3))
    assert actual_next_rng[:2] == expected_next_rng[:2]
    assert torch.equal(actual_next_rng[2], expected_next_rng[2])
