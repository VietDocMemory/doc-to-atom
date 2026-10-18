from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from doc_to_atom.training.checkpoint import (
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)
from doc_to_atom.training.distributed import AccelerateTrainingRuntime, DistributedEpochPlan
from doc_to_atom.training.state import TrainerState


SOURCE_SIZE = 7
GLOBAL_BATCH = 4
TOTAL_OPTIMIZER_STEPS = 2
SEED = 314159


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("continuous", "interrupt", "resume"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--crash-rank", type=int)
    return parser.parse_args()


def _config(world_size: int, accumulation_steps: int) -> dict[str, Any]:
    return {
        "training": {
            "gradient_accumulation_steps": accumulation_steps,
            "micro_batch_size": 1,
            "effective_global_batch": GLOBAL_BATCH,
        },
        "runtime": {"distributed_backend": "accelerate", "device": "cpu"},
    }


def _example(sample_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = float(sample_id + 1)
    inputs = torch.tensor([[value, value * value / 10.0]], dtype=torch.float64)
    target = torch.tensor([[0.35 * value - 0.2]], dtype=torch.float64)
    return inputs, target


def _optimizer_snapshot(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for parameter_state in optimizer.state_dict()["state"].values():
        item: dict[str, Any] = {}
        for name, value in sorted(parameter_state.items()):
            if isinstance(value, torch.Tensor):
                item[name] = value.detach().cpu().to(torch.float64).reshape(-1).tolist()
            else:
                item[name] = value
        snapshot.append(item)
    return snapshot


def main() -> None:
    args = _parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    accumulation_steps = GLOBAL_BATCH // world_size
    if args.crash_rank == rank:
        raise RuntimeError(f"intentional rank crash on rank {rank}")

    runtime = AccelerateTrainingRuntime(_config(world_size, accumulation_steps))
    torch.manual_seed(20261018)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).to(torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    model, optimizer, scheduler = runtime.prepare(model, optimizer, scheduler)

    state = TrainerState(
        world_size=world_size,
        distributed_backend="accelerate",
        accumulation_steps=accumulation_steps,
        config_fingerprint="distributed-worker-v1",
    )
    if args.mode == "resume":
        if args.resume is None:
            raise ValueError("--resume is required in resume mode")
        state = load_distributed_checkpoint(
            runtime,
            args.resume,
            expected_fingerprints={"config": "distributed-worker-v1"},
        )
    else:
        random.seed(SEED + rank)
        np.random.seed(SEED + rank)
        torch.manual_seed(SEED + rank)

    plan = DistributedEpochPlan.build(
        source_size=SOURCE_SIZE,
        seed=SEED,
        epoch=0,
        world_size=world_size,
        accumulation_steps=accumulation_steps,
    )
    if state.sampler_state:
        restored = DistributedEpochPlan.from_state_dict(
            state.sampler_state,
            expected_world_size=world_size,
            expected_accumulation_steps=accumulation_steps,
        )
        if restored != plan:
            raise RuntimeError("restored sampler permutation differs from deterministic plan")

    cursor = state.record_cursor
    optimizer_step = state.optimizer_step
    micro_step = state.micro_step
    consumed: list[dict[str, int]] = []
    indices = plan.rank_indices(rank)
    padding = plan.rank_padding_mask(rank)
    target_step = 1 if args.mode == "interrupt" else TOTAL_OPTIMIZER_STEPS
    parameters = list(model.parameters())

    for local_cursor in range(cursor, len(indices)):
        sample_id = indices[local_cursor]
        with runtime.accumulate(model):
            inputs, target = _example(sample_id)
            prediction = model(inputs)
            loss = torch.nn.functional.mse_loss(prediction, target)
            group_begin = (local_cursor // accumulation_steps) * GLOBAL_BATCH
            real_count = sum(
                not value for value in plan.padding_mask[group_begin : group_begin + GLOBAL_BATCH]
            )
            loss = loss * (0.0 if padding[local_cursor] else GLOBAL_BATCH / real_count)
            runtime.backward(loss)
            micro_step += 1
            result = runtime.finish_micro_step(
                optimizer,
                scheduler,
                parameters,
                max_grad_norm=10.0,
            )
        if not padding[local_cursor]:
            consumed.append(
                {"position": local_cursor * world_size + rank, "sample_id": sample_id}
            )
        if result.did_update:
            optimizer_step += 1
        cursor = local_cursor + 1
        if optimizer_step >= target_step:
            break

    sampler_state = plan.state_dict(cursor=cursor)
    final_state = TrainerState(
        epoch=0 if cursor < len(indices) else 1,
        record_cursor=cursor if cursor < len(indices) else 0,
        micro_step=micro_step,
        optimizer_step=optimizer_step,
        curriculum_step=11 + optimizer_step,
        sampler_state=sampler_state,
        config_fingerprint="distributed-worker-v1",
        world_size=world_size,
        distributed_backend="accelerate",
        accumulation_steps=accumulation_steps,
    )
    checkpoint = args.output / "checkpoints" / f"step-{optimizer_step:08d}"
    save_distributed_checkpoint(
        runtime,
        {"model": runtime.unwrap_model(model)},
        checkpoint,
        state=final_state,
    )

    report = {
        "rank": rank,
        "world_size": world_size,
        "optimizer_step": optimizer_step,
        "curriculum_step": final_state.curriculum_step,
        "record_cursor": final_state.record_cursor,
        "sampler_cursor": sampler_state["cursor"],
        "learning_rate": float(scheduler.get_last_lr()[0]),
        "consumed": consumed,
        "optimizer": _optimizer_snapshot(optimizer),
        "next_rng": {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch": float(torch.rand(())),
        },
        "checkpoint": str(checkpoint.resolve()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime.wait_for_everyone()


if __name__ == "__main__":
    main()
