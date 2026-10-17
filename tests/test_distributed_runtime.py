from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from doc_to_atom.training.distributed import (
    AccelerateTrainingRuntime,
    DistributedPreflightError,
)


class _FakePreparedOptimizer:
    def __init__(self, optimizer, accelerator):
        self.optimizer = optimizer
        self.accelerator = accelerator

    def step(self):
        self.accelerator.optimizer_step_calls += 1
        if self.accelerator.sync_gradients:
            self.optimizer.step()

    def zero_grad(self, **kwargs):
        if self.accelerator.sync_gradients:
            self.optimizer.zero_grad(**kwargs)


class _FakePreparedScheduler:
    def __init__(self, scheduler, accelerator):
        self.scheduler = scheduler
        self.accelerator = accelerator

    def step(self):
        self.accelerator.scheduler_step_calls += 1
        if self.accelerator.sync_gradients:
            self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


class FakeAccelerator:
    def __init__(self, *, world_size=2, rank=0, accumulation_steps=2):
        self.num_processes = world_size
        self.process_index = rank
        self.is_main_process = rank == 0
        self.gradient_accumulation_steps = accumulation_steps
        self.distributed_type = "MULTI_CPU"
        self.device = torch.device("cpu")
        self.sync_gradients = False
        self.prepare_calls = 0
        self.backward_calls = 0
        self.clip_calls = 0
        self.optimizer_step_calls = 0
        self.scheduler_step_calls = 0
        self._micro_step = 0

    def prepare(self, model, optimizer, scheduler):
        self.prepare_calls += 1
        return (
            model,
            _FakePreparedOptimizer(optimizer, self),
            _FakePreparedScheduler(scheduler, self),
        )

    @contextmanager
    def accumulate(self, model):
        del model
        self._micro_step += 1
        self.sync_gradients = self._micro_step % self.gradient_accumulation_steps == 0
        yield

    def backward(self, loss):
        self.backward_calls += 1
        (loss / self.gradient_accumulation_steps).backward()

    def clip_grad_norm_(self, parameters, max_norm):
        self.clip_calls += 1
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        return None


def _config(*, effective_global_batch=4):
    return {
        "runtime": {"distributed_backend": "accelerate"},
        "training": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "effective_global_batch": effective_global_batch,
        },
    }


def test_accelerate_runtime_prepares_exactly_once_and_exposes_rank_contract() -> None:
    accelerator = FakeAccelerator(rank=1)
    runtime = AccelerateTrainingRuntime(_config(), accelerator=accelerator)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    prepared_model, prepared_optimizer, prepared_scheduler = runtime.prepare(
        model, optimizer, scheduler
    )

    assert prepared_model is model
    assert prepared_optimizer is not optimizer
    assert prepared_scheduler is not scheduler
    assert runtime.world_size == 2
    assert runtime.rank == 1
    assert not runtime.is_main_process
    with pytest.raises(RuntimeError, match="prepare.*once"):
        runtime.prepare(model, optimizer, scheduler)


def test_accelerate_runtime_advances_optimizer_scheduler_and_curriculum_once() -> None:
    accelerator = FakeAccelerator()
    runtime = AccelerateTrainingRuntime(_config(), accelerator=accelerator)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    model, optimizer, scheduler = runtime.prepare(model, optimizer, scheduler)
    optimizer_step = 0
    curriculum_step = 0

    for value in (1.0, 2.0):
        with runtime.accumulate(model):
            loss = model(torch.tensor([[value]])).square().mean()
            runtime.backward(loss)
            result = runtime.finish_micro_step(
                optimizer,
                scheduler,
                model.parameters(),
                max_grad_norm=1.0,
            )
        if result.did_update:
            optimizer_step += 1
            curriculum_step += 1

    assert accelerator.backward_calls == 2
    assert accelerator.optimizer_step_calls == 2
    assert accelerator.scheduler_step_calls == 2
    assert accelerator.clip_calls == 1
    assert optimizer_step == curriculum_step == 1
    assert scheduler.get_last_lr() == pytest.approx([0.05])


def test_accelerate_runtime_rejects_effective_batch_mismatch() -> None:
    with pytest.raises(DistributedPreflightError, match="effective_global_batch"):
        AccelerateTrainingRuntime(
            _config(effective_global_batch=128), accelerator=FakeAccelerator()
        )


def test_deepspeed_zero2_backend_requires_deepspeed_distributed_type() -> None:
    config = _config()
    config["runtime"]["distributed_backend"] = "deepspeed_zero2"
    with pytest.raises(DistributedPreflightError, match="DEEPSPEED"):
        AccelerateTrainingRuntime(config, accelerator=FakeAccelerator())
