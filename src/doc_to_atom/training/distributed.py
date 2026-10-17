from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch


class DistributedTopologyError(ValueError):
    """Raised when a saved distributed plan cannot be resumed exactly."""


class DistributedPreflightError(RuntimeError):
    """Raised before training when the launched topology violates the config."""


@dataclass(frozen=True)
class OptimizationStepResult:
    did_update: bool
    gradient_norm: torch.Tensor | float | None


class AccelerateTrainingRuntime:
    """Narrow ownership boundary around an initialized Accelerate runtime."""

    def __init__(self, config: dict[str, Any], *, accelerator=None) -> None:
        training = config["training"]
        runtime = config.get("runtime", {})
        self.backend = str(runtime.get("distributed_backend", "accelerate"))
        self.accumulation_steps = int(training["gradient_accumulation_steps"])
        if self.accumulation_steps <= 0:
            raise DistributedPreflightError(
                "gradient_accumulation_steps must be positive"
            )
        if accelerator is None:
            from accelerate import Accelerator

            accelerator = Accelerator(
                gradient_accumulation_steps=self.accumulation_steps
            )
        self.accelerator = accelerator
        self.world_size = int(accelerator.num_processes)
        self.rank = int(accelerator.process_index)
        self.is_main_process = bool(accelerator.is_main_process)
        self.device = accelerator.device
        self._prepared = False
        self._validate_preflight(config)

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def sync_gradients(self) -> bool:
        return bool(self.accelerator.sync_gradients)

    def _validate_preflight(self, config: dict[str, Any]) -> None:
        actual_accumulation = int(self.accelerator.gradient_accumulation_steps)
        if actual_accumulation != self.accumulation_steps:
            raise DistributedPreflightError(
                "Accelerate gradient_accumulation_steps mismatch: "
                f"runtime={actual_accumulation}, config={self.accumulation_steps}"
            )
        training = config["training"]
        expected_batch = (
            int(training["micro_batch_size"])
            * self.accumulation_steps
            * self.world_size
        )
        declared_batch = int(training["effective_global_batch"])
        if declared_batch != expected_batch:
            raise DistributedPreflightError(
                "effective_global_batch mismatch: "
                f"declared={declared_batch}, computed={expected_batch} "
                "(micro_batch_size * gradient_accumulation_steps * world_size)"
            )
        distributed_name = getattr(
            self.accelerator.distributed_type,
            "name",
            str(self.accelerator.distributed_type).split(".")[-1],
        )
        if self.backend == "deepspeed_zero2":
            if distributed_name != "DEEPSPEED":
                raise DistributedPreflightError(
                    "deepspeed_zero2 requires Accelerate distributed type DEEPSPEED, "
                    f"got {distributed_name}"
                )
            plugin = getattr(getattr(self.accelerator, "state", None), "deepspeed_plugin", None)
            ds_config = getattr(plugin, "deepspeed_config", {}) if plugin else {}
            stage = ds_config.get("zero_optimization", {}).get("stage")
            if int(stage or -1) != 2:
                raise DistributedPreflightError(
                    f"deepspeed_zero2 requires ZeRO stage 2, got {stage!r}"
                )
        elif self.backend != "accelerate":
            raise DistributedPreflightError(
                f"unsupported Accelerate backend: {self.backend!r}"
            )

    def prepare(self, model, optimizer, scheduler):
        if self._prepared:
            raise RuntimeError("Accelerate prepare may be called exactly once")
        self._prepared = True
        return self.accelerator.prepare(model, optimizer, scheduler)

    def accumulate(self, model, *, force_sync: bool = False):
        if force_sync:
            raise DistributedPreflightError(
                "distributed partial accumulation is forbidden; pad the epoch plan"
            )
        return self.accelerator.accumulate(model)

    def backward(self, loss: torch.Tensor) -> None:
        self.accelerator.backward(loss)

    def finish_micro_step(
        self,
        optimizer,
        scheduler,
        parameters: Iterable[torch.nn.Parameter],
        *,
        max_grad_norm: float,
        partial_window_scale: float = 1.0,
    ) -> OptimizationStepResult:
        if partial_window_scale != 1.0:
            raise DistributedPreflightError(
                "distributed partial accumulation correction is not supported"
            )
        gradient_norm = None
        if self.sync_gradients:
            gradient_norm = self.accelerator.clip_grad_norm_(parameters, max_grad_norm)
        # AcceleratedOptimizer/AcceleratedScheduler (and DeepSpeed wrappers) own
        # skip/step behavior on non-synchronizing micro-steps.
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return OptimizationStepResult(self.sync_gradients, gradient_norm)

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model)

    def wait_for_everyone(self) -> None:
        self.accelerator.wait_for_everyone()

    def save_state(self, path) -> None:
        self.accelerator.save_state(str(path), safe_serialization=True)

    def load_state(self, path) -> None:
        self.accelerator.load_state(str(path))

    def collective_errors(self, local_error: str | None) -> list[str | None]:
        if self.world_size == 1:
            return [local_error]
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise DistributedPreflightError(
                "multi-rank checkpointing requires an initialized torch.distributed group"
            )
        errors: list[str | None] = [None] * self.world_size
        torch.distributed.all_gather_object(errors, local_error)
        return errors


class LocalTrainingRuntime:
    """Single-process oracle with the same optimizer-step interface."""

    backend = "none"
    world_size = 1
    rank = 0
    is_main_process = True
    is_distributed = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, accumulation_steps: int) -> None:
        if accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.accumulation_steps = accumulation_steps
        self.sync_gradients = False
        self._micro_step = 0
        self._prepared = False

    def prepare(self, model, optimizer, scheduler):
        if self._prepared:
            raise RuntimeError("runtime prepare may be called exactly once")
        self._prepared = True
        return model, optimizer, scheduler

    @contextmanager
    def accumulate(self, model, *, force_sync: bool = False) -> Iterator[None]:
        del model
        self._micro_step += 1
        self.sync_gradients = (
            self._micro_step % self.accumulation_steps == 0 or force_sync
        )
        yield

    def backward(self, loss: torch.Tensor) -> None:
        (loss / self.accumulation_steps).backward()

    def finish_micro_step(
        self,
        optimizer,
        scheduler,
        parameters: Iterable[torch.nn.Parameter],
        *,
        max_grad_norm: float,
        partial_window_scale: float = 1.0,
    ) -> OptimizationStepResult:
        parameters = list(parameters)
        if not self.sync_gradients:
            return OptimizationStepResult(False, None)
        if partial_window_scale != 1.0:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(partial_window_scale)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        self._micro_step = 0
        return OptimizationStepResult(True, gradient_norm)

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def wait_for_everyone() -> None:
        return None


@dataclass(frozen=True)
class DistributedEpochPlan:
    """Deterministic, equally sharded work for one distributed epoch.

    The global sequence is padded to a complete optimizer-update group. Padding
    is explicit so metrics and sample-accounting can exclude repeated records
    while every rank still participates in the same collectives.
    """

    source_size: int
    seed: int
    epoch: int
    world_size: int
    accumulation_steps: int
    global_indices: tuple[int, ...]
    padding_mask: tuple[bool, ...]

    SCHEMA_VERSION = 1

    @classmethod
    def build(
        cls,
        *,
        source_size: int,
        seed: int,
        epoch: int,
        world_size: int,
        accumulation_steps: int,
    ) -> "DistributedEpochPlan":
        if source_size <= 0:
            raise ValueError("source_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        order = list(range(source_size))
        random.Random(seed + epoch).shuffle(order)
        group_size = world_size * accumulation_steps
        padded_size = ((source_size + group_size - 1) // group_size) * group_size
        padding_count = padded_size - source_size
        padded = order + [order[index % source_size] for index in range(padding_count)]
        mask = [False] * source_size + [True] * padding_count
        return cls(
            source_size=source_size,
            seed=seed,
            epoch=epoch,
            world_size=world_size,
            accumulation_steps=accumulation_steps,
            global_indices=tuple(padded),
            padding_mask=tuple(mask),
        )

    def _validate_rank(self, rank: int) -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} outside [0, {self.world_size})")

    def rank_indices(self, rank: int) -> tuple[int, ...]:
        self._validate_rank(rank)
        return self.global_indices[rank :: self.world_size]

    def rank_padding_mask(self, rank: int) -> tuple[bool, ...]:
        self._validate_rank(rank)
        return self.padding_mask[rank :: self.world_size]

    def resume_rank_indices(self, rank: int, cursor: int) -> tuple[int, ...]:
        indices = self.rank_indices(rank)
        if not 0 <= cursor <= len(indices):
            raise ValueError(f"cursor {cursor} outside [0, {len(indices)}]")
        if cursor % self.accumulation_steps:
            raise ValueError("resume cursor must be at an optimizer boundary")
        return indices[cursor:]

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return len(self.rank_indices(0)) // self.accumulation_steps

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "source_size": self.source_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "world_size": self.world_size,
            "accumulation_steps": self.accumulation_steps,
            "global_indices": list(self.global_indices),
            "padding_mask": list(self.padding_mask),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        expected_world_size: int,
        expected_accumulation_steps: int,
    ) -> "DistributedEpochPlan":
        schema_version = int(state.get("schema_version", -1))
        if schema_version != cls.SCHEMA_VERSION:
            raise DistributedTopologyError(
                f"unsupported sampler schema_version {schema_version}"
            )
        saved_world_size = int(state["world_size"])
        saved_accumulation = int(state["accumulation_steps"])
        if saved_world_size != expected_world_size:
            raise DistributedTopologyError(
                f"world_size mismatch: checkpoint={saved_world_size}, runtime={expected_world_size}"
            )
        if saved_accumulation != expected_accumulation_steps:
            raise DistributedTopologyError(
                "accumulation_steps mismatch: "
                f"checkpoint={saved_accumulation}, runtime={expected_accumulation_steps}"
            )

        plan = cls(
            source_size=int(state["source_size"]),
            seed=int(state["seed"]),
            epoch=int(state["epoch"]),
            world_size=saved_world_size,
            accumulation_steps=saved_accumulation,
            global_indices=tuple(int(index) for index in state["global_indices"]),
            padding_mask=tuple(bool(value) for value in state["padding_mask"]),
        )
        plan._validate_contract()
        return plan

    def _validate_contract(self) -> None:
        if self.source_size <= 0:
            raise DistributedTopologyError("saved source_size must be positive")
        if len(self.global_indices) != len(self.padding_mask):
            raise DistributedTopologyError("saved padding mask length does not match indices")
        group_size = self.world_size * self.accumulation_steps
        if len(self.global_indices) % group_size:
            raise DistributedTopologyError(
                "saved global indices do not form complete accumulation groups"
            )
        if any(index < 0 or index >= self.source_size for index in self.global_indices):
            raise DistributedTopologyError("saved sampler index is outside source_size")
        real_indices = [
            index
            for index, is_padding in zip(
                self.global_indices, self.padding_mask, strict=True
            )
            if not is_padding
        ]
        if sorted(real_indices) != list(range(self.source_size)):
            raise DistributedTopologyError(
                "saved non-padding indices are not an exact source permutation"
            )
