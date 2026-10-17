from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


class DistributedTopologyError(ValueError):
    """Raised when a saved distributed plan cannot be resumed exactly."""


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
