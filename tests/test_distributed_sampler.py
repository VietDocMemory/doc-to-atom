from __future__ import annotations

from collections import Counter

import pytest

from doc_to_atom.training.distributed import (
    DistributedEpochPlan,
    DistributedTopologyError,
)


def test_epoch_plan_is_deterministic_and_epoch_specific() -> None:
    first = DistributedEpochPlan.build(
        source_size=17, seed=91, epoch=3, world_size=2, accumulation_steps=2
    )
    repeated = DistributedEpochPlan.build(
        source_size=17, seed=91, epoch=3, world_size=2, accumulation_steps=2
    )
    next_epoch = DistributedEpochPlan.build(
        source_size=17, seed=91, epoch=4, world_size=2, accumulation_steps=2
    )

    assert first == repeated
    assert first.global_indices != next_epoch.global_indices


def test_epoch_plan_pads_complete_accumulation_windows_and_tracks_padding() -> None:
    plan = DistributedEpochPlan.build(
        source_size=5, seed=13, epoch=0, world_size=2, accumulation_steps=2
    )

    assert len(plan.global_indices) == 8
    assert len(plan.padding_mask) == 8
    assert sum(plan.padding_mask) == 3
    assert plan.optimizer_steps_per_epoch == 2

    rank_lengths = []
    observed_real_indices: list[int] = []
    for rank in range(2):
        indices = plan.rank_indices(rank)
        mask = plan.rank_padding_mask(rank)
        rank_lengths.append(len(indices))
        assert len(indices) % plan.accumulation_steps == 0
        observed_real_indices.extend(
            index for index, is_padding in zip(indices, mask, strict=True) if not is_padding
        )

    assert rank_lengths == [4, 4]
    assert Counter(observed_real_indices) == Counter(range(5))


def test_rank_cursor_resumes_at_next_unconsumed_local_sample() -> None:
    plan = DistributedEpochPlan.build(
        source_size=9, seed=7, epoch=2, world_size=2, accumulation_steps=2
    )

    for rank in range(2):
        assert plan.resume_rank_indices(rank, cursor=2) == plan.rank_indices(rank)[2:]

    with pytest.raises(ValueError, match="optimizer boundary"):
        plan.resume_rank_indices(rank=0, cursor=1)
    with pytest.raises(ValueError, match="outside"):
        plan.resume_rank_indices(rank=0, cursor=100)


def test_serialized_sampler_contract_carries_optimizer_boundary_cursor() -> None:
    plan = DistributedEpochPlan.build(
        source_size=9, seed=7, epoch=2, world_size=2, accumulation_steps=2
    )

    state = plan.state_dict(cursor=2)

    assert state["cursor"] == 2
    assert DistributedEpochPlan.from_state_dict(
        state, expected_world_size=2, expected_accumulation_steps=2
    ) == plan
    with pytest.raises(ValueError, match="optimizer boundary"):
        plan.state_dict(cursor=1)


def test_serialized_plan_rejects_changed_distributed_topology() -> None:
    plan = DistributedEpochPlan.build(
        source_size=8, seed=5, epoch=1, world_size=2, accumulation_steps=4
    )
    state = plan.state_dict()

    restored = DistributedEpochPlan.from_state_dict(
        state, expected_world_size=2, expected_accumulation_steps=4
    )
    assert restored == plan

    with pytest.raises(DistributedTopologyError, match="world_size"):
        DistributedEpochPlan.from_state_dict(
            state, expected_world_size=1, expected_accumulation_steps=4
        )
    with pytest.raises(DistributedTopologyError, match="accumulation_steps"):
        DistributedEpochPlan.from_state_dict(
            state, expected_world_size=2, expected_accumulation_steps=2
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_size": 0}, "source_size"),
        ({"world_size": 0}, "world_size"),
        ({"accumulation_steps": 0}, "accumulation_steps"),
    ],
)
def test_epoch_plan_rejects_invalid_contract(kwargs: dict[str, int], message: str) -> None:
    arguments = {
        "source_size": 3,
        "seed": 1,
        "epoch": 0,
        "world_size": 1,
        "accumulation_steps": 1,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        DistributedEpochPlan.build(**arguments)


def test_rank_bounds_are_fail_fast() -> None:
    plan = DistributedEpochPlan.build(
        source_size=4, seed=1, epoch=0, world_size=2, accumulation_steps=1
    )
    with pytest.raises(ValueError, match="rank"):
        plan.rank_indices(2)
