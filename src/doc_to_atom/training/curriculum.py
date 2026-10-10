from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start_step: int
    end_step: int | None
    top_k: int
    active_losses: frozenset[str]
    increase_irrelevant: bool = False
    strengthen_sparse: bool = False


CORE = frozenset({"lm", "distill", "route", "irrelevant", "protect"})
STAGES = (
    CurriculumStage("single_atom", 0, 2000, 1, CORE),
    CurriculumStage("multi_atom", 2000, 6000, 4, CORE | {"composition", "sparse"}),
    CurriculumStage(
        "robustness",
        6000,
        9000,
        8,
        CORE | {"composition", "sparse", "conflict", "confidence"},
        increase_irrelevant=True,
    ),
    CurriculumStage(
        "full_composition",
        9000,
        None,
        8,
        CORE | {"composition", "sparse", "conflict", "confidence", "rerank"},
        increase_irrelevant=True,
        strengthen_sparse=True,
    ),
)


def stage_for_step(step: int) -> CurriculumStage:
    if step < 0:
        raise ValueError("step must be non-negative")
    for stage in STAGES:
        if step >= stage.start_step and (stage.end_step is None or step < stage.end_step):
            return stage
    raise AssertionError("unreachable")

