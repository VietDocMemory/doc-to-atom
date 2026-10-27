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


def stage_for_step(
    step: int, boundaries: tuple[int, int, int] | list[int] | None = None
) -> CurriculumStage:
    if step < 0:
        raise ValueError("step must be non-negative")
    if boundaries is None:
        stages = STAGES
    else:
        if (len(boundaries) != 3 or any(type(value) is not int for value in boundaries)
                or not 0 < boundaries[0] < boundaries[1] < boundaries[2]):
            raise ValueError("curriculum_boundaries must be three strictly increasing positive steps")
        starts = (0, *boundaries)
        ends = (*boundaries, None)
        stages = tuple(
            CurriculumStage(stage.name, start, end, stage.top_k, stage.active_losses,
                            stage.increase_irrelevant, stage.strengthen_sparse)
            for stage, start, end in zip(STAGES, starts, ends, strict=True)
        )
    for stage in stages:
        if step >= stage.start_step and (stage.end_step is None or step < stage.end_step):
            return stage
    raise AssertionError("unreachable")
