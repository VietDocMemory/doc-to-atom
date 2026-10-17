from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from doc_to_atom.losses.objectives import (
    composition_consistency_loss,
    confidence_alignment_loss,
    conflict_listwise_loss,
    distillation_loss,
    irrelevant_suppression_loss,
    language_modeling_loss,
    listwise_rerank_loss,
    protection_loss,
    routing_loss,
    sparse_regularization_loss,
)
from doc_to_atom.training.curriculum import stage_for_step

LOGGER = logging.getLogger(__name__)


@dataclass
class LossBundle:
    values: dict[str, torch.Tensor]
    total: torch.Tensor


@dataclass(frozen=True)
class EarlyStoppingObservation:
    improved: bool
    should_stop: bool
    nonfinite: bool = False


@dataclass
class EarlyStopping:
    mode: str
    patience: int
    min_delta: float = 0.0
    warmup_observations: int = 0
    observations: int = 0
    best: float | None = None
    bad_validations: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("early stopping mode must be 'min' or 'max'")
        if self.patience < 1:
            raise ValueError("early stopping patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("early stopping min_delta must be non-negative")
        if self.warmup_observations < 0:
            raise ValueError("early stopping warmup_observations must be non-negative")

    def observe(self, value: float) -> EarlyStoppingObservation:
        self.observations += 1
        nonfinite = not math.isfinite(value)
        if self.best is None:
            improved = not nonfinite
        elif self.mode == "min":
            improved = not nonfinite and value < self.best - self.min_delta
        else:
            improved = not nonfinite and value > self.best + self.min_delta
        if improved:
            self.best = value
            self.bad_validations = 0
        elif self.observations <= self.warmup_observations:
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        self.stopped = self.bad_validations >= self.patience
        return EarlyStoppingObservation(improved, self.stopped, nonfinite)


class MultiObjectiveTrainer:
    def __init__(self, config: dict):
        self.config = config

    def combine(self, step: int, values: dict[str, torch.Tensor]) -> LossBundle:
        stage = stage_for_step(step)
        configured = self.config["training"]["loss_weights"]
        total = next(iter(values.values())).sum() * 0
        for name in stage.active_losses:
            if name not in values:
                continue
            weight = configured.get(name)
            if weight is None:
                raise ValueError(f"PAPER_UNSPECIFIED loss coefficient: {name}")
            if name == "irrelevant" and stage.increase_irrelevant:
                multiplier = self.config["training"].get("irrelevant_stage3_multiplier")
                if multiplier is None:
                    raise ValueError("PAPER_UNSPECIFIED Stage-3 irrelevant-loss multiplier")
                weight *= multiplier
            if name == "sparse" and stage.strengthen_sparse:
                multiplier = self.config["training"].get("sparse_stage4_multiplier")
                if multiplier is None:
                    raise ValueError("PAPER_UNSPECIFIED Stage-4 sparse-loss multiplier")
                weight *= multiplier
            total = total + weight * values[name]
        return LossBundle(values=values, total=total)


def all_objectives(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    forced_logits: torch.Tensor,
    labels: torch.Tensor,
    answer_mask: torch.Tensor,
    route_scores: torch.Tensor,
    gold_mask: torch.Tensor,
    distractor_mask: torch.Tensor,
    reranker_scores: torch.Tensor,
    composed,
    gates: torch.Tensor,
    rho: float,
    group_ids: list[str | None],
    routing_weights: torch.Tensor,
    selected_confidences: torch.Tensor,
    distillation_temperature: float = 1.5,
) -> dict[str, torch.Tensor]:
    return {
        "lm": language_modeling_loss(student_logits, labels),
        "distill": distillation_loss(
            student_logits, teacher_logits, answer_mask, distillation_temperature
        ),
        "route": routing_loss(route_scores, gold_mask, distractor_mask),
        "rerank": listwise_rerank_loss(reranker_scores, gold_mask),
        "irrelevant": irrelevant_suppression_loss(composed),
        "protect": protection_loss(composed),
        "sparse": sparse_regularization_loss(gates, rho),
        "composition": composition_consistency_loss(student_logits, forced_logits),
        "conflict": conflict_listwise_loss(route_scores, group_ids, gold_mask),
        "confidence": confidence_alignment_loss(routing_weights, selected_confidences),
    }
