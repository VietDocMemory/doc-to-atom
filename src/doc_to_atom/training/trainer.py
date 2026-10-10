from __future__ import annotations

import logging
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
        "distill": distillation_loss(student_logits, teacher_logits, answer_mask, distillation_temperature),
        "route": routing_loss(route_scores, gold_mask, distractor_mask),
        "rerank": listwise_rerank_loss(reranker_scores, gold_mask),
        "irrelevant": irrelevant_suppression_loss(composed),
        "protect": protection_loss(composed),
        "sparse": sparse_regularization_loss(gates, rho),
        "composition": composition_consistency_loss(student_logits, forced_logits),
        "conflict": conflict_listwise_loss(route_scores, group_ids, gold_mask),
        "confidence": confidence_alignment_loss(routing_weights, selected_confidences),
    }

