from __future__ import annotations

import torch
from torch.nn import functional as F

from doc_to_atom.composer.model import ComposedMemory


def language_modeling_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    return F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=ignore_index,
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    answer_mask: torch.Tensor,
    temperature: float = 1.5,
) -> torch.Tensor:
    mask = answer_mask.bool()
    student = student_logits[mask] / temperature
    teacher = teacher_logits[mask] / temperature
    if student.numel() == 0:
        return student_logits.sum() * 0
    return F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction="batchmean")


def routing_loss(
    scores: torch.Tensor,
    gold_mask: torch.Tensor,
    distractor_mask: torch.Tensor,
    negative_weight: float = 0.5,
    gold_weight: float = 1.0,
    distractor_weight: float = 1.5,
) -> torch.Tensor:
    targets = gold_mask.float()
    weights = torch.full_like(scores, negative_weight)
    weights = torch.where(gold_mask.bool(), gold_weight, weights)
    weights = torch.where(distractor_mask.bool(), distractor_weight, weights)
    return F.binary_cross_entropy_with_logits(scores, targets, weight=weights)


def listwise_rerank_loss(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if not positive_mask.any():
        return scores.sum() * 0
    log_probs = F.log_softmax(scores, dim=-1)
    target = positive_mask.float() / positive_mask.float().sum().clamp_min(1)
    return -(target * log_probs).sum()


def irrelevant_suppression_loss(memory: ComposedMemory) -> torch.Tensor:
    values = []
    for key in memory.factors_a:
        values.append(memory.factors_a[key].pow(2).sum() + memory.factors_b[key].pow(2).sum())
    return torch.stack(values).sum() if values else torch.tensor(0.0)


def protection_loss(memory: ComposedMemory, lora_scale: float = 1.0) -> torch.Tensor:
    """Penalize the actual injected delta: (alpha/r * gate * B @ A)^2."""
    values = []
    for key in memory.factors_a:
        delta = memory.factors_b[key] @ memory.factors_a[key]
        values.append((lora_scale * memory.gates[key] * delta).pow(2).sum())
    return torch.stack(values).sum() if values else torch.tensor(0.0)


def sparse_regularization_loss(
    gates: torch.Tensor, rho: float, valid_atoms: torch.Tensor | None = None
) -> torch.Tensor:
    """Eq. 7: mean per-atom gate for each layer/module, excluding padding."""
    if gates.ndim != 2:
        raise ValueError("expected [atoms, layer_modules] gate values")
    if valid_atoms is None:
        valid_atoms = torch.ones(gates.shape[0], dtype=torch.bool, device=gates.device)
    if valid_atoms.shape != (gates.shape[0],):
        raise ValueError("valid_atoms must be a vector matching the atom dimension")
    if not valid_atoms.any():
        return gates.sum() * 0
    return (gates[valid_atoms].mean(dim=0) - rho).pow(2).sum()


def _kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(left, -1), F.softmax(right, -1), reduction="batchmean")


def composition_consistency_loss(
    main_logits: torch.Tensor, forced_logits: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (_kl(main_logits, forced_logits) + _kl(forced_logits, main_logits))


def conflict_listwise_loss(
    scores: torch.Tensor,
    group_ids: list[str | None],
    gold_mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for group in sorted({group for group in group_ids if group is not None}):
        indices = torch.tensor([value == group for value in group_ids], device=scores.device)
        positives = gold_mask.bool() & indices
        if indices.sum() > 1 and positives.any():
            losses.append(listwise_rerank_loss(scores[indices], positives[indices]))
    return torch.stack(losses).mean() if losses else scores.sum() * 0


def confidence_alignment_loss(
    routing_weights: torch.Tensor, selected_confidences: torch.Tensor
) -> torch.Tensor:
    target = selected_confidences / selected_confidences.sum().clamp_min(1e-8)
    return F.mse_loss(routing_weights, target)
