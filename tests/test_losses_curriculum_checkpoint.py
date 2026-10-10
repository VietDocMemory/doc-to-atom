from pathlib import Path

import torch
from torch import nn

from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.composer.model import MemoryComposer
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
from doc_to_atom.training.checkpoint import load_checkpoint, save_checkpoint
from doc_to_atom.training.curriculum import stage_for_step


def test_every_loss_is_finite():
    logits = torch.randn(2, 4, 7)
    teacher = torch.randn(2, 4, 7)
    labels = torch.randint(0, 7, (2, 4))
    scores = torch.randn(3)
    gold = torch.tensor([True, False, False])
    distractor = torch.tensor([False, True, False])
    bank = MemoryBank(keys=torch.randn(3, 2), factors_a={"k": torch.randn(3, 2, 3)}, factors_b={"k": torch.randn(3, 4, 2)}, mask_logits=torch.randn(3, 1), atoms=[])
    composed = MemoryComposer()(bank, torch.tensor([0, 1]), torch.tensor([0.4, 0.6]))
    losses = [
        language_modeling_loss(logits, labels),
        distillation_loss(logits, teacher, torch.ones(2, 4, dtype=torch.bool)),
        routing_loss(scores, gold, distractor),
        listwise_rerank_loss(scores, gold),
        irrelevant_suppression_loss(composed),
        protection_loss(composed),
        sparse_regularization_loss(torch.tensor([0.2, 0.5]), 0.25),
        composition_consistency_loss(logits, teacher),
        conflict_listwise_loss(scores, ["g", "g", None], gold),
        confidence_alignment_loss(torch.tensor([0.4, 0.6]), torch.tensor([0.8, 0.5])),
    ]
    assert all(torch.isfinite(loss) for loss in losses)


def test_curriculum_switches_exact_boundaries():
    assert (stage_for_step(0).top_k, stage_for_step(1999).top_k) == (1, 1)
    assert stage_for_step(2000).top_k == 4
    assert stage_for_step(6000).top_k == 8
    assert stage_for_step(9000).strengthen_sparse


def test_checkpoint_save_load(tmp_path: Path):
    module = nn.Linear(3, 2)
    original = module.weight.detach().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint({"module": module}, path, step=17)
    with torch.no_grad():
        module.weight.zero_()
    assert load_checkpoint({"module": module}, path) == 17
    assert torch.allclose(module.weight, original)

