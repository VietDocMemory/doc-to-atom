from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from doc_to_atom.atomization.models import Atom
from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.compiler.model import AtomMemoryCompiler, TargetShape
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.config import (
    ROOT,
    load_config,
    require_disclosed_training_values,
    write_run_manifest,
)
from doc_to_atom.losses.objectives import (
    composition_consistency_loss,
    confidence_alignment_loss,
    conflict_listwise_loss,
    irrelevant_suppression_loss,
    listwise_rerank_loss,
    protection_loss,
    routing_loss,
    sparse_regularization_loss,
)
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.training.checkpoint import load_checkpoint, save_checkpoint
from doc_to_atom.training.trainer import MultiObjectiveTrainer


def _atoms() -> list[Atom]:
    types = ["fact_claim", "entity_attribute", "event_relation", "process_step", "evidence_fragment"]
    return [
        Atom(
            atom_id=f"atom_{i}",
            semantic_type=types[i % len(types)],
            content=f"Synthetic fact {i}",
            source_span=f"fact {i}",
            retrieval_text=f"synthetic retrieval {i}",
            is_answer_bearing=i % 2 == 0,
            abstraction_level="evidence",
            conflict_group="g1" if i in (0, 1) else None,
            confidence=0.9 - i * 0.05,
        )
        for i in range(6)
    ]


def run_smoke(config: dict, output_dir: Path) -> dict:
    torch.manual_seed(7)
    require_disclosed_training_values(config)
    targets = [
        TargetShape(2, "q_proj", 16, 12),
        TargetShape(2, "v_proj", 16, 8),
        TargetShape(3, "o_proj", 12, 16),
        TargetShape(3, "down_proj", 24, 16),
    ]
    compiler = AtomMemoryCompiler(targets, input_dim=256, latent_dim=256, rank=8)
    router = TwoStageRouter(stage1_top_k=32)
    composer = MemoryComposer()
    optimizer = torch.optim.AdamW(
        list(compiler.parameters()) + list(router.parameters()),
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        weight_decay=config["training"]["weight_decay"],
    )
    atoms = _atoms()
    compiled = compiler(torch.randn(len(atoms), 256))
    bank = MemoryBank(**compiled.__dict__, atoms=atoms)
    routed = router(torch.randn(1, 256), bank.keys, atoms, "synthetic query", final_top_k=4)
    composed = composer(bank, routed.selected_indices, routed.routing_weights)
    selected = routed.selected_indices
    gold = selected == 0
    if not gold.any():
        gold[0] = True
    distractor = selected == 1
    group_ids = [atoms[index].conflict_group for index in selected.tolist()]
    confidences = torch.tensor([atoms[index].confidence for index in selected.tolist()])
    gates = torch.stack(list(composed.gates.values()))
    random_main = torch.randn(2, 5, 13, requires_grad=True)
    random_forced = torch.randn(2, 5, 13, requires_grad=True)
    losses = {
        "lm": random_main.square().mean(),
        "distill": (random_main - random_forced.detach()).square().mean(),
        "route": routing_loss(routed.final_scores, gold, distractor),
        "rerank": listwise_rerank_loss(routed.final_scores, gold),
        "irrelevant": irrelevant_suppression_loss(composed),
        "protect": protection_loss(composed),
        "sparse": sparse_regularization_loss(gates, config["training"]["sparse_target_density"]),
        "composition": composition_consistency_loss(random_main, random_forced),
        "conflict": conflict_listwise_loss(routed.final_scores, group_ids, gold),
        "confidence": confidence_alignment_loss(routed.routing_weights, confidences),
    }
    assert all(torch.isfinite(value) for value in losses.values())
    bundle = MultiObjectiveTrainer(config).combine(9000, losses)
    optimizer.zero_grad(set_to_none=True)
    bundle.total.backward()
    optimizer.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "smoke_checkpoint.pt"
    save_checkpoint({"compiler": compiler, "router": router}, checkpoint, optimizer, step=1)
    step = load_checkpoint({"compiler": compiler, "router": router}, checkpoint, optimizer)
    result = {
        "step": step,
        "total_loss": float(bundle.total.detach()),
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "routing_weight_sum": float(routed.routing_weights.detach().sum()),
        "selected_atom_ids": [atoms[index].atom_id for index in selected.tolist()],
    }
    (output_dir / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/resource_constrained.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/smoke"))
    args = parser.parse_args()
    config = load_config(args.config)
    write_run_manifest(config, Path(args.output_dir) / "run_manifest.json")
    print(json.dumps(run_smoke(config, Path(args.output_dir)), indent=2))

