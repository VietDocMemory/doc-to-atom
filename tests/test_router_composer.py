import torch

from doc_to_atom.atomization.models import Atom
from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.losses.objectives import irrelevant_suppression_loss
from doc_to_atom.router.model import TwoStageRouter


def atoms(n=6):
    return [
        Atom(
            atom_id=f"a{i}",
            semantic_type="fact_claim",
            content=str(i),
            source_span=str(i),
            retrieval_text=str(i),
            is_answer_bearing=True,
            abstraction_level="evidence",
            confidence=0.8,
        )
        for i in range(n)
    ]


def bank(n=6):
    keys = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    return MemoryBank(
        keys=keys,
        factors_a={"0:q_proj": torch.randn(n, 2, 3)},
        factors_b={"0:q_proj": torch.randn(n, 5, 2)},
        mask_logits=torch.zeros(n, 1),
        atoms=atoms(n),
    )


def test_router_topk_and_weights_include_null_candidate():
    memory = bank()
    router = TwoStageRouter(embedding_dim=4, key_dim=4, stage1_top_k=4)
    output = router(torch.randn(1, 4), memory.keys, memory.atoms, "q", final_top_k=3)
    assert output.stage1_indices.numel() == 4
    assert output.selected_indices.numel() == 3
    assert torch.allclose(output.routing_weights.sum() + output.no_memory_weight, torch.tensor(1.0))


def test_irrelevant_query_can_abstain_without_damaging_answerable_route():
    memory = bank(2)
    router = TwoStageRouter(embedding_dim=4, key_dim=4, stage1_top_k=2)
    with torch.no_grad():
        router.no_memory_head.weight.copy_(torch.tensor([[20.0, 0, 0, 0]]))
        router.no_memory_head.bias.zero_()
    off = router(torch.tensor([[1.0, 0, 0, 0]]), memory.keys, memory.atoms, "off", 1)
    on = router(torch.tensor([[-1.0, 0, 0, 0]]), memory.keys, memory.atoms, "on", 1)
    off_memory = MemoryComposer()(memory, off.selected_indices, off.routing_weights)
    on_memory = MemoryComposer()(memory, on.selected_indices, on.routing_weights)
    assert off.no_memory_weight > 0.999
    assert off_memory.adapter_norm() < on_memory.adapter_norm() * 1e-5
    assert on.routing_weights.sum() > 0.999


def test_composer_equation_and_gate():
    memory = bank(3)
    indices = torch.tensor([0, 2])
    weights = torch.tensor([0.25, 0.75])
    composed = MemoryComposer()(memory, indices, weights)
    expected_a = 0.25 * memory.factors_a["0:q_proj"][0] + 0.75 * memory.factors_a["0:q_proj"][2]
    expected_b = 0.25 * memory.factors_b["0:q_proj"][0] + 0.75 * memory.factors_b["0:q_proj"][2]
    assert torch.allclose(composed.factors_a["0:q_proj"], expected_a)
    assert torch.allclose(composed.factors_b["0:q_proj"], expected_b)
    assert torch.allclose(composed.gates["0:q_proj"], torch.tensor(0.5))


def test_irrelevant_objective_suppresses_composed_norm():
    memory = bank(3)
    composed = MemoryComposer()(memory, torch.tensor([0, 1]), torch.tensor([0.5, 0.5]))
    loss = irrelevant_suppression_loss(composed)
    assert torch.isfinite(loss) and loss > 0
