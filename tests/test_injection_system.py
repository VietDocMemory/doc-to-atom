import torch
import pytest
from conftest import ToyLM
from torch.nn import functional as F

from doc_to_atom.atomization.models import Atom
from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector, discover_target_shapes
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.training.system import TeacherStudentSystem


def test_dynamic_lora_changes_forward_only_when_active():
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    injector = DynamicLoRAInjector(model, targets, rank=2, alpha=4)
    ids = torch.randint(0, 31, (1, 4))
    baseline = model(ids).logits
    from doc_to_atom.composer.model import ComposedMemory

    memory = ComposedMemory(
        factors_a={target.key: torch.randn(2, target.input_dim) for target in targets},
        factors_b={target.key: torch.randn(target.output_dim, 2) for target in targets},
        gates={target.key: torch.tensor(1.0) for target in targets},
        micro_kv=None,
    )
    with injector.activate(memory):
        adapted = model(ids).logits
    assert not torch.allclose(baseline, adapted)
    assert torch.allclose(baseline, model(ids).logits)


def test_dynamic_lora_applies_gate_and_alpha_over_rank_exactly():
    model = ToyLM(depth=6)
    target = discover_target_shapes(model, [5], ["q_proj"])[0]
    injector = DynamicLoRAInjector(model, [target], rank=2, alpha=4)
    module = model.model.layers[5].self_attn.q_proj
    x = torch.randn(1, 3, target.input_dim)
    a = torch.randn(2, target.input_dim)
    b = torch.randn(target.output_dim, 2)
    gate = torch.tensor(0.25)
    from doc_to_atom.composer.model import ComposedMemory

    memory = ComposedMemory(
        factors_a={target.key: a},
        factors_b={target.key: b},
        gates={target.key: gate},
        micro_kv=None,
    )
    baseline = module(x)
    with injector.activate(memory):
        adapted = module(x)
    expected = baseline + F.linear(F.linear(x, a), b) * gate * (4 / 2)
    assert torch.allclose(adapted, expected)


def test_teacher_student_forward():
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    encoder = SharedAtomQueryEncoder(model, hidden_size=8, output_dim=16, num_layers=4)
    compiler = AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2)
    router = TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32)
    composer = MemoryComposer()
    injector = DynamicLoRAInjector(model, targets, rank=2, alpha=4)
    system = TeacherStudentSystem(model, encoder, compiler, router, composer, injector)
    atoms = [
        Atom(
            atom_id=f"a{i}",
            semantic_type="fact_claim",
            content=str(i),
            source_span=str(i),
            retrieval_text=str(i),
            is_answer_bearing=True,
            abstraction_level="evidence",
            confidence=1,
        )
        for i in range(3)
    ]
    atom_inputs = {
        "input_ids": torch.randint(0, 31, (3, 4)),
        "attention_mask": torch.ones(3, 4, dtype=torch.long),
    }
    bank = system.build_memory(atom_inputs, atoms)
    query = {
        "input_ids": torch.randint(0, 31, (1, 4)),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    teacher = system.teacher_forward(query)
    student = system.student_forward(query, query, "q", bank, top_k=2)
    assert teacher.shape == student.logits.shape
    assert student.router.selected_indices.numel() == 2


def test_teacher_student_dispatches_every_trainable_operation():
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    system = TeacherStudentSystem(
        model,
        SharedAtomQueryEncoder(model, hidden_size=8, output_dim=16, num_layers=4),
        AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2),
        TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32),
        MemoryComposer(),
        DynamicLoRAInjector(model, targets, rank=2, alpha=4),
    )
    atoms = [
        Atom(
            atom_id=f"a{i}",
            semantic_type="fact_claim",
            content=str(i),
            source_span=str(i),
            retrieval_text=str(i),
            is_answer_bearing=True,
            abstraction_level="evidence",
            confidence=1,
        )
        for i in range(3)
    ]
    atom_inputs = {
        "input_ids": torch.randint(0, 31, (3, 4)),
        "attention_mask": torch.ones(3, 4, dtype=torch.long),
    }
    query = {
        "input_ids": torch.randint(0, 31, (1, 4)),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    bank = system(operation="build_memory", atom_inputs=atom_inputs, atoms=atoms)
    teacher = system(operation="teacher", full_context_query_inputs=query)
    student = system(
        operation="student",
        query_inputs=query,
        query_embedding_inputs=query,
        query_text="q",
        bank=bank,
        top_k=2,
    )
    forced = system(
        operation="forced_gold",
        query_inputs=query,
        bank=bank,
        selected_indices=torch.tensor([0, 1]),
        routing_weights=torch.tensor([0.6, 0.4]),
    )

    assert teacher.shape == student.logits.shape == forced.shape
    assert student.router.selected_indices.numel() == 2
    with pytest.raises(ValueError, match="unknown TeacherStudentSystem operation"):
        system(operation="not-real")
