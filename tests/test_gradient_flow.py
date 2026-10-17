from pathlib import Path

import torch
from conftest import ToyLM

from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.config import load_config
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector, discover_target_shapes
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.training.loop import run_training
from doc_to_atom.training.system import TeacherStudentSystem
from test_training_loop import ToyTokenizer, _record


def test_training_updates_heads_but_not_frozen_base(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    encoder = SharedAtomQueryEncoder(model, 8, 16, 4)
    compiler = AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=2)
    router = TwoStageRouter(embedding_dim=16, key_dim=16, stage1_top_k=32)
    system = TeacherStudentSystem(
        model,
        encoder,
        compiler,
        router,
        MemoryComposer(),
        DynamicLoRAInjector(model, targets, rank=2, alpha=4),
    )
    frozen_before = [parameter.detach().clone() for parameter in model.parameters()]
    projection_before = [parameter.detach().clone() for parameter in encoder.projection.parameters()]
    compiler_before = [parameter.detach().clone() for parameter in compiler.parameters()]
    router_before = [parameter.detach().clone() for parameter in router.parameters()]
    config = load_config(Path(__file__).parents[1] / "configs/resource_constrained.yaml")
    config["training"].update(
        {"max_epochs": 1, "max_steps": 1, "gradient_accumulation_steps": 1, "warmup_steps": 0}
    )

    run_training(system, ToyTokenizer(), [_record("train")], config, tmp_path)

    assert all(
        torch.equal(before, after)
        for before, after in zip(frozen_before, model.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(projection_before, encoder.projection.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(compiler_before, compiler.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(router_before, router.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in model.parameters())
