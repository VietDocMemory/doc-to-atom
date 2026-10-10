import torch
from conftest import ToyLM

from doc_to_atom.compiler.model import AtomMemoryCompiler, TargetShape
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import discover_target_shapes, freeze_base_model


def test_shared_encoder_shape_and_frozen_backbone():
    model = ToyLM()
    encoder = SharedAtomQueryEncoder(model, hidden_size=8, output_dim=16, num_layers=4)
    output = encoder(torch.randint(0, 31, (3, 5)), torch.ones(3, 5, dtype=torch.long))
    assert output.shape == (3, 16)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in encoder.projection.parameters())


def test_compiler_shapes_key_norm_and_zero_b():
    targets = [TargetShape(2, "q_proj", 8, 12), TargetShape(3, "down_proj", 20, 8)]
    compiler = AtomMemoryCompiler(targets, input_dim=16, latent_dim=16, key_dim=16, rank=8)
    output = compiler(torch.randn(4, 16))
    assert output.keys.shape == (4, 16)
    assert torch.allclose(output.keys.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert output.factors_a["2:q_proj"].shape == (4, 8, 8)
    assert output.factors_b["2:q_proj"].shape == (4, 12, 8)
    assert torch.count_nonzero(output.factors_b["2:q_proj"]) == 0


def test_targets_are_only_last_four_layers():
    model = ToyLM(depth=6)
    targets = discover_target_shapes(
        model, [2, 3, 4, 5], ["q_proj", "v_proj", "o_proj", "down_proj"]
    )
    assert len(targets) == 16
    assert {target.layer_index for target in targets} == {2, 3, 4, 5}


def test_freeze_base_has_no_gradients():
    model = ToyLM()
    freeze_base_model(model)
    assert not any(parameter.requires_grad for parameter in model.parameters())
