"""Numerical and lifecycle regressions identified by the D2L source comparison."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from conftest import ToyLM
from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.composer.model import ComposedMemory, MemoryComposer
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector, discover_target_shapes
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.training.system import TeacherStudentSystem


@pytest.mark.parametrize('autocast', [False, True])
def test_bf16_base_preserves_fp32_adapter_delta_and_gradients(autocast):
    model = ToyLM(hidden=2, depth=1).to(torch.bfloat16)
    model.requires_grad_(False)
    target = discover_target_shapes(model, [0], ['q_proj'])[0]
    module = model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        module.weight.zero_()
    injector = DynamicLoRAInjector(model, [target], rank=1, alpha=2)
    a = torch.tensor([[1.001, -1.0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    gate = torch.tensor(0.25, requires_grad=True)
    memory = ComposedMemory({target.key:a}, {target.key:b}, {target.key:gate}, None)
    x = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
            with injector.activate(memory):
                result = module(x)
        expected = (F.linear(F.linear(x.float(), a), b) * gate * 2).to(torch.bfloat16)
        assert result.dtype == torch.bfloat16
        assert torch.equal(result, expected)
        assert result.abs().sum() > 0  # Downcasting A first would cancel to zero.
        result.float().sum().backward()
        for value in [a,b,gate]:
            assert value.grad is not None and torch.isfinite(value.grad).all()
            assert value.grad.abs().sum() > 0
    finally:
        injector.close()


def test_injector_context_is_restored_after_exception_and_close():
    model=ToyLM(hidden=2,depth=1)
    target=discover_target_shapes(model,[0],['q_proj'])[0]
    module=model.model.layers[0].self_attn.q_proj
    injector=DynamicLoRAInjector(model,[target],rank=1,alpha=1)
    x=torch.ones(1,1,2)
    baseline=module(x).detach()
    memory=ComposedMemory({target.key:torch.ones(1,2)},
                          {target.key:torch.ones(2,1)}, {target.key:torch.tensor(1.)},None)
    with pytest.raises(RuntimeError,match='stop'):
        with injector.activate(memory):
            assert not torch.equal(module(x),baseline)
            raise RuntimeError('stop')
    assert torch.equal(module(x),baseline)
    injector.close()
    with injector.activate(memory):
        assert torch.equal(module(x),baseline)


def test_pooling_masks_nonfinite_padding_before_fp32_reduction(monkeypatch):
    encoder=SharedAtomQueryEncoder(ToyLM(hidden=2,depth=1),hidden_size=2,output_dim=2,num_layers=1)
    encoder.projection=nn.Identity()
    hidden=torch.tensor([[[1.,2.],[3.,4.],[float('nan'),float('inf')]]],dtype=torch.bfloat16)
    monkeypatch.setattr(encoder,'_early_hidden',lambda **kw:hidden)
    result=encoder(torch.tensor([[1,2,0]]),torch.tensor([[1,1,0]]))
    assert result.dtype==torch.float32
    assert torch.equal(result,torch.tensor([[2.,3.]]))


def test_encoder_train_keeps_base_eval_and_projection_trainable():
    model=ToyLM(depth=2)
    model.dropout=nn.Dropout(0.9)
    encoder=SharedAtomQueryEncoder(model,hidden_size=8,output_dim=4,num_layers=1)
    assert encoder.train() is encoder
    assert encoder.training and encoder.projection.training
    assert not model.training and not model.dropout.training
    ids=torch.tensor([[1,2,3]])
    encoder(ids,torch.ones_like(ids)).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.projection.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_encoder_backbone_runs_without_grad_but_projection_keeps_grad():
    class GradProbe(ToyLM):
        def forward(self,*args,**kwargs):
            self.observed_grad_mode=torch.is_grad_enabled()
            return super().forward(*args,**kwargs)
    model=GradProbe(depth=2)
    encoder=SharedAtomQueryEncoder(model,8,4,num_layers=1)
    ids=torch.tensor([[1,2]])
    result=encoder(ids,torch.ones_like(ids))
    assert model.observed_grad_mode is False
    assert result.requires_grad


def test_system_train_keeps_frozen_base_eval_without_freezing_memory_modules():
    model=ToyLM(depth=2)
    model.dropout=nn.Dropout(0.9)
    targets=discover_target_shapes(model,[1],['q_proj'])
    system=TeacherStudentSystem(
        model,SharedAtomQueryEncoder(model,8,4,num_layers=1),
        AtomMemoryCompiler(targets,input_dim=4,latent_dim=4,key_dim=4,rank=1),
        TwoStageRouter(embedding_dim=4,key_dim=4),MemoryComposer(),
        DynamicLoRAInjector(model,targets,rank=1,alpha=1))
    try:
        assert system.train() is system
        assert system.training and system.compiler.training and system.router.training
        assert system.encoder.projection.training
        assert not model.training and not model.dropout.training
        system.eval()
        assert not system.training and not system.compiler.training
        system.train()
        assert not model.training and system.compiler.training
    finally:
        system.injector.close()


def test_factory_uses_transformers_451_compatible_dtype_keyword(monkeypatch):
    from doc_to_atom.training import factory
    calls=[]
    class ModelLoader:
        @staticmethod
        def from_pretrained(name, *, torch_dtype, **kwargs):
            assert 'dtype' not in kwargs
            calls.append((name,torch_dtype))
            return ToyLM(depth=2).to(torch_dtype)
    class TokenizerLoader:
        @staticmethod
        def from_pretrained(name):
            return object()
    monkeypatch.setattr(factory,'AutoModelForCausalLM',ModelLoader)
    monkeypatch.setattr(factory,'AutoTokenizer',TokenizerLoader)
    config={
        'runtime':{'device':'cpu'},
        'model':{'name':'test/local-only','hidden_size':8,'encoder_layers':1,
                 'memory_layer_indices':[1],'target_modules':['q_proj']},
        'encoder':{'output_dim':4},
        'compiler':{'latent_dim':4,'key_dim':4,'lora_rank':1,'lora_alpha':2,
                    'lora_a_std':0.02,'micro_kv':{'enabled':False}},
        'router':{'stage1_top_k':32,'rerank_fusion_weight':1.0,'temperature':1.0},
    }
    system,tokenizer,targets=factory.build_system(config,enable_reranker=False)
    try:
        assert calls==[('test/local-only',torch.bfloat16)]
        assert len(targets)==1
        assert next(system.base_model.parameters()).dtype==torch.bfloat16
        assert next(system.compiler.parameters()).dtype==torch.float32
    finally:
        system.injector.close()
