from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F

from doc_to_atom.compiler.model import TargetShape
from doc_to_atom.composer.model import ComposedMemory
from doc_to_atom.encoder.shared import decoder_layers


def freeze_base_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _resolve_target(layer: nn.Module, module_name: str) -> nn.Module:
    if module_name in {"q_proj", "v_proj", "o_proj"}:
        return getattr(layer.self_attn, module_name)
    if module_name == "down_proj":
        return layer.mlp.down_proj
    raise ValueError(f"unsupported paper target module {module_name}")


def discover_target_shapes(
    model: nn.Module, layer_indices: list[int], module_names: list[str]
) -> list[TargetShape]:
    layers = decoder_layers(model)
    targets = []
    for layer_index in layer_indices:
        for module_name in module_names:
            module = _resolve_target(layers[layer_index], module_name)
            if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
                raise TypeError(f"{layer_index}:{module_name} is not linear-like")
            targets.append(
                TargetShape(
                    layer_index=layer_index,
                    module_name=module_name,
                    input_dim=module.in_features,
                    output_dim=module.out_features,
                )
            )
    return targets


class DynamicLoRAInjector:
    """Forward-hook injection of one query-specific composed LoRA into paper targets only."""

    def __init__(
        self,
        model: nn.Module,
        targets: list[TargetShape],
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        self.model = model
        self.targets = targets
        self.scale = alpha / rank
        self._active: ContextVar[ComposedMemory | None] = ContextVar(
            "doc2atom_composed_memory", default=None
        )
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        layers = decoder_layers(model)
        for target in targets:
            module = _resolve_target(layers[target.layer_index], target.module_name)
            self._handles.append(module.register_forward_hook(self._hook(target.key)))

    def _hook(self, key: str):
        def add_delta(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            memory = self._active.get()
            if memory is None:
                return output
            x = inputs[0]
            a = memory.factors_a[key].to(device=x.device, dtype=x.dtype)
            b = memory.factors_b[key].to(device=x.device, dtype=x.dtype)
            gate = memory.gates[key].to(device=x.device, dtype=x.dtype)
            delta = F.linear(F.linear(x, a), b)
            return output + delta * gate * self.scale

        return add_delta

    @contextmanager
    def activate(self, memory: ComposedMemory) -> Iterator[None]:
        token = self._active.set(memory)
        try:
            yield
        finally:
            self._active.reset(token)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

