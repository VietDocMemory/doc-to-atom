from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _EarlyExit(RuntimeError):
    pass


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        current: Any = model
        try:
            for name in path:
                current = getattr(current, name)
            if isinstance(current, (nn.ModuleList, list)):
                return current
        except AttributeError:
            continue
    raise ValueError("Cannot locate decoder layers on base model")


class SharedAtomQueryEncoder(nn.Module):
    """One shared encoder for atoms and queries using an early exit after four frozen layers."""

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        output_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_layers = num_layers
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size, dtype=torch.float32),
            nn.Linear(hidden_size, output_dim, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(output_dim, output_dim, dtype=torch.float32),
        )
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def _early_hidden(self, **model_inputs: torch.Tensor) -> torch.Tensor:
        layers = decoder_layers(self.base_model)
        if self.num_layers < 1 or self.num_layers > len(layers):
            raise ValueError(f"encoder layer count {self.num_layers} outside model depth {len(layers)}")
        captured: list[torch.Tensor] = []

        def stop_after_layer(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captured.append(output[0] if isinstance(output, tuple) else output)
            raise _EarlyExit

        handle = layers[self.num_layers - 1].register_forward_hook(stop_after_layer)
        try:
            try:
                self.base_model(**model_inputs, use_cache=False, return_dict=True)
            except _EarlyExit:
                pass
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("early-exit hook did not observe a decoder output")
        return captured[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **extra_inputs: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._early_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **extra_inputs,
        )
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(pooled.float())

    encode_atoms = forward
    encode_queries = forward

