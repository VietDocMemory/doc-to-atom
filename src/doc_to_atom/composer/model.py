from __future__ import annotations

from dataclasses import dataclass

import torch

from doc_to_atom.compiler.memory import MemoryBank, MicroKV


@dataclass
class ComposedMemory:
    factors_a: dict[str, torch.Tensor]
    factors_b: dict[str, torch.Tensor]
    gates: dict[str, torch.Tensor]
    micro_kv: MicroKV | None

    def adapter_norm(self) -> torch.Tensor:
        norms = []
        for key in self.factors_a:
            delta = self.factors_b[key] @ self.factors_a[key]
            norms.append((self.gates[key] * delta).pow(2).sum())
        return torch.stack(norms).sum().sqrt() if norms else torch.tensor(0.0)


class MemoryComposer:
    """Implements Eq. 1 exactly; sparse gate is applied once at injection."""

    def __call__(
        self,
        bank: MemoryBank,
        selected_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> ComposedMemory:
        if selected_indices.numel() != routing_weights.numel():
            raise ValueError("selected indices and routing weights must have equal length")
        factors_a = {
            key: torch.einsum("k,kri->ri", routing_weights, values[selected_indices])
            for key, values in bank.factors_a.items()
        }
        factors_b = {
            key: torch.einsum("k,kdr->dr", routing_weights, values[selected_indices])
            for key, values in bank.factors_b.items()
        }
        selected_mask_logits = bank.mask_logits[selected_indices]
        aggregate_logits = torch.einsum("k,kt->t", routing_weights, selected_mask_logits)
        gate_values = torch.sigmoid(aggregate_logits)
        gates = {key: gate_values[index] for index, key in enumerate(bank.factors_a)}

        micro_kv = None
        if bank.micro_kv is not None:
            weights = routing_weights.view(-1, 1, 1, 1)
            weighted_keys = bank.micro_kv.keys[selected_indices] * weights  # K,L,S,H
            weighted_values = bank.micro_kv.values[selected_indices] * weights
            k, layers, slots, head_dim = weighted_keys.shape
            keys = weighted_keys.permute(1, 0, 2, 3).reshape(layers, k * slots, head_dim)
            values = weighted_values.permute(1, 0, 2, 3).reshape(layers, k * slots, head_dim)
            micro_kv = MicroKV(keys=keys, values=values)
        return ComposedMemory(factors_a, factors_b, gates, micro_kv)
