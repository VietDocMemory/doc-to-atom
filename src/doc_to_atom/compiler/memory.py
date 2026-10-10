from __future__ import annotations

from dataclasses import dataclass

import torch

from doc_to_atom.atomization.models import Atom


@dataclass
class MicroKV:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class CompiledAtoms:
    keys: torch.Tensor
    factors_a: dict[str, torch.Tensor]
    factors_b: dict[str, torch.Tensor]
    mask_logits: torch.Tensor
    micro_kv: MicroKV | None = None


@dataclass
class MemoryBank(CompiledAtoms):
    atoms: list[Atom] | None = None

    def to(self, device: torch.device | str) -> MemoryBank:
        return MemoryBank(
            keys=self.keys.to(device),
            factors_a={key: value.to(device) for key, value in self.factors_a.items()},
            factors_b={key: value.to(device) for key, value in self.factors_b.items()},
            mask_logits=self.mask_logits.to(device),
            micro_kv=None
            if self.micro_kv is None
            else MicroKV(self.micro_kv.keys.to(device), self.micro_kv.values.to(device)),
            atoms=self.atoms,
        )
