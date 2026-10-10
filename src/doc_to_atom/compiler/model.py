from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .memory import CompiledAtoms, MicroKV


@dataclass(frozen=True)
class TargetShape:
    layer_index: int
    module_name: str
    input_dim: int
    output_dim: int

    @property
    def key(self) -> str:
        return f"{self.layer_index}:{self.module_name}"


class AtomMemoryCompiler(nn.Module):
    """Paper compiler: shared 256-D trunk with provenance, LoRA, mask and optional KV heads."""

    def __init__(
        self,
        targets: list[TargetShape],
        input_dim: int = 256,
        latent_dim: int = 256,
        key_dim: int = 256,
        rank: int = 8,
        a_std: float = 0.02,
        micro_kv_slots: int | None = None,
        micro_kv_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.targets = targets
        self.rank = rank
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim, dtype=torch.float32),
            nn.Linear(input_dim, latent_dim, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim, dtype=torch.float32),
        )
        self.key_head = nn.Sequential(
            nn.LayerNorm(latent_dim, dtype=torch.float32),
            nn.Linear(latent_dim, key_dim, dtype=torch.float32),
        )
        self.a_heads = nn.ModuleDict()
        self.b_heads = nn.ModuleDict()
        for target in targets:
            safe_key = target.key.replace(":", "__")
            a_head = nn.Linear(latent_dim, rank * target.input_dim, dtype=torch.float32)
            b_head = nn.Linear(latent_dim, target.output_dim * rank, dtype=torch.float32)
            nn.init.normal_(a_head.weight, mean=0.0, std=a_std)
            nn.init.zeros_(a_head.bias)
            nn.init.zeros_(b_head.weight)
            nn.init.zeros_(b_head.bias)
            self.a_heads[safe_key] = a_head
            self.b_heads[safe_key] = b_head
        self.mask_head = nn.Linear(latent_dim, len(targets), dtype=torch.float32)
        self.micro_kv_slots = micro_kv_slots
        self.micro_kv_head_dim = micro_kv_head_dim
        if (micro_kv_slots is None) != (micro_kv_head_dim is None):
            raise ValueError("micro-KV slots and head_dim must be configured together")
        self.micro_kv_head = (
            None
            if micro_kv_slots is None
            else nn.Linear(
                latent_dim,
                2 * len({target.layer_index for target in targets}) * micro_kv_slots * micro_kv_head_dim,
                dtype=torch.float32,
            )
        )

    def forward(self, atom_embeddings: torch.Tensor) -> CompiledAtoms:
        latent = self.trunk(atom_embeddings.float())
        keys = F.normalize(self.key_head(latent), p=2, dim=-1)
        factors_a: dict[str, torch.Tensor] = {}
        factors_b: dict[str, torch.Tensor] = {}
        for target in self.targets:
            safe_key = target.key.replace(":", "__")
            factors_a[target.key] = self.a_heads[safe_key](latent).view(
                -1, self.rank, target.input_dim
            )
            factors_b[target.key] = self.b_heads[safe_key](latent).view(
                -1, target.output_dim, self.rank
            )
        micro_kv = None
        if self.micro_kv_head is not None:
            layer_count = len({target.layer_index for target in self.targets})
            kv = self.micro_kv_head(latent).view(
                -1, 2, layer_count, self.micro_kv_slots, self.micro_kv_head_dim
            )
            micro_kv = MicroKV(keys=kv[:, 0], values=kv[:, 1])
        return CompiledAtoms(
            keys=keys,
            factors_a=factors_a,
            factors_b=factors_b,
            mask_logits=self.mask_head(latent),
            micro_kv=micro_kv,
        )

