from __future__ import annotations

from pathlib import Path

import torch

from doc_to_atom.atomization.models import Atom
from doc_to_atom.compiler.memory import MemoryBank, MicroKV
from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder


def build_memory_bank(
    atoms: list[Atom],
    tokenized_atoms: dict[str, torch.Tensor],
    encoder: SharedAtomQueryEncoder,
    compiler: AtomMemoryCompiler,
) -> MemoryBank:
    compiled = compiler(encoder.encode_atoms(**tokenized_atoms))
    return MemoryBank(**compiled.__dict__, atoms=atoms)


def save_memory_bank(bank: MemoryBank, path: str | Path) -> None:
    payload = {
        "keys": bank.keys.detach().cpu(),
        "factors_a": {key: value.detach().cpu() for key, value in bank.factors_a.items()},
        "factors_b": {key: value.detach().cpu() for key, value in bank.factors_b.items()},
        "mask_logits": bank.mask_logits.detach().cpu(),
        "micro_kv": None
        if bank.micro_kv is None
        else {"keys": bank.micro_kv.keys.detach().cpu(), "values": bank.micro_kv.values.detach().cpu()},
        "atoms": [atom.model_dump() for atom in bank.atoms or []],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_memory_bank(path: str | Path, device: str | torch.device = "cpu") -> MemoryBank:
    payload = torch.load(path, map_location=device, weights_only=True)
    micro_kv = payload["micro_kv"]
    return MemoryBank(
        keys=payload["keys"],
        factors_a=payload["factors_a"],
        factors_b=payload["factors_b"],
        mask_logits=payload["mask_logits"],
        micro_kv=None if micro_kv is None else MicroKV(**micro_kv),
        atoms=[Atom.model_validate(atom) for atom in payload["atoms"]],
    )

