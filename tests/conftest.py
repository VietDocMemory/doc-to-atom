from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from torch import nn


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-molab",
        action="store_true",
        default=False,
        help="run tests requiring the Molab 2-GPU CUDA/NCCL/DeepSpeed environment",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-molab"):
        return
    skip = pytest.mark.skip(reason="requires --run-molab and two CUDA GPUs")
    for item in items:
        if "molab" in item.keywords:
            item.add_marker(skip)


class ToyAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.o_proj(torch.tanh(self.q_proj(x) + self.v_proj(x)))


class ToyMLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden, hidden, bias=False)


class ToyBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = ToyAttention(hidden)
        self.mlp = ToyMLP(hidden)

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp.down_proj(x)


class ToyBody(nn.Module):
    def __init__(self, vocab: int = 31, hidden: int = 8, depth: int = 6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([ToyBlock(hidden) for _ in range(depth)])


class ToyLM(nn.Module):
    def __init__(self, vocab: int = 31, hidden: int = 8, depth: int = 6):
        super().__init__()
        self.model = ToyBody(vocab, hidden, depth)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))
