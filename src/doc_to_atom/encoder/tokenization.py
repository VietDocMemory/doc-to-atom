"""Shared query encoding contract for training and generation-time routing."""
from __future__ import annotations
import torch
QUERY_MAX_TOKENS = 256

def tokenize_query(tokenizer, question: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer([question], padding=True, truncation=True,
                        max_length=QUERY_MAX_TOKENS, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}
