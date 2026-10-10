from __future__ import annotations

import torch


class FrozenBGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 192,
        device: str | None = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        for parameter in self.model.model.parameters():
            parameter.requires_grad_(False)
        self.model.model.eval()

    def predict(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        return torch.as_tensor(self.model.predict(pairs, convert_to_numpy=True))
