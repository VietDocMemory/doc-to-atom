from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from doc_to_atom.atomization.models import Atom

SEMANTIC_TYPES = [
    "fact_claim",
    "entity_attribute",
    "event_relation",
    "process_step",
    "evidence_fragment",
]


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> torch.Tensor | list[float]: ...


@dataclass
class RouterOutput:
    all_stage1_scores: torch.Tensor
    stage1_indices: torch.Tensor
    selected_indices: torch.Tensor
    stage1_scores: torch.Tensor
    reranker_scores: torch.Tensor
    metadata_biases: torch.Tensor
    final_scores: torch.Tensor
    routing_weights: torch.Tensor


class TwoStageRouter(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        key_dim: int = 256,
        stage1_top_k: int = 32,
        rerank_fusion_weight: float = 1.0,
        temperature: float = 1.0,
        reranker: Reranker | None = None,
    ) -> None:
        super().__init__()
        self.query_key_head = nn.Sequential(
            nn.LayerNorm(embedding_dim, dtype=torch.float32),
            nn.Linear(embedding_dim, key_dim, dtype=torch.float32),
        )
        self.stage1_top_k = stage1_top_k
        self.rerank_fusion_weight = rerank_fusion_weight
        self.temperature = temperature
        self.reranker = reranker
        self.metadata_weights = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.semantic_type_scalar = nn.Embedding(len(SEMANTIC_TYPES), 1, dtype=torch.float32)

    def _metadata_features(self, atoms: list[Atom], device: torch.device) -> torch.Tensor:
        type_ids = torch.tensor(
            [SEMANTIC_TYPES.index(atom.semantic_type) for atom in atoms], device=device
        )
        type_signal = self.semantic_type_scalar(type_ids).squeeze(-1)
        return torch.stack(
            [
                torch.tensor([atom.confidence for atom in atoms], device=device),
                torch.tensor([float(atom.is_answer_bearing) for atom in atoms], device=device),
                torch.tensor(
                    [float(atom.conflict_group is not None) for atom in atoms], device=device
                ),
                type_signal,
            ],
            dim=-1,
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        atom_keys: torch.Tensor,
        atoms: list[Atom],
        query_text: str,
        final_top_k: int,
    ) -> RouterOutput:
        if query_embedding.ndim == 2:
            if query_embedding.shape[0] != 1:
                raise ValueError("current memory-bank router API handles one query at a time")
            query_embedding = query_embedding[0]
        query_key = F.normalize(self.query_key_head(query_embedding.float()), p=2, dim=-1)
        normalized_atom_keys = F.normalize(atom_keys.float(), p=2, dim=-1)
        similarities = normalized_atom_keys @ query_key
        k1 = min(self.stage1_top_k, len(atoms))
        stage1_scores, stage1_indices = torch.topk(similarities, k=k1)
        candidate_atoms = [atoms[index] for index in stage1_indices.tolist()]

        if self.reranker is None:
            reranker_scores = torch.zeros_like(stage1_scores)
        else:
            pairs = [(query_text, atom.retrieval_text) for atom in candidate_atoms]
            with torch.no_grad():
                raw = self.reranker.predict(pairs)
            reranker_scores = torch.as_tensor(
                raw, device=stage1_scores.device, dtype=stage1_scores.dtype
            )

        features = self._metadata_features(candidate_atoms, stage1_scores.device)
        metadata_biases = features @ self.metadata_weights
        candidate_final = (
            stage1_scores + self.rerank_fusion_weight * reranker_scores + metadata_biases
        )
        top_k = min(final_top_k, k1)
        final_scores, local_indices = torch.topk(candidate_final, k=top_k)
        selected_indices = stage1_indices[local_indices]
        routing_weights = F.softmax(final_scores / self.temperature, dim=0)
        return RouterOutput(
            all_stage1_scores=similarities,
            stage1_indices=stage1_indices,
            selected_indices=selected_indices,
            stage1_scores=stage1_scores,
            reranker_scores=reranker_scores,
            metadata_biases=metadata_biases,
            final_scores=final_scores,
            routing_weights=routing_weights,
        )
