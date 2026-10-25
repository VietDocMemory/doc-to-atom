from __future__ import annotations

import torch

from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.inference.answer_format import build_qa_prompt, generate_short_answer


class QueryEngine:
    def __init__(
        self,
        model,
        tokenizer,
        encoder: SharedAtomQueryEncoder,
        router: TwoStageRouter,
        composer: MemoryComposer,
        injector: DynamicLoRAInjector,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.router = router
        self.composer = composer
        self.injector = injector

    @torch.no_grad()
    def answer(
        self, query: str, bank: MemoryBank, top_k: int = 8, max_new_tokens: int = 128
    ) -> dict:
        query_encoded = self.tokenizer(query, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        generation_prompt = build_qa_prompt(query)
        query_embedding = self.encoder.encode_queries(**query_encoded)
        routed = self.router(query_embedding, bank.keys, bank.atoms or [], query, top_k)
        composed = self.composer(bank, routed.selected_indices, routed.routing_weights)
        candidate_scores = (
            routed.stage1_scores
            + self.router.rerank_fusion_weight * routed.reranker_scores
            + routed.metadata_biases
        )
        ranked_indices = routed.stage1_indices[torch.argsort(candidate_scores, descending=True)]
        with self.injector.activate(composed):
            generation = generate_short_answer(
                self.model, self.tokenizer, generation_prompt, max_new_tokens=max_new_tokens
            )
        return {
            **generation,
            "selected_atom_ids": [
                (bank.atoms or [])[i].atom_id for i in routed.selected_indices.tolist()
            ],
            "ranked_atom_ids": [(bank.atoms or [])[i].atom_id for i in ranked_indices.tolist()],
            "stage1_scores": routed.stage1_scores.tolist(),
            "reranker_scores": routed.reranker_scores.tolist(),
            "metadata_biases": routed.metadata_biases.tolist(),
            "routing_weights": routed.routing_weights.tolist(),
            "composed_adapter_norm": float(composed.adapter_norm()),
        }
