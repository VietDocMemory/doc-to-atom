from __future__ import annotations

import torch

from doc_to_atom.compiler.memory import MemoryBank
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import DynamicLoRAInjector
from doc_to_atom.router.model import TwoStageRouter


class QueryEngine:
    def __init__(self, model, tokenizer, encoder: SharedAtomQueryEncoder, router: TwoStageRouter, composer: MemoryComposer, injector: DynamicLoRAInjector):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.router = router
        self.composer = composer
        self.injector = injector

    @torch.no_grad()
    def answer(self, query: str, bank: MemoryBank, top_k: int = 8, max_new_tokens: int = 128) -> dict:
        encoded = self.tokenizer(query, return_tensors="pt").to(next(self.model.parameters()).device)
        query_embedding = self.encoder.encode_queries(**encoded)
        routed = self.router(query_embedding, bank.keys, bank.atoms or [], query, top_k)
        composed = self.composer(bank, routed.selected_indices, routed.routing_weights)
        with self.injector.activate(composed):
            output = self.model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens)
        prediction = self.tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        return {
            "prediction": prediction,
            "selected_atom_ids": [(bank.atoms or [])[i].atom_id for i in routed.selected_indices.tolist()],
            "stage1_scores": routed.stage1_scores.tolist(),
            "reranker_scores": routed.reranker_scores.tolist(),
            "metadata_biases": routed.metadata_biases.tolist(),
            "routing_weights": routed.routing_weights.tolist(),
            "composed_adapter_norm": float(composed.adapter_norm()),
        }

