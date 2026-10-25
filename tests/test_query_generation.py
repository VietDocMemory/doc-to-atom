from __future__ import annotations

import torch
from types import SimpleNamespace
from contextlib import nullcontext

from doc_to_atom.inference.answer_format import build_qa_prompt, generate_short_answer
from doc_to_atom.inference.query import QueryEngine


class _Tokens(dict):
    def to(self, device):
        return self


class _Tokenizer:
    eos_token_id = 9
    pieces = {
        2: " Denver Broncos",
        3: "\nExplanation: the Broncos won.",
        4: "\n",
        5: "unanswerable",
        6: "\n\n",
        7: " Paris",
        9: "<eos>",
    }

    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        return _Tokens(input_ids=torch.tensor([[1]]), attention_mask=torch.tensor([[1]]))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            self.pieces.get(int(token), "")
            for token in ids
            if not (skip_special_tokens and int(token) == 9)
        )


class _Model(torch.nn.Module):
    def __init__(self, continuation):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.continuation = continuation
        self.received = None

    def generate(self, input_ids, attention_mask, *, do_sample, max_new_tokens,
                 stopping_criteria):
        self.received = (do_sample, max_new_tokens)
        output = input_ids.clone()
        for token in self.continuation[:max_new_tokens]:
            output = torch.cat([output, torch.tensor([[token]])], dim=1)
            if token == 9 or stopping_criteria(output, None):
                break
        return output


def test_short_answer_stops_on_first_completed_answer_line():
    model = _Model([2, 4, 3])

    result = generate_short_answer(model, _Tokenizer(), "Question: Who?\nAnswer:", max_new_tokens=8)

    assert result["prediction"] == "Denver Broncos"
    assert result["raw_generation"] == " Denver Broncos\n"
    assert result["stop_reason"] == "answer_line"
    assert model.received == (False, 8)


def test_short_answer_ignores_leading_newline_and_truncates_combined_token():
    model = _Model([6, 7, 3])

    result = generate_short_answer(model, _Tokenizer(), "Question: Where?\nAnswer:", max_new_tokens=8)

    assert result["prediction"] == "Paris"
    assert result["raw_generation"].endswith("\nExplanation: the Broncos won.")
    assert result["stop_reason"] == "answer_line"


def test_short_answer_respects_eos_and_token_cap():
    eos = generate_short_answer(_Model([5, 9, 3]), _Tokenizer(), "q", max_new_tokens=8)
    capped = generate_short_answer(_Model([2, 3]), _Tokenizer(), "q", max_new_tokens=1)

    assert eos["prediction"] == "unanswerable"
    assert eos["stop_reason"] == "eos"
    assert capped["prediction"] == "Denver Broncos"
    assert capped["stop_reason"] == "max_new_tokens"


def test_prompt_requests_short_answer_without_gold_text():
    prompt = build_qa_prompt("Who represented the AFC?", context=None)

    assert prompt.endswith("Question: Who represented the AFC?\nAnswer:")
    assert "short" in prompt.lower()
    assert "unanswerable" in prompt


def test_query_engine_uses_short_answer_generation_without_changing_routing():
    atom = SimpleNamespace(atom_id="a1")
    bank = SimpleNamespace(keys=torch.zeros(1, 2), atoms=[atom])
    route = SimpleNamespace(
        stage1_scores=torch.tensor([1.0]),
        reranker_scores=torch.tensor([0.0]),
        metadata_biases=torch.tensor([0.0]),
        stage1_indices=torch.tensor([0]),
        selected_indices=torch.tensor([0]),
        routing_weights=torch.tensor([1.0]),
    )
    encoder = SimpleNamespace(encode_queries=lambda **kwargs: torch.zeros(1, 2))
    class _Router:
        rerank_fusion_weight = 1.0

        def __call__(self, *args):
            return route

    router = _Router()
    composer = lambda *args: SimpleNamespace(adapter_norm=lambda: 1.0)
    injector = SimpleNamespace(activate=lambda composed: nullcontext())
    engine = QueryEngine(_Model([2, 4, 3]), _Tokenizer(), encoder, router, composer, injector)

    result = engine.answer("Who?", bank, top_k=1, max_new_tokens=8)

    assert result["prediction"] == "Denver Broncos"
    assert result["raw_generation"] == " Denver Broncos\n"
    assert result["selected_atom_ids"] == ["a1"]
