from __future__ import annotations

import torch

from doc_to_atom.evaluation.baselines import BaseQueryOnly, BaseWithContext, DocToLoRABaseline
from test_query_generation import _Model, _Tokenizer


def test_query_and_context_baselines_share_one_line_stop_policy():
    query = BaseQueryOnly(_Model([2, 4, 3]), _Tokenizer(), max_new_tokens=8)
    context = BaseWithContext(_Model([2, 4, 3]), _Tokenizer(), max_new_tokens=8)

    assert query.answer("Document", "Who?") == "Denver Broncos"
    assert context.answer("Document", "Who?") == "Denver Broncos"


def test_doc_to_lora_baseline_keeps_its_prompt_but_shares_stop_policy():
    class D2L(_Model):
        device = torch.device("cpu")

        def reset(self):
            pass

        def internalize(self, document):
            self.document = document

    model = D2L([2, 4, 3])
    baseline = DocToLoRABaseline(model, _Tokenizer(), variant="raw", max_new_tokens=8)

    assert baseline.answer("Document", "Who?") == "Denver Broncos"
    assert model.document == "Document"
