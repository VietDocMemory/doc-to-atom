import torch
from doc_to_atom.encoder.tokenization import QUERY_MAX_TOKENS, tokenize_query

def test_query_encoder_uses_one_explicit_train_inference_contract():
    received = {}
    def tokenizer(texts, **kwargs):
        received.update(texts=texts, **kwargs)
        ids = torch.arange(QUERY_MAX_TOKENS).reshape(1,-1)
        return {"input_ids":ids,"attention_mask":torch.ones_like(ids)}
    encoded = tokenize_query(tokenizer, "long question " * 300, torch.device("cpu"))
    assert received["texts"] == ["long question " * 300]
    assert received["max_length"] == 256
    assert received["padding"] is True and received["truncation"] is True
    assert received["return_tensors"] == "pt"
    assert encoded["input_ids"].shape == (1,256)
