from unittest.mock import Mock
import pytest
from doc_to_atom.evaluation import baselines

def test_chat_baseline_uses_template_without_duplicating_special_tokens(monkeypatch):
    model, tokenizer = Mock(), Mock()
    tokenizer.apply_chat_template.return_value = "<bos>templated"
    generate = Mock(return_value={"prediction":"answer"})
    monkeypatch.setattr(baselines,"generate_short_answer",generate)
    baseline=baselines.DocToLoRABaseline(model,tokenizer,"official",prompt_protocol="chat_v1")
    assert baseline.answer("document","question")=="answer"
    assert generate.call_args.args[2]=="<bos>templated"
    assert generate.call_args.kwargs["add_special_tokens"] is False
    assert model.reset.call_count==2

def test_d2l_adapter_is_cleared_on_failure(monkeypatch):
    model,tokenizer=Mock(),Mock()
    monkeypatch.setattr(baselines,"generate_short_answer",Mock(side_effect=RuntimeError("generation")))
    with pytest.raises(RuntimeError):
        baselines.DocToLoRABaseline(model,tokenizer,"raw").answer("doc","q")
    assert model.reset.call_count==2
