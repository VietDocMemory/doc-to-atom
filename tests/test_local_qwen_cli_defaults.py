"""Local provider selection must never silently fall back to a paid API."""
import sys
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from doc_to_atom.atomization import cli


def test_cli_defaults_to_local_qwen(monkeypatch, capsys):
    run = Mock(return_value={"provider": "local_qwen"})
    monkeypatch.setattr(cli, "atomize_file", run)
    monkeypatch.setattr(cli, "load_project_env", Mock())
    monkeypatch.setattr(sys, "argv", ["doc2atom-atomize", "--input", "raw", "--output", "atoms"])
    cli.main()
    assert run.call_args.kwargs["provider"] == "local_qwen"
    assert run.call_args.kwargs["model"] is None


def test_cli_minimax_requires_explicit_provider(monkeypatch, capsys):
    run = Mock(return_value={"provider": "minimax"})
    monkeypatch.setattr(cli, "atomize_file", run)
    monkeypatch.setattr(cli, "load_project_env", Mock())
    monkeypatch.setattr(sys, "argv", [
        "doc2atom-atomize", "--input", "raw", "--output", "atoms", "--provider", "minimax",
    ])
    cli.main()
    assert run.call_args.kwargs["provider"] == "minimax"


def test_invalid_budget_rejected_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        cli.atomize_file(tmp_path / "missing", tmp_path / "atoms", provider="local_qwen",
                         max_completion_tokens=0)
    assert not (tmp_path / "atoms").exists()


def fake_local_atomizer(output_ids):
    import torch
    from doc_to_atom.atomization.local import LocalQwenAtomizer, LocalQwenConfig

    atomizer = LocalQwenAtomizer(LocalQwenConfig(max_completion_tokens=4))

    class Inputs(dict):
        def to(self, device):
            return self

    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "prompt"
    tokenizer.return_value = Inputs(input_ids=torch.tensor([[1, 2]]))
    tokenizer.decode.return_value = "<atoms/>"
    tokenizer.eos_token_id = 9
    atomizer._tokenizer = tokenizer
    atomizer._model = SimpleNamespace(
        device="cpu", config=SimpleNamespace(max_position_embeddings=100),
        generation_config=SimpleNamespace(eos_token_id=[9]),
        generate=Mock(return_value=torch.tensor([[1, 2, *output_ids]])),
    )
    return atomizer


def test_local_generation_and_provenance():
    atomizer = fake_local_atomizer([3, 9])
    atomizer._begin_record("r1")
    assert atomizer._call("decomposition", "system", "user") == "<atoms/>"
    captured = atomizer.last_raw_responses[0]
    assert captured.provider == "local_qwen"
    assert captured.model == "Qwen/Qwen3-14B"
    assert captured.request_parameters["enable_thinking"] is False
    assert json.loads(captured.raw_body)["finish_reason"] == "stop"
    assert atomizer._model.generate.call_args.kwargs["do_sample"] is False


def test_truncation_is_preserved_and_rejected():
    from doc_to_atom.atomization.client import MiniMaxAPIError

    atomizer = fake_local_atomizer([3, 4, 5, 6])
    atomizer._begin_record("r1")
    with pytest.raises(MiniMaxAPIError, match="incomplete"):
        atomizer._call("decomposition", "system", "user")
    assert json.loads(atomizer.last_raw_responses[0].raw_body)["finish_reason"] == "length"


def test_context_overflow_does_not_truncate():
    from doc_to_atom.atomization.client import MiniMaxAPIError

    atomizer = fake_local_atomizer([9])
    atomizer._model.config.max_position_embeddings = 3
    with pytest.raises(MiniMaxAPIError, match="context"):
        atomizer._request("system", "user")
    atomizer._model.generate.assert_not_called()


def test_local_does_not_read_api_credentials(tmp_path, monkeypatch):
    from doc_to_atom.atomization import local

    monkeypatch.setattr(cli.MiniMaxConfig, "from_env", Mock(side_effect=AssertionError("API forbidden")))
    fake = Mock()
    fake.atomize.return_value = None
    fake.last_errors = ["intentional"]
    factory = Mock(return_value=fake)
    monkeypatch.setattr(local, "LocalQwenAtomizer", factory)
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"id": "r1", "context": "text", "questions": []}) + "\n")
    with pytest.raises(RuntimeError, match="discarded"):
        cli.atomize_file(source, tmp_path / "atoms.jsonl", provider="local_qwen")
    assert factory.call_args.args[0].model == "Qwen/Qwen3-14B"


def test_cache_only_resume_does_not_load_gpu(tmp_path, monkeypatch):
    from doc_to_atom.atomization import local

    factory = Mock(side_effect=AssertionError("model load forbidden"))
    monkeypatch.setattr(local, "LocalQwenAtomizer", factory)
    source = tmp_path / "raw.jsonl"
    source.write_text("")
    cli.atomize_file(source, tmp_path / "atoms.jsonl", provider="local_qwen")
    factory.assert_not_called()


def test_initialization_failure_is_fatal(monkeypatch):
    from doc_to_atom.atomization.local import LocalQwenAtomizer, LocalQwenConfig
    from doc_to_atom.atomization.client import MiniMaxAPIError

    atomizer = LocalQwenAtomizer(LocalQwenConfig())
    monkeypatch.setattr(atomizer, "_load_model", Mock(side_effect=RuntimeError("missing GPU")))
    with pytest.raises(MiniMaxAPIError, match="initialization failed"):
        atomizer.atomize("r1", "text", [])
