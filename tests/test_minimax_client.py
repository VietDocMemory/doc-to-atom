"""Official MiniMax HTTP contract, tested offline with no real credentials."""

import hashlib
import json

import httpx
import pytest

from doc_to_atom.atomization import cli
from doc_to_atom.atomization.client import (
    MINIMAX_CHAT_ENDPOINT,
    MiniMaxAPIError,
    MiniMaxAtomizer,
    MiniMaxConfig,
)
from doc_to_atom.atomization.models import AnnotationMeta, Atom, StructuredRecord
from doc_to_atom.atomization.provenance import RawProviderResponse


def mock_post(monkeypatch, handler):
    """Intercept the SDK's HTTP transport without bypassing SDK request building."""
    def send(client, request, **kwargs):
        response = handler(
            str(request.url), headers=request.headers,
            json=json.loads(request.content), follow_redirects=client.follow_redirects,
        )
        response.request = request
        return response

    monkeypatch.setattr(httpx.Client, "send", send)


def reply(content="<atoms/>", finish="stop", **extra):
    return {
        "id": "minimax-request-id", "model": "MiniMax-M3.1-Flash",
        "choices": [{"finish_reason": finish, "message": {
            "content": content, "reasoning_content": "private thinking, not annotation XML",
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "base_resp": {"status_code": 0, "status_msg": ""}, **extra,
    }


def test_official_endpoint_bearer_and_schema(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return httpx.Response(200, json=reply())

    mock_post(monkeypatch, post)
    # A legacy MiniMax key takes precedence; arbitrary hosts cannot redirect it.
    monkeypatch.setenv("OPENAI_API_KEY", "obsolete-secret")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://untrusted.invalid")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    config = MiniMaxConfig.from_env()
    assert config.model == "MiniMax-M3.1-Flash"
    assert "test-key" not in repr(config)
    atomizer = MiniMaxAtomizer(config)
    atomizer._begin_record("r")
    assert atomizer._call("decomposition", "system", "document") == "<atoms/>"
    assert seen["url"] == MINIMAX_CHAT_ENDPOINT
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert not seen["follow_redirects"]
    assert seen["json"] == {
        "model": "MiniMax-M3.1-Flash", "max_completion_tokens": 8192,
        "reasoning_effort": "low", "reasoning_split": True,
        "messages": [{"role": "system", "content": "system"},
                     {"role": "user", "content": "document"}],
    }
    raw = atomizer.last_raw_responses[0]
    body = json.loads(raw.raw_body)
    assert body["id"] == "minimax-request-id"
    assert body["usage"]["completion_tokens"] == 20
    assert body["choices"][0]["message"]["reasoning_content"]
    assert "private thinking" not in raw.content
    assert raw.raw_body_sha256 == hashlib.sha256(raw.raw_body.encode()).hexdigest()


def test_documented_sdk_environment(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "minimax-sdk-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.minimax.io/v1")
    config = MiniMaxConfig.from_env()
    assert config.api_key == "minimax-sdk-key"
    mock_post(monkeypatch, lambda *a, **k: httpx.Response(200, json=reply()))
    atomizer = MiniMaxAtomizer(config)
    assert atomizer._call("decomposition", "system", "user") == "<atoms/>"


def test_sdk_environment_cannot_redirect_minimax_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "minimax-sdk-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.invalid/v1")
    with pytest.raises(ValueError, match="OPENAI_BASE_URL must be"):
        MiniMaxConfig.from_env()


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_sdk_forwards_supported_reasoning_depths(effort, monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(kwargs["json"])
        return httpx.Response(200, json=reply())

    mock_post(monkeypatch, post)
    atomizer = MiniMaxAtomizer(MiniMaxConfig(api_key="test", reasoning_effort=effort))
    atomizer._call("decomposition", "system", "user")
    assert seen["reasoning_effort"] == effort
    assert "temperature" not in seen
    assert "thinking" not in seen  # Default adaptive thinking, never disabled.


def test_older_model_omits_flash_only_reasoning_effort(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(kwargs["json"])
        return httpx.Response(200, json=reply(model="MiniMax-M2.5"))

    mock_post(monkeypatch, post)
    atomizer = MiniMaxAtomizer(MiniMaxConfig(api_key="test", model="MiniMax-M2.5"))
    atomizer._call("decomposition", "system", "user")
    assert "reasoning_effort" not in seen


@pytest.mark.parametrize("status", [301, 400, 401, 402, 403, 429, 500])
def test_http_failures_are_captured_and_stop_without_retries(status, monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return httpx.Response(status, json={"error": {"type": "provider_error"}})

    mock_post(monkeypatch, post)
    atomizer = MiniMaxAtomizer(MiniMaxConfig(api_key="test"))
    with pytest.raises(MiniMaxAPIError, match=f"HTTP {status}"):
        atomizer.atomize("r", "document", [])
    assert calls == [1]
    assert atomizer.last_raw_responses[0].http_status == status
    assert json.loads(atomizer.last_raw_responses[0].raw_body)["error"]


@pytest.mark.parametrize("payload", [
    reply("partial XML", "length"), reply("", "stop"), reply("", "content_filter"),
    reply("", "tool_calls"), {"base_resp": {"status_code": 1008}}, {"choices": []},
])
def test_incomplete_or_invalid_response_is_retained_not_parsed(payload, monkeypatch):
    mock_post(monkeypatch, lambda *a, **k: httpx.Response(200, json=payload))
    atomizer = MiniMaxAtomizer(MiniMaxConfig(api_key="test"))
    with pytest.raises(MiniMaxAPIError):
        atomizer.atomize("r", "document", [])
    assert len(atomizer.last_raw_responses) == 1
    assert json.loads(atomizer.last_raw_responses[0].raw_body) == payload


def test_non_json_and_network_errors_do_not_leak_credentials(monkeypatch):
    atomizer = MiniMaxAtomizer(MiniMaxConfig(api_key="do-not-print"))
    mock_post(monkeypatch, lambda *a, **k: httpx.Response(200, text="not-json"))
    with pytest.raises(MiniMaxAPIError, match="invalid Chat Completions"):
        atomizer.atomize("r", "document", [])
    assert atomizer.last_raw_responses[0].raw_body == "not-json"

    def failed(*args, **kwargs):
        raise httpx.ConnectError("potentially sensitive diagnostic")

    mock_post(monkeypatch, failed)
    with pytest.raises(MiniMaxAPIError, match="connection failed") as error:
        atomizer.atomize("r", "document", [])
    assert "do-not-print" not in str(error.value)
    assert "sensitive diagnostic" not in str(error.value)


def test_inline_thinking_is_only_stripped_from_parser_input(monkeypatch):
    content = "<think>analysis with <atoms>not XML</atoms></think>\n<atoms/>"
    mock_post(monkeypatch, lambda *a, **k: httpx.Response(200, json=reply(content)))
    atomizer = MiniMaxAtomizer(MiniMaxConfig(api_key="test"))
    assert atomizer._call("decomposition", "system", "user") == "\n<atoms/>"
    assert atomizer.last_raw_responses[0].content == content


def legacy_record(*, valid=True):
    return StructuredRecord(
        record_id="cached", context="A", atoms=[Atom(
            atom_id="a", semantic_type="fact_claim", content="A", source_span="A",
            retrieval_text="A", is_answer_bearing=True, abstraction_level="evidence",
            confidence=1, span_valid=valid,
        )],
        annotation_meta=AnnotationMeta(prompt_provenance="HISTORICAL provider=openai; gpt-test"),
    )


def test_legacy_cache_and_raw_sidecar_are_not_relabelled_or_rewritten(tmp_path, monkeypatch):
    source, output = tmp_path / "raw.jsonl", tmp_path / "atoms.jsonl"
    source.write_text('{"id":"cached","context":"A","questions":[]}\n', encoding="utf-8")
    output.write_text(legacy_record().model_dump_json() + "\n", encoding="utf-8")
    sidecar = tmp_path / "atoms.jsonl.raw-responses.jsonl"
    old = RawProviderResponse.capture(record_id="cached", stage="decomposition", attempt=1,
                                       provider="openai", model="gpt-test", content="old XML")
    # Simulate the exact old sidecar schema with no optional additions.
    sidecar.write_text(old.model_dump_json(exclude_none=True) + "\n", encoding="utf-8")
    original, raw_original = output.read_bytes(), sidecar.read_bytes()
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    mock_post(monkeypatch, lambda *a, **k: pytest.fail("cache-only run made a request"))
    result = cli.atomize_file(source, output, resume=True, retry_invalid=True)
    assert result["records_cached"] == 1
    assert output.read_bytes() == original
    assert sidecar.read_bytes() == raw_original
    assert RawProviderResponse.model_validate_json(raw_original).provider == "openai"


def test_retry_invalid_keeps_full_snapshot_and_raw_sidecar(tmp_path, monkeypatch):
    source, output = tmp_path / "source.jsonl", tmp_path / "atoms.jsonl"
    source.write_text('{"id":"cached","context":"A","questions":[]}\n', encoding="utf-8")
    output.write_text(legacy_record(valid=False).model_dump_json() + "\n", encoding="utf-8")
    sidecar = tmp_path / "atoms.jsonl.raw-responses.jsonl"
    sidecar.write_text("historical raw bytes\n", encoding="utf-8")
    original, raw_original = output.read_bytes(), sidecar.read_bytes()

    class FakeAtomizer:
        def __init__(self, config):
            pass

        def atomize(self, *args):
            return legacy_record()  # Transport irrelevant to snapshot retention.

    monkeypatch.setattr(cli, "MiniMaxAtomizer", FakeAtomizer)
    monkeypatch.setenv("MINIMAX_API_KEY", "test")
    cli.atomize_file(source, output, retry_invalid=True)
    [backup] = list(tmp_path.glob("*.before-retry-*.jsonl"))
    assert backup.read_bytes() == original
    assert sidecar.read_bytes() == raw_original


def test_non_minimax_provider_is_rejected_before_opening_files(tmp_path):
    with pytest.raises(ValueError, match="unsupported annotation provider"):
        cli.atomize_file(tmp_path / "missing", tmp_path / "out", provider="obsolete")
    assert not list(tmp_path.iterdir())


def test_new_record_provenance_records_provider_and_models():
    meta = AnnotationMeta()
    atomizer = MiniMaxAtomizer(MiniMaxConfig(api_key="test"))
    atomizer._stamp_provenance(meta)
    assert meta.prompt_provenance == (
        "provider=minimax; requested_model=MiniMax-M3.1-Flash; model=MiniMax-M3.1-Flash"
    )
    paper_atomizer = MiniMaxAtomizer(MiniMaxConfig(api_key="test", model="MiniMax-M2.5"))
    paper_atomizer._stamp_provenance(meta)
    assert meta.prompt_provenance == (
        "provider=minimax; requested_model=MiniMax-M2.5; model=MiniMax-M2.5"
    )
