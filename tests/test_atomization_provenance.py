import hashlib
import importlib
from types import SimpleNamespace

from doc_to_atom.atomization.client import OpenAIConfig, ThreeStageXMLAtomizer
from doc_to_atom.atomization.parser import parse_question_roles_xml

try:
    provenance = importlib.import_module("doc_to_atom.atomization.provenance")
except ModuleNotFoundError:
    provenance = SimpleNamespace()


class _FakeTransportAtomizer(ThreeStageXMLAtomizer):
    def __init__(self, responses):
        super().__init__(OpenAIConfig(api_key="test"), provider="openai")
        self._responses = iter(responses)

    def _request(self, system: str, user: str) -> str:
        return next(self._responses)


def test_raw_response_keeps_exact_content_and_hash_before_parsing():
    raw = "  <question_annotations>\n\t<question_annotation id=\"q1\"/>\n</question_annotations>  "
    atomizer = _FakeTransportAtomizer([raw])
    atomizer._begin_record("r1")

    returned = atomizer._call("question_annotation", "system", "user")
    parse_question_roles_xml(returned)

    assert returned == raw
    assert atomizer.last_raw_responses[0].content == raw
    assert atomizer.last_raw_responses[0].sha256 == hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()
    assert atomizer.last_raw_responses[0].stage == "question_annotation"
    assert atomizer.last_raw_responses[0].attempt == 1


def test_retries_are_retained_as_separate_immutable_attempts():
    atomizer = _FakeTransportAtomizer([" first ", " second "])
    atomizer._begin_record("r1")

    atomizer._call("decomposition", "system", "user")
    atomizer._call("decomposition", "system repair", "user")

    assert [item.attempt for item in atomizer.last_raw_responses] == [1, 2]
    assert [item.content for item in atomizer.last_raw_responses] == [" first ", " second "]
    try:
        atomizer.last_raw_responses[0].content = "mutated"
    except Exception:
        pass
    assert atomizer.last_raw_responses[0].content == " first "


def test_raw_response_sidecar_append_does_not_rewrite_prior_entries(tmp_path):
    path = tmp_path / "raw.jsonl"
    first = provenance.RawProviderResponse.capture(
        record_id="r1",
        stage="decomposition",
        attempt=1,
        provider="openai",
        model="test",
        content=" first ",
    )
    second = provenance.RawProviderResponse.capture(
        record_id="r1",
        stage="decomposition",
        attempt=2,
        provider="openai",
        model="test",
        content=" second ",
    )
    provenance.write_raw_response_jsonl(first, path)
    original_line = path.read_text(encoding="utf-8").splitlines()[0]

    provenance.write_raw_response_jsonl(second, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == original_line
    assert len(lines) == 2
    assert provenance.RawProviderResponse.model_validate_json(lines[1]).content == " second "
