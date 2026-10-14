from doc_to_atom.atomization.client import GroqAtomizer, GroqConfig


def test_groq_config_uses_gpt_oss_default_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    config = GroqConfig.from_env()
    assert config.api_key == "test-key"
    assert config.model == "openai/gpt-oss-20b"


def test_groq_atomizer_uses_openai_compatible_endpoint(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("doc_to_atom.atomization.client.OpenAI", FakeOpenAI)
    atomizer = GroqAtomizer(GroqConfig(api_key="test-key"))
    assert atomizer.provider == "groq"
    assert atomizer.client.kwargs == {
        "api_key": "test-key",
        "base_url": "https://api.groq.com/openai/v1",
    }
