from doc_to_atom.atomization.client import GeminiAtomizer, GeminiConfig


def test_gemini_config_uses_documented_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config = GeminiConfig.from_env()
    assert config.api_key == "test-key"
    assert config.model == "gemini-3.5-flash-lite"


def test_gemini_atomizer_uses_google_compatible_endpoint(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("doc_to_atom.atomization.client.OpenAI", FakeOpenAI)
    atomizer = GeminiAtomizer(GeminiConfig(api_key="test-key"))
    assert atomizer.provider == "gemini"
    assert atomizer.client.kwargs == {
        "api_key": "test-key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
