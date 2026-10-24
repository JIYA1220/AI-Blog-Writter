import pytest

from utils.llm import GROQ_BASE_URL, LLMConfigError, get_llm, resolve_provider


# ---- Groq provider ---------------------------------------------------------------------

def test_resolve_groq_prefix():
    assert resolve_provider("groq/llama-3.3-70b-versatile") == ("groq", "llama-3.3-70b-versatile")


def test_groq_without_key_raises_and_names_the_variable(monkeypatch):
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        get_llm(model_name="groq/llama-3.3-70b-versatile")


def test_groq_builds_openai_compatible_client_with_json_mode(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    client = get_llm(temperature=0.1, model_name="groq/llama-3.3-70b-versatile", json_mode=True, max_tokens=700)
    assert type(client).__name__ == "ChatOpenAI"
    assert client.openai_api_base == GROQ_BASE_URL == "https://api.groq.com/openai/v1"
    payload = client._get_request_payload("hello")
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["temperature"] == 0.1
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_completion_tokens"] == 700
    assert client.max_retries == 0


def test_groq_without_json_mode_sends_no_response_format(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    client = get_llm(model_name="groq/llama-3.3-70b-versatile")
    assert "response_format" not in client._get_request_payload("hello")
