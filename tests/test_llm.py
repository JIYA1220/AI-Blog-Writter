import pytest

from utils import llm
from utils.llm import LLMConfigError, get_llm, resolve_provider


def test_resolve_ollama_prefix():
    assert resolve_provider("ollama/qwen2.5:7b") == ("ollama", "qwen2.5:7b")


def test_resolve_openrouter_prefix():
    assert resolve_provider("openrouter/meta-llama/llama-3.1-8b-instruct:free") == (
        "openrouter", "meta-llama/llama-3.1-8b-instruct:free")


def test_resolve_bare_name_is_gemini():
    assert resolve_provider("gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")


def test_resolve_legacy_vendor_model_uses_openrouter_only_when_it_is_the_only_key(monkeypatch):
    name = "meta-llama/llama-3.1-8b-instruct:free"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert resolve_provider(name) == ("openrouter", name)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    assert resolve_provider(name)[0] == "gemini"


def test_resolve_rejects_empty_and_prefix_only_names():
    with pytest.raises(LLMConfigError):
        resolve_provider("")
    with pytest.raises(LLMConfigError):
        resolve_provider("ollama/")


def test_get_llm_default_is_local_ollama():
    client = get_llm(temperature=0.3)
    assert type(client).__name__ == "ChatOllama"
    assert client.model == llm.DEFAULT_MODEL.split("/", 1)[1]


def test_get_llm_ollama_uses_requested_model(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "ollama/llama3")
    client = get_llm()
    assert type(client).__name__ == "ChatOllama"
    assert client.model == "llama3"


def test_get_llm_openrouter_uses_openai_client_with_openrouter_base_url(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "openrouter/meta-llama/llama-3.1-8b-instruct:free")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = get_llm()
    assert type(client).__name__ == "ChatOpenAI"
    assert client.model_name == "meta-llama/llama-3.1-8b-instruct:free"
    assert client.openai_api_base == llm.OPENROUTER_BASE_URL


def test_get_llm_openrouter_without_key_raises(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "openrouter/some/model")
    with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY"):
        get_llm()


def test_get_llm_openrouter_placeholder_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "openrouter/some/model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
    with pytest.raises(LLMConfigError):
        get_llm()


def test_get_llm_gemini_without_key_raises(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
    with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
        get_llm()


def test_get_llm_gemini_with_key_builds_google_client(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    client = get_llm()
    assert type(client).__name__ == "ChatGoogleGenerativeAI"


def test_openrouter_key_is_never_sent_to_gemini_client(monkeypatch):
    """Regression: the old code passed OPENROUTER_API_KEY to ChatGoogleGenerativeAI."""
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    with pytest.raises(LLMConfigError):
        get_llm()


def test_get_llm_explicit_model_name_overrides_env(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "ollama/llama3")
    assert get_llm(model_name="ollama/qwen2.5:3b").model == "qwen2.5:3b"


def test_get_llm_ollama_num_ctx_is_passed_through():
    assert get_llm(model_name="ollama/llama3.2:3b", num_ctx=8192).num_ctx == 8192
    assert get_llm(model_name="ollama/llama3.2:3b").num_ctx is None
