# utils/llm.py
# ============================================================
# AI Blog Writer: LLM client factory
#
# MODEL_NAME selects the provider:
#   ollama/<model>       local Ollama server        (no key)
#   openrouter/<model>   OpenRouter, OpenAI-compatible API (OPENROUTER_API_KEY)
#   groq/<model>         Groq, OpenAI-compatible API    (GROQ_API_KEY)
#   <anything else>      Google Gemini model name   (GOOGLE_API_KEY)
#
# Backward compatibility: a bare "vendor/model" name (for example
# "meta-llama/llama-3.1-8b-instruct:free") is sent to OpenRouter when only
# OPENROUTER_API_KEY is set. Prefer the explicit "openrouter/" prefix.
# ============================================================

import os
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()

# Enable LangSmith Tracing if API key is present
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ai-blog-writer")

DEFAULT_MODEL = "ollama/qwen2.5:7b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMConfigError(ValueError):
    """Raised when MODEL_NAME or the matching API key is missing or invalid."""


def _get_key(name: str) -> str:
    """Return an API key from the environment, or '' if unset or still the .env.example placeholder."""
    value = os.getenv(name, "").strip()
    if not value or (value.startswith("your_") and value.endswith("_here")):
        return ""
    return value


def resolve_provider(model_name: str) -> Tuple[str, str]:
    """Map MODEL_NAME to (provider, model) where provider is 'ollama', 'openrouter' or 'gemini'."""
    name = (model_name or "").strip()
    if not name:
        raise LLMConfigError("MODEL_NAME is empty. Example: MODEL_NAME=ollama/qwen2.5:7b")

    for prefix, provider in (("ollama/", "ollama"), ("openrouter/", "openrouter"), ("groq/", "groq")):
        if name.startswith(prefix):
            model = name[len(prefix):]
            if not model:
                raise LLMConfigError(f"MODEL_NAME '{name}' has no model after '{prefix}'")
            return provider, model

    # Legacy form: bare "vendor/model" with only an OpenRouter key available
    if "/" in name and not _get_key("GOOGLE_API_KEY") and _get_key("OPENROUTER_API_KEY"):
        return "openrouter", name

    return "gemini", name


def get_llm(temperature: float = 0.7, model_name: str = None, num_ctx: int = None,
            json_mode: bool = False, max_tokens: int = None):
    """
    Return a chat model for `model_name`, or for MODEL_NAME when it is not given.
    num_ctx: Ollama context window. json_mode: ask Groq for a guaranteed JSON-object reply.
    max_tokens: reply-length cap for the OpenAI-compatible providers (default 4096).
    """
    provider, model = resolve_provider(model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL))

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        extra = {"num_ctx": num_ctx} if num_ctx else {}      # Ollama defaults to a short context window
        return ChatOllama(model=model, temperature=temperature, **extra)

    if provider == "groq":
        key = _get_key("GROQ_API_KEY")
        if not key:
            raise LLMConfigError(
                f"MODEL_NAME selects Groq ('{model}') but GROQ_API_KEY is not set "
                "(put it in the environment or in a git-ignored .env file)."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=GROQ_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            max_retries=0,   # rate-limit retries are handled by the caller
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
        )

    if provider == "openrouter":
        key = _get_key("OPENROUTER_API_KEY")
        if not key:
            raise LLMConfigError(
                f"MODEL_NAME selects OpenRouter ('{model}') but OPENROUTER_API_KEY is not set."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=OPENROUTER_BASE_URL,
            temperature=temperature,
            max_tokens=4096,
        )

    key = _get_key("GOOGLE_API_KEY")
    if not key:
        raise LLMConfigError(
            f"MODEL_NAME '{model}' is treated as a Gemini model but GOOGLE_API_KEY is not set.\n"
            "  Local Ollama:  MODEL_NAME=ollama/qwen2.5:7b\n"
            "  OpenRouter:    MODEL_NAME=openrouter/<model>  with OPENROUTER_API_KEY\n"
            "  Gemini:        MODEL_NAME=<gemini model name>  with GOOGLE_API_KEY"
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=temperature,
        max_tokens=4096,
    )
