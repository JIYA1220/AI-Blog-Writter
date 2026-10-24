import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeLLM:
    """Offline stand-in for a chat model. `responder(prompt) -> str` scripts the reply."""

    def __init__(self, responder):
        self.responder = responder
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.responder(prompt))


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No test may depend on, or leak, real keys or model names from a developer's .env."""
    for var in ("MODEL_NAME", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY",
                "ENABLE_SEARCH", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
