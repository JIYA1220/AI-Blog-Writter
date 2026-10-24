import asyncio
import json

import pytest

from agents.planner import planner_node, section_word_budget

LLM_PLAN = {"topic": "T", "target_audience": "a", "tone": "t", "total_word_budget": 999,
            "sections": [{"section_id": i, "title": f"S{i}", "goal": "g", "constraints": [],
                          "word_budget": 999, "requires_retrieval": False} for i in range(4)]}


def _run(monkeypatch, fake_llm, reply):
    monkeypatch.setattr("agents.planner.get_llm", lambda temperature=0.4: fake_llm(lambda p: reply))
    return asyncio.run(planner_node({"topic": "T", "target_audience": "a", "tone": "t"}))


def test_section_word_budget_default_and_env(monkeypatch):
    assert section_word_budget() == 300
    monkeypatch.setenv("SECTION_WORD_BUDGET", "150")
    assert section_word_budget() == 150


@pytest.mark.parametrize("bad", ["abc", "0", "-5", ""])
def test_section_word_budget_ignores_invalid_values(monkeypatch, bad):
    monkeypatch.setenv("SECTION_WORD_BUDGET", bad)
    assert section_word_budget() == 300


def test_plan_uses_configured_budget_not_the_models_echo(monkeypatch, fake_llm):
    monkeypatch.setenv("SECTION_WORD_BUDGET", "150")
    out = _run(monkeypatch, fake_llm, json.dumps(LLM_PLAN))
    assert [s.word_budget for s in out["blog_plan"].sections] == [150] * 4


def test_planner_prompt_states_configured_budget(monkeypatch, fake_llm):
    monkeypatch.setenv("SECTION_WORD_BUDGET", "400")
    llm = fake_llm(lambda p: json.dumps(LLM_PLAN))
    monkeypatch.setattr("agents.planner.get_llm", lambda temperature=0.4: llm)
    asyncio.run(planner_node({"topic": "T", "target_audience": "a", "tone": "t"}))
    assert "Per-Section Budget: 400 words" in llm.prompts[0]
    assert "Total Word Budget: 2000 words" in llm.prompts[0]


def test_fallback_plan_on_bad_json_uses_configured_budget(monkeypatch, fake_llm):
    monkeypatch.setenv("SECTION_WORD_BUDGET", "150")
    out = _run(monkeypatch, fake_llm, "not json at all")
    assert len(out["blog_plan"].sections) == 5
    assert {s.word_budget for s in out["blog_plan"].sections} == {150}
    assert out["errors"] and out["errors"][0].startswith("Planner parse error")
