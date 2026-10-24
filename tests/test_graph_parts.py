import asyncio

import pytest
from pydantic import ValidationError

from agents.reducer import reducer_node
from agents.router import route_decision, router_node
from agents.writer import dispatch_sections
from schemas.models import (BlogEvaluation, BlogPlan, RetrievalResult, SectionPlan,
                            WrittenSection)


def _plan(n=3):
    return BlogPlan(topic="Topic", target_audience="devs", tone="plain",
                    sections=[SectionPlan(section_id=i, title=f"S{i}", goal="g") for i in range(n)])


# ---- schemas -------------------------------------------------------------------------

def test_section_plan_defaults():
    s = SectionPlan(section_id=0, title="t", goal="g")
    assert s.word_budget == 300 and s.constraints == [] and s.requires_retrieval is False


def test_section_plan_requires_fields():
    with pytest.raises(ValidationError):
        SectionPlan(section_id=0, title="t")           # goal missing


def test_blog_evaluation_requires_integer_score():
    with pytest.raises(ValidationError):
        BlogEvaluation(score="not a number", reasoning="r")


def test_blog_plan_round_trips():
    plan = _plan()
    assert BlogPlan.model_validate(plan.model_dump()) == plan


# ---- routing -------------------------------------------------------------------------

def test_route_decision_both_branches():
    assert route_decision({"needs_retrieval": True}) == "retriever"
    assert route_decision({"needs_retrieval": False}) == "planner"
    assert route_decision({}) == "planner"


def test_router_forces_generation_without_tavily_key():
    out = asyncio.run(router_node({"topic": "t", "target_audience": "a"}))
    assert out["needs_retrieval"] is False


def test_router_uses_llm_decision_when_search_enabled(monkeypatch, fake_llm):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr("agents.router.get_llm", lambda temperature=0.1: fake_llm(
        lambda p: '```json\n{"needs_retrieval": true, "reason": "recent"}\n```'))
    out = asyncio.run(router_node({"topic": "t", "target_audience": "a"}))
    assert out["needs_retrieval"] is True


def test_router_falls_back_to_generation_on_bad_json(monkeypatch, fake_llm):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr("agents.router.get_llm", lambda temperature=0.1: fake_llm(lambda p: "garbage"))
    out = asyncio.run(router_node({"topic": "t", "target_audience": "a"}))
    assert out["needs_retrieval"] is False and out["errors"]


# ---- dispatcher ----------------------------------------------------------------------

def test_dispatch_yields_one_send_per_section():
    sends = dispatch_sections({"blog_plan": _plan(4)})
    assert len(sends) == 4
    assert all(s.node == "section_writer" for s in sends)
    assert [s.arg["section"].section_id for s in sends] == [0, 1, 2, 3]


def test_dispatch_passes_retrieval_snippets_to_workers():
    state = {"blog_plan": _plan(2),
             "retrieval_results": RetrievalResult(query="q", snippets=["fact one"], sources=["u"])}
    assert "fact one" in dispatch_sections(state)[0].arg["research_context"]


def test_dispatch_without_plan_is_empty():
    assert dispatch_sections({"blog_plan": None}) == []


# ---- reducer -------------------------------------------------------------------------

def _section(i, words=120, valid=True):
    return WrittenSection(section_id=i, title=f"Title {i}", content=f"body {i}",
                          word_count=words, is_valid=valid)


@pytest.fixture
def no_disk(monkeypatch):
    monkeypatch.setattr("agents.reducer._save_blog", lambda blog, topic: None)


def test_reducer_orders_sections_by_id_regardless_of_arrival(no_disk):
    shuffled = [_section(2), _section(0), _section(1)]
    out = asyncio.run(reducer_node({"topic": "my topic", "written_sections": shuffled}))
    text = out["final_blog"].full_content
    assert text.index("Title 0") < text.index("Title 1") < text.index("Title 2")


def test_reducer_is_deterministic_across_permutations(no_disk):
    a = asyncio.run(reducer_node({"topic": "x", "written_sections": [_section(0), _section(1), _section(2)]}))
    b = asyncio.run(reducer_node({"topic": "x", "written_sections": [_section(2), _section(1), _section(0)]}))
    assert a["final_blog"].full_content == b["final_blog"].full_content


def test_reducer_excludes_invalid_sections_and_flags_not_ready(no_disk):
    out = asyncio.run(reducer_node({"topic": "x", "written_sections": [
        _section(0), _section(1), _section(2, valid=False)]}))
    blog = out["final_blog"]
    assert blog.sections_count == 2 and "Title 2" not in blog.full_content
    assert blog.is_production_ready is False


def test_reducer_production_ready_threshold(no_disk):
    ok = asyncio.run(reducer_node({"topic": "x", "written_sections": [
        _section(i, words=150) for i in range(3)]}))
    assert ok["final_blog"].is_production_ready is True


def test_reducer_empty_input(no_disk):
    out = asyncio.run(reducer_node({"topic": "x", "written_sections": []}))
    assert out["final_blog"].sections_count == 0 and out["final_blog"].is_production_ready is False


def test_reducer_appends_sources_when_retrieval_used(no_disk):
    retrieval = RetrievalResult(query="q", snippets=["s"], sources=["https://a.example", "https://b.example"])
    out = asyncio.run(reducer_node({"topic": "x", "written_sections": [_section(0)],
                                    "retrieval_results": retrieval}))
    text = out["final_blog"].full_content
    assert "## Sources" in text and text.index("https://a.example") < text.index("https://b.example")
