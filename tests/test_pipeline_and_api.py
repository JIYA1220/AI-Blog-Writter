import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import server
from agents.retriever import retriever_node
from graph import ai_blog_writer_graph
from schemas.models import BlogEvaluation, FinalBlog

PLAN = {"topic": "T", "target_audience": "devs", "tone": "plain", "total_word_budget": 1500,
        "sections": [{"section_id": i, "title": f"Section {i}", "goal": "g", "constraints": ["c"],
                      "word_budget": 300, "requires_retrieval": False} for i in range(3)]}
EVAL = {"score": 8, "reasoning": "good", "suggestions": [], "is_pass": True}


def scripted(prompt: str) -> str:
    if "routing agent" in prompt:
        return json.dumps({"needs_retrieval": True, "reason": "recent facts"})
    if "content strategist" in prompt:
        return "```json\n" + json.dumps(PLAN) + "\n```"
    if "expert blog writer" in prompt:
        return "word " * 120
    if "senior editor" in prompt:
        return json.dumps(EVAL)
    raise AssertionError("unexpected prompt")


@pytest.fixture
def offline_pipeline(monkeypatch, fake_llm):
    llms = []

    def factory(temperature=0.7):
        llm = fake_llm(scripted)
        llms.append(llm)
        return llm

    for module in ("router", "planner", "writer", "evaluator"):
        monkeypatch.setattr(f"agents.{module}.get_llm", factory)
    monkeypatch.setattr("agents.reducer._save_blog", lambda blog, topic: None)
    return llms


def _inputs():
    return {"topic": "Topic", "target_audience": "devs", "tone": "plain", "needs_retrieval": False,
            "retrieval_results": None, "blog_plan": None, "written_sections": [],
            "final_blog": None, "evaluation": None, "errors": []}


def test_full_graph_without_search_key_skips_retrieval(offline_pipeline, monkeypatch):
    def boom():
        raise AssertionError("retriever must not run without a Tavily key")
    monkeypatch.setattr("agents.retriever._get_tavily_client", boom)

    final = asyncio.run(ai_blog_writer_graph.ainvoke(_inputs()))
    assert final["needs_retrieval"] is False
    assert final["final_blog"].sections_count == 3
    assert final["evaluation"].score == 8
    assert final["final_blog"].full_content.index("Section 0") < final["final_blog"].full_content.index("Section 2")


def test_full_graph_with_search_runs_retriever_and_adds_sources(offline_pipeline, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    fake_tavily = SimpleNamespace(search=lambda **kw: {
        "answer": "synthesised answer",
        "results": [{"content": "fact A", "url": "https://a.example"},
                    {"content": "fact B", "url": "https://b.example"}]})
    monkeypatch.setattr("agents.retriever._get_tavily_client", lambda: fake_tavily)

    final = asyncio.run(ai_blog_writer_graph.ainvoke(_inputs()))
    assert final["needs_retrieval"] is True
    assert final["retrieval_results"].is_grounded is True
    assert "https://a.example" in final["final_blog"].full_content


def test_retriever_without_client_returns_ungrounded(monkeypatch):
    monkeypatch.setattr("agents.retriever._get_tavily_client", lambda: None)
    out = asyncio.run(retriever_node({"topic": "t"}))
    assert out["retrieval_results"].is_grounded is False and out["retrieval_results"].snippets == []


def test_retriever_source_order_is_stable(monkeypatch):
    results = [{"content": f"c{i}", "url": f"https://{i}.example"} for i in range(5)]
    monkeypatch.setattr("agents.retriever._get_tavily_client",
                        lambda: SimpleNamespace(search=lambda **kw: {"results": results}))
    out = asyncio.run(retriever_node({"topic": "t"}))
    assert out["retrieval_results"].sources == [r["url"] for r in results]


# ---- FastAPI -------------------------------------------------------------------------

class FakeGraph:
    def __init__(self, state=None, error=None):
        self.state, self.error, self.seen = state, error, None

    async def ainvoke(self, inputs, config=None):
        self.seen = inputs
        if self.error:
            raise self.error
        return self.state


def _blog_state():
    return {"final_blog": FinalBlog(title="Hello", full_content="# Hello\n\nbody", total_words=2,
                                    sections_count=1),
            "evaluation": BlogEvaluation(score=9, reasoning="nice")}


def test_root_health():
    assert TestClient(server.app).get("/").json()["status"] == "online"


def test_generate_returns_blog_and_score(monkeypatch):
    graph = FakeGraph(_blog_state())
    monkeypatch.setattr(server, "ai_blog_writer_graph", graph)
    r = TestClient(server.app).post("/generate", json={"topic": "Agents", "tone": "casual"})
    assert r.status_code == 200
    body = r.json()
    assert (body["title"], body["score"], body["word_count"]) == ("Hello", 9, 2)
    assert graph.seen["topic"] == "Agents" and graph.seen["tone"] == "casual"
    assert graph.seen["target_audience"] == "general readers"      # default


def test_generate_without_evaluation_returns_null_score(monkeypatch):
    state = _blog_state()
    state["evaluation"] = None
    monkeypatch.setattr(server, "ai_blog_writer_graph", FakeGraph(state))
    body = TestClient(server.app).post("/generate", json={"topic": "x"}).json()
    assert body["score"] is None and body["evaluation"] is None


def test_generate_missing_topic_is_422():
    assert TestClient(server.app).post("/generate", json={}).status_code == 422


def test_generate_reports_500_when_no_blog_produced(monkeypatch):
    monkeypatch.setattr(server, "ai_blog_writer_graph", FakeGraph({"final_blog": None}))
    assert TestClient(server.app).post("/generate", json={"topic": "x"}).status_code == 500


def test_generate_reports_500_on_graph_error(monkeypatch):
    monkeypatch.setattr(server, "ai_blog_writer_graph", FakeGraph(error=RuntimeError("boom")))
    r = TestClient(server.app).post("/generate", json={"topic": "x"})
    assert r.status_code == 500 and "boom" in r.json()["detail"]
