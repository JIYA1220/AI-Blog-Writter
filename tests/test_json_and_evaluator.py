import asyncio
import json

import pytest

from agents.evaluator import (EVAL_MAX_CHARS, build_evaluator_prompt, evaluator_node,
                              parse_evaluation)
from schemas.models import FinalBlog
from utils.json_utils import parse_llm_json

GOOD = {"score": 8, "reasoning": "clear", "suggestions": ["add example"], "is_pass": True}


def test_parse_plain_json():
    assert parse_llm_json(json.dumps(GOOD)) == GOOD


def test_parse_fenced_json():
    assert parse_llm_json("```json\n" + json.dumps(GOOD) + "\n```") == GOOD
    assert parse_llm_json("```\n" + json.dumps(GOOD) + "\n```") == GOOD


def test_parse_leading_json_label():
    assert parse_llm_json("json\n" + json.dumps(GOOD)) == GOOD


def test_parse_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("this is not json")


def test_parse_evaluation_builds_schema():
    ev = parse_evaluation(json.dumps(GOOD))
    assert (ev.score, ev.is_pass, ev.suggestions) == (8, True, ["add example"])


def test_parse_evaluation_defaults_when_fields_missing():
    ev = parse_evaluation('{"score": 5}')
    assert ev.score == 5 and ev.is_pass is False and ev.suggestions == []


def test_parse_evaluation_rejects_non_numeric_score():
    with pytest.raises(Exception):
        parse_evaluation('{"score": "high"}')


def test_prompt_truncates_content_to_cap():
    long = "a" * 3999 + "TAIL_MARKER" + "b" * 500
    prompt = build_evaluator_prompt("t", "aud", "tone", long)
    assert "TAIL_MARKER" not in prompt          # characters beyond the cap are invisible
    assert "a" * 3999 in prompt
    assert EVAL_MAX_CHARS == 4000


def test_prompt_without_cap_keeps_full_content():
    long = "a" * 5000 + "TAIL_MARKER"
    assert "TAIL_MARKER" in build_evaluator_prompt("t", "aud", "tone", long, max_chars=None)


def _state(content="# Title\n\nbody"):
    blog = FinalBlog(title="T", full_content=content, total_words=2, sections_count=1)
    return {"topic": "t", "target_audience": "a", "tone": "x", "final_blog": blog}


def test_evaluator_node_happy_path(monkeypatch, fake_llm):
    monkeypatch.setattr("agents.evaluator.get_llm", lambda temperature=0.1: fake_llm(lambda p: json.dumps(GOOD)))
    out = asyncio.run(evaluator_node(_state()))
    assert out["evaluation"].score == 8


def test_evaluator_node_invalid_json_records_error(monkeypatch, fake_llm):
    monkeypatch.setattr("agents.evaluator.get_llm", lambda temperature=0.1: fake_llm(lambda p: "nope"))
    out = asyncio.run(evaluator_node(_state()))
    assert out["evaluation"] is None
    assert out["errors"] and out["errors"][0].startswith("Evaluator error")


def test_evaluator_node_skips_when_no_blog():
    assert asyncio.run(evaluator_node({"topic": "t", "final_blog": None})) == {"evaluation": None}
