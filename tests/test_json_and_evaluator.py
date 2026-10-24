import asyncio
import json

import pytest

from agents.evaluator import (DEFAULT_EVALUATOR_MAX_CHARS, LEGACY_EVAL_MAX_CHARS, build_evaluator_prompt,
                              evaluator_max_chars, evaluator_node,
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


def test_long_post_reaches_the_judge_in_full_by_default():
    """Regression for the audit finding: a post past the old 4000-character cap must not be cut."""
    long = "a" * 15900 + "TAIL_MARKER"            # about the length of the longest post the pipeline produced
    prompt = build_evaluator_prompt("t", "aud", "tone", long)
    assert "TAIL_MARKER" in prompt
    assert long in prompt


def test_default_cap_covers_posts_of_about_2500_words():
    assert DEFAULT_EVALUATOR_MAX_CHARS >= 2500 * 7
    assert LEGACY_EVAL_MAX_CHARS == 4000           # kept only so the audit can reproduce the old view


def test_post_over_the_cap_is_truncated_from_the_end_with_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("EVALUATOR_MAX_CHARS", "1000")
    long = "a" * 1000 + "TAIL_MARKER"
    with caplog.at_level("WARNING", logger="ai_blog_writer.evaluator"):
        prompt = build_evaluator_prompt("t", "aud", "tone", long)
    assert "TAIL_MARKER" not in prompt and "a" * 1000 in prompt
    warnings = [r.getMessage() for r in caplog.records]
    assert any("1011" in w and "1000" in w and "11 characters" in w for w in warnings)


def test_no_warning_when_the_post_fits(monkeypatch, caplog):
    monkeypatch.setenv("EVALUATOR_MAX_CHARS", "1000")
    with caplog.at_level("WARNING", logger="ai_blog_writer.evaluator"):
        build_evaluator_prompt("t", "aud", "tone", "a" * 1000)
    assert caplog.records == []


def test_cap_setting_is_honored(monkeypatch):
    monkeypatch.setenv("EVALUATOR_MAX_CHARS", "2000")
    assert evaluator_max_chars() == 2000
    text = "a" * 2500
    assert "a" * 2000 in build_evaluator_prompt("t", "aud", "tone", text, warn=False)
    assert "a" * 2001 not in build_evaluator_prompt("t", "aud", "tone", text, warn=False)
    monkeypatch.setenv("EVALUATOR_MAX_CHARS", "3000")
    assert "a" * 2500 in build_evaluator_prompt("t", "aud", "tone", text)


def test_explicit_cap_overrides_the_setting(monkeypatch):
    monkeypatch.setenv("EVALUATOR_MAX_CHARS", "5000")
    prompt = build_evaluator_prompt("t", "aud", "tone", "a" * 6000, max_chars=LEGACY_EVAL_MAX_CHARS, warn=False)
    assert "a" * 4000 in prompt and "a" * 4001 not in prompt


@pytest.mark.parametrize("bad", ["abc", "0", "-5"])
def test_invalid_cap_setting_falls_back_to_the_default(monkeypatch, bad):
    monkeypatch.setenv("EVALUATOR_MAX_CHARS", bad)
    assert evaluator_max_chars() == DEFAULT_EVALUATOR_MAX_CHARS


def test_evaluator_node_sends_the_whole_post(monkeypatch, fake_llm):
    llm = fake_llm(lambda p: json.dumps(GOOD))
    monkeypatch.setattr("agents.evaluator.get_llm", lambda temperature=0.1: llm)
    asyncio.run(evaluator_node(_state("x" * 9000 + "END_OF_POST")))
    assert "END_OF_POST" in llm.prompts[0]


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


def test_parse_tolerates_raw_control_characters_inside_strings():
    # small models often emit a literal newline inside a JSON string
    raw = '{"score": 7, "reasoning": "line one\nline two", "suggestions": [], "is_pass": true}'
    assert parse_llm_json(raw)["score"] == 7


def test_message_text_handles_string_and_content_blocks():
    from utils.json_utils import message_text
    assert message_text("plain") == "plain"
    blocks = [{"type": "text", "text": '{"score": 7}', "extras": {"signature": "xyz"}},
              {"type": "reasoning", "text": "hidden"}]
    assert message_text(blocks) == '{"score": 7}'
    assert message_text(None) == ""


def test_parse_json_from_content_blocks():
    blocks = [{"type": "text", "text": "```json\n" + json.dumps(GOOD) + "\n```"}]
    assert parse_llm_json(blocks) == GOOD


def test_section_writer_accepts_block_style_reply(monkeypatch):
    from agents.writer import section_writer_node
    from schemas.models import SectionPlan
    fake = type("L", (), {"ainvoke": staticmethod(lambda p: _async_reply([{"type": "text", "text": " hello world "}]))})
    monkeypatch.setattr("agents.writer.get_llm", lambda temperature=0.7: fake)
    state = {"section": SectionPlan(section_id=0, title="t", goal="g"), "topic": "x", "tone": "y",
             "research_context": ""}
    out = asyncio.run(section_writer_node(state))
    assert out["written_sections"][0].content == "hello world"
    assert out["written_sections"][0].is_valid


async def _async_reply(content):
    from types import SimpleNamespace
    return SimpleNamespace(content=content)
