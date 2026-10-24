import asyncio

import pytest

from evaluation import judge_common as jc


def test_judge_llm_uses_json_mode_only_for_groq(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    groq = jc.judge_llm("groq/llama-3.3-70b-versatile", 0.1, str(tmp_path / "c.jsonl")).inner
    assert groq._get_request_payload("x")["response_format"] == {"type": "json_object"}
    ollama = jc.judge_llm("ollama/llama3.2:3b", 0.1, str(tmp_path / "c.jsonl")).inner
    assert ollama.num_ctx == jc.JUDGE_NUM_CTX


def test_key_is_not_written_to_call_logs(monkeypatch, tmp_path):
    """The logger records prompts and replies only, never client settings or headers."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value_for_test")
    log = tmp_path / "calls.jsonl"

    class Stub:
        async def ainvoke(self, prompt):
            from types import SimpleNamespace
            return SimpleNamespace(content='{"ok": true}')

    from evaluation.llm_log import LoggedLLM
    asyncio.run(LoggedLLM(Stub(), str(log), "groq/x", 0.1).ainvoke("prompt text"))
    assert "gsk_secret_value_for_test" not in log.read_text(encoding="utf-8")


# ---- rate-limit helpers ------------------------------------------------------------------


def test_retry_after_parsing():
    assert jc.retry_after_seconds(Exception("Please try again in 7m12.5s.")) == pytest.approx(432.5)
    assert jc.retry_after_seconds(Exception("try again in 850ms")) == pytest.approx(0.85)
    assert jc.retry_after_seconds(Exception("try again in 2.5s")) == pytest.approx(2.5)
    assert jc.retry_after_seconds(Exception("try again in 1h2m3s")) == pytest.approx(3723)
    assert jc.retry_after_seconds(Exception("no hint here")) is None


def test_retry_after_header_wins_over_message():
    from types import SimpleNamespace
    err = Exception("429 try again in 30s")
    err.response = SimpleNamespace(headers={"retry-after": "4"})
    assert jc.retry_after_seconds(err) == 4.0


def test_error_classification():
    assert jc.is_quota_error(Exception("Error code: 429 - rate limit reached"))
    assert not jc.is_quota_error(Exception("connection reset"))
    assert jc.is_daily_limit(Exception("Rate limit reached ... on tokens per day (TPD): Limit 100000"))
    assert not jc.is_daily_limit(Exception("Rate limit reached ... on tokens per minute (TPM): Limit 12000"))


class FlakyLLM:
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "reply"


def _run_retry(llm, **kw):
    waits = []

    async def fake_sleep(s):
        waits.append(s)

    jc.QUOTA["exhausted"] = False
    result = asyncio.run(jc.call_with_retry(llm, "p", sleeper=fake_sleep, **kw))
    return result, waits


def test_backoff_is_exponential_and_then_succeeds():
    llm = FlakyLLM([Exception("429 rate limit")] * 3)
    result, waits = _run_retry(llm, base_wait=5.0, max_wait=120.0)
    assert result == "reply" and llm.calls == 4
    assert waits == [5.0, 10.0, 20.0]
    assert jc.QUOTA["exhausted"] is False


def test_backoff_never_shorter_than_server_hint():
    llm = FlakyLLM([Exception("429 rate limit, try again in 40s")])
    _, waits = _run_retry(llm, base_wait=5.0)
    assert waits == [40.0]


def test_daily_limit_stops_without_retrying():
    llm = FlakyLLM([Exception("429 rate limit on tokens per day (TPD)")] * 5)
    jc.QUOTA["exhausted"] = False
    waits = []

    async def fake_sleep(s):
        waits.append(s)

    with pytest.raises(Exception, match="per day"):
        asyncio.run(jc.call_with_retry(llm, "p", sleeper=fake_sleep))
    assert llm.calls == 1 and waits == [] and jc.QUOTA["exhausted"] is True
    jc.QUOTA["exhausted"] = False


def test_other_errors_are_not_retried():
    llm = FlakyLLM([ValueError("bad request")])
    with pytest.raises(ValueError):
        _run_retry(llm)
    assert llm.calls == 1


def test_retries_exhausted_marks_quota():
    llm = FlakyLLM([Exception("429 rate limit")] * 10)
    with pytest.raises(Exception):
        _run_retry(llm, retries=2, base_wait=1.0)
    assert llm.calls == 3 and jc.QUOTA["exhausted"] is True
    jc.QUOTA["exhausted"] = False


# ---- token / request limiter ---------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def now(self):
        return self.t

    async def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_limiter_spaces_calls_by_min_interval():
    clk = FakeClock()
    limiter = jc.RateLimiter(min_interval=2.0, clock=clk.now, sleeper=clk.sleep)

    async def go():
        for _ in range(3):
            await limiter.wait(0)

    asyncio.run(go())
    assert clk.t == pytest.approx(4.0)


def test_limiter_keeps_tokens_per_minute_under_cap():
    clk = FakeClock()
    limiter = jc.RateLimiter(tpm=10_000, clock=clk.now, sleeper=clk.sleep)
    starts = []

    async def go():
        for _ in range(4):
            await limiter.wait(4_000)
            starts.append(clk.t)

    asyncio.run(go())
    # two 4k calls fit in the first minute; the third must wait for the first to leave the window
    assert starts[0] == 0.0 and starts[1] == 0.0
    assert starts[2] >= 60.0 and starts[3] >= 60.0


def test_limiter_lets_one_oversized_prompt_through():
    clk = FakeClock()
    limiter = jc.RateLimiter(tpm=1_000, clock=clk.now, sleeper=clk.sleep)
    asyncio.run(limiter.wait(50_000))
    assert clk.t == 0.0


def test_limiter_disabled_by_default():
    clk = FakeClock()
    asyncio.run(jc.RateLimiter(clock=clk.now, sleeper=clk.sleep).wait(10**9))
    assert clk.t == 0.0 and clk.sleeps == []


def test_plan_tasks_orders_by_run_so_every_post_gets_a_score_first():
    from evaluation.judge_pointwise import plan_tasks
    posts = {"a": {"content": "x" * 5000}, "b": {"content": "x" * 5000}}
    tasks = plan_tasks(posts, 3, "full")
    assert [t[2] for t in tasks] == [1, 1, 2, 2, 3, 3]
    assert {t[0] for t in tasks[:2]} == {"a", "b"}
