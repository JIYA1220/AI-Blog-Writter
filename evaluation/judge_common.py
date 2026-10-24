"""
Shared pieces for the judge scripts: loading posts, logged judge client, resumable CSV rows,
rate limiting and rate-limit backoff.

Each judge writes into its own folder, chosen with --tag:
  evaluation/results/<tag>/   CSV results and analysis.json
  evaluation/runs/<tag>/      every prompt and raw reply, verbatim (calls.jsonl)
so results from different judges are never mixed.
"""

import asyncio
import csv
import glob
import json
import os
import re
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.llm_log import LoggedLLM  # noqa: E402
from utils.llm import get_llm              # noqa: E402

EVAL_DIR = os.path.join(ROOT, "evaluation")
POSTS_DIR = os.path.join(EVAL_DIR, "posts")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")


def load_topics():
    with open(os.path.join(EVAL_DIR, "topics.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_posts() -> dict:
    """post_id -> {meta..., 'content': markdown}. Only posts that have both .md and .json."""
    posts = {}
    for meta_path in sorted(glob.glob(os.path.join(POSTS_DIR, "*.json"))):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        md_path = meta_path[:-5] + ".md"
        if os.path.exists(md_path):
            with open(md_path, encoding="utf-8") as f:
                meta["content"] = f.read()
            posts[meta["post_id"]] = meta
    return posts


def judge_dirs(tag: str):
    """(results_dir, runs_dir) for a judge tag, created if missing."""
    results, runs = os.path.join(RESULTS_DIR, tag), os.path.join(RUNS_DIR, tag)
    os.makedirs(results, exist_ok=True)
    os.makedirs(runs, exist_ok=True)
    return results, runs


def record_invocation(runs_dir: str, info: dict):
    """Append one invocation (model, settings, call count, date) to runs/<tag>/run_meta.json."""
    path = os.path.join(runs_dir, "run_meta.json")
    meta = {"invocations": []}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    meta["invocations"].append(info)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


JUDGE_NUM_CTX = 8192     # Ollama's default window (2048) would silently cut long posts
JUDGE_MAX_TOKENS = 2000        # reply cap; reasoning models spend part of it on hidden reasoning tokens
ESTIMATED_REPLY_TOKENS = 700   # typical reply size, used only for token-per-minute planning


def judge_llm(model: str, temperature: float, log_path: str):
    ctx = JUDGE_NUM_CTX if model.startswith("ollama/") else None
    is_groq = model.startswith("groq/")
    inner = get_llm(temperature=temperature, model_name=model, num_ctx=ctx, json_mode=is_groq,
                    max_tokens=JUDGE_MAX_TOKENS if is_groq else None)
    return LoggedLLM(inner, log_path, model=model, temperature=temperature, tag="judge")


def read_rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: str, fieldnames: list, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def estimate_tokens(prompt: str, reply_tokens: int = ESTIMATED_REPLY_TOKENS) -> int:
    """Rough token count for rate limiting: about 3.2 characters per token, plus a typical reply size."""
    return int(len(prompt) / 3.2) + reply_tokens


class RateLimiter:
    """
    Spaces call starts at least `min_interval` seconds apart (request-per-minute caps) and keeps the
    estimated tokens sent in any 60-second window under `tpm` (tokens-per-minute caps).
    A value of 0 disables that limit.
    """

    def __init__(self, min_interval: float = 0.0, tpm: int = 0, clock=time.monotonic, sleeper=asyncio.sleep):
        self.min_interval, self.tpm = min_interval, tpm
        self._clock, self._sleep = clock, sleeper
        self._lock, self._next, self._window = asyncio.Lock(), 0.0, []

    async def wait(self, tokens: int = 0):
        if self.min_interval <= 0 and self.tpm <= 0:
            return
        async with self._lock:
            delay = self._next - self._clock()
            if delay > 0:
                await self._sleep(delay)
            if self.tpm > 0:
                budget = min(tokens, self.tpm)       # a single oversized prompt must still be allowed through
                while True:
                    now = self._clock()
                    self._window = [(t, n) for t, n in self._window if now - t < 60.0]
                    if not self._window or sum(n for _, n in self._window) + budget <= self.tpm:
                        break
                    await self._sleep(max(0.05, 60.0 - (now - self._window[0][0])))
                self._window.append((self._clock(), budget))
            self._next = self._clock() + self.min_interval


QUOTA = {"exhausted": False}


def is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate limit" in msg


def is_daily_limit(e: Exception) -> bool:
    """Per-day caps (tokens or requests) will not clear within minutes, so retrying is pointless."""
    msg = str(e).lower()
    return any(k in msg for k in ("per day", "perday", "(tpd)", "(rpd)", "daily"))


_RETRY_HINT = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+(?:\.\d+)?)(ms|s))?")


def retry_after_seconds(e: Exception):
    """Seconds the server asked us to wait: Retry-After header, else 'try again in 1m7.5s' in the message."""
    headers = getattr(getattr(e, "response", None), "headers", None)
    if headers:
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    m = _RETRY_HINT.search(str(e))
    if not m or not any(m.groups()[:3]):
        return None
    hours, minutes, seconds, unit = m.groups()
    total = int(hours or 0) * 3600 + int(minutes or 0) * 60
    if seconds:
        total += float(seconds) / (1000 if unit == "ms" else 1)
    return total


async def call_with_retry(llm, prompt, retries: int = 6, base_wait: float = 5.0, max_wait: float = 120.0,
                          sleeper=asyncio.sleep):
    """
    ainvoke with exponential backoff (base_wait * 2**attempt, capped at max_wait) on rate-limit errors,
    never shorter than the server's own retry hint (itself capped at 2 * max_wait). A daily cap, or running
    out of retries, sets QUOTA["exhausted"] so the rest of the run stops cleanly and can be resumed later.
    """
    for attempt in range(retries + 1):
        try:
            return await llm.ainvoke(prompt)
        except Exception as e:
            if not is_quota_error(e):
                raise
            if is_daily_limit(e) or attempt == retries:
                QUOTA["exhausted"] = True
                raise
            wait = max(retry_after_seconds(e) or 0.0, min(max_wait, base_wait * 2 ** attempt))
            await sleeper(min(wait, 2 * max_wait))
