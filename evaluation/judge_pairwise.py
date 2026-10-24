"""
Study 2 data collection: NEW pairwise prompt. For every topic, variant A is compared with
variant B in BOTH orders (A shown first, then B shown first).

In the prompt the two slots are called "Post A" and "Post B". In the CSV the slots are
called "first" and "second" and the experiment variants A / B are kept apart, to avoid
confusing the slot label with the variant name.

Results go to evaluation/results/<tag>/pairwise_raw.csv and every call is logged verbatim to
evaluation/runs/<tag>/calls.jsonl. Already-parsed calls are skipped (resumable).
Prints the number of LLM calls and exits unless --yes is given.

Usage:
  python -m evaluation.judge_pairwise --judge-model groq/llama-3.3-70b-versatile --tag groq-llama-3.3-70b --yes
"""

import argparse
import asyncio
import datetime
import os

from evaluation.judge_common import (QUOTA, RateLimiter, append_row, call_with_retry, estimate_tokens,
                                     is_quota_error, judge_dirs, judge_llm, load_posts, read_rows,
                                     record_invocation)
from utils.json_utils import parse_llm_json

FIELDS = ["topic_id", "order", "slot_winner", "variant_winner", "reason", "parse_ok", "judge_model"]
TEMPERATURE = 0.1

PAIRWISE_PROMPT = """You are a senior editor comparing two blog posts written on the same topic.

Topic: {topic}
Target Audience: {audience}
Requested Tone: {tone}

===== Post A =====
{post_a}

===== Post B =====
{post_b}

Decide which post is better overall, judging:
1. Relevance to topic and audience.
2. Adherence to the requested tone.
3. Logical flow and structure.
4. Depth of information.

Respond ONLY with valid JSON:
{{
  "winner": "A" or "B" or "tie",
  "reason": "one short sentence"
}}
"""


def parse_pairwise(raw: str) -> tuple:
    """Return (slot_winner, reason) with slot_winner in {'first','second','tie'}."""
    data = parse_llm_json(raw)
    w = str(data.get("winner", "")).strip().lower()
    slot = {"a": "first", "b": "second", "tie": "tie"}.get(w)
    if slot is None:
        raise ValueError(f"unexpected winner value: {data.get('winner')!r}")
    return slot, str(data.get("reason", ""))


def to_variant(order: str, slot: str) -> str:
    """order 'AB' = variant A shown first. Map a slot verdict back to a variant verdict."""
    if slot == "tie":
        return "tie"
    shown = {"first": order[0], "second": order[1]}
    return shown[slot]


async def run_task(llm, posts, task, sem, judge_model, limiter, csv_path):
    topic_id, order = task
    first, second = posts[f"{topic_id}_{order[0]}"], posts[f"{topic_id}_{order[1]}"]
    prompt = PAIRWISE_PROMPT.format(topic=first["topic"], audience=first["audience"], tone=first["tone"],
                                    post_a=first["content"], post_b=second["content"])
    async with sem:
        if QUOTA["exhausted"]:
            return                      # daily cap reached: leave undone, rerun later
        await limiter.wait(estimate_tokens(prompt))
        row = {"topic_id": topic_id, "order": order, "slot_winner": "", "variant_winner": "",
               "reason": "", "parse_ok": False, "judge_model": judge_model}
        try:
            response = await call_with_retry(llm, prompt)
            slot, reason = parse_pairwise(response.content)
            row.update(slot_winner=slot, variant_winner=to_variant(order, slot), reason=reason, parse_ok=True)
        except Exception as e:
            if is_quota_error(e):
                print("  rate limit / quota reached, stopping; rerun later to resume", flush=True)
                return
            print(f"  {topic_id} {order}: failed ({type(e).__name__}: {str(e)[:80]})", flush=True)
        append_row(csv_path, FIELDS, row)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--tag", required=True, help="folder name for this judge under results/ and runs/")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between call starts (request caps)")
    ap.add_argument("--tpm", type=int, default=0, help="max estimated tokens per minute (token caps), 0 = off")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    results_dir, runs_dir = judge_dirs(args.tag)
    csv_path = os.path.join(results_dir, "pairwise_raw.csv")
    posts = load_posts()
    topic_ids = sorted({p["topic_id"] for p in posts.values()
                        if f"{p['topic_id']}_A" in posts and f"{p['topic_id']}_B" in posts})
    done = {(r["topic_id"], r["order"]) for r in read_rows(csv_path) if r["parse_ok"] == "True"}
    tasks = [(t, o) for t in topic_ids for o in ("AB", "BA") if (t, o) not in done]
    print(f"{len(topic_ids)} topics with both variants; {len(tasks)} pairwise judge calls to make with "
          f"{args.judge_model} ({len(done)} already done)")
    if not args.yes:
        print("Not running. Re-run with --yes to start.")
        return

    record_invocation(runs_dir, {"script": "judge_pairwise", "judge_model": args.judge_model,
                                 "temperature": TEMPERATURE, "n_calls_planned": len(tasks),
                                 "min_interval": args.min_interval, "tpm": args.tpm, "concurrency": args.concurrency,
                                 "prompt": PAIRWISE_PROMPT, "date": datetime.date.today().isoformat()})
    llm = judge_llm(args.judge_model, TEMPERATURE, os.path.join(runs_dir, "calls.jsonl"))
    sem = asyncio.Semaphore(args.concurrency)
    limiter = RateLimiter(args.min_interval, args.tpm)
    await asyncio.gather(*(run_task(llm, posts, t, sem, args.judge_model, limiter, csv_path) for t in tasks))
    print("done ->", csv_path)


if __name__ == "__main__":
    asyncio.run(main())
