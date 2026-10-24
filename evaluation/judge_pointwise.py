"""
Study 1 data collection: the EXISTING pointwise evaluator prompt (agents/evaluator.py, unchanged).

For every post:
  - condition "truncated": the prompt exactly as the pipeline builds it (first 4000 characters), 3 runs
  - condition "full": only for posts longer than 4000 characters, the same prompt with the full text, 3 runs

Results go to evaluation/results/<tag>/pointwise_raw.csv and every call is logged verbatim to
evaluation/runs/<tag>/calls.jsonl, so judges are never mixed. Already-parsed calls are skipped (resumable).
Prints the number of LLM calls and exits unless --yes is given.

Usage:
  python -m evaluation.judge_pointwise --judge-model groq/llama-3.3-70b-versatile --tag groq-llama-3.3-70b --yes
"""

import argparse
import asyncio
import datetime
import os

from agents.evaluator import LEGACY_EVAL_MAX_CHARS, build_evaluator_prompt, parse_evaluation
from evaluation.judge_common import (QUOTA, RateLimiter, append_row, call_with_retry, estimate_tokens,
                                     is_quota_error, judge_dirs, judge_llm, load_posts, read_rows,
                                     record_invocation)

FIELDS = ["post_id", "condition", "run", "score", "is_pass", "parse_ok", "judge_model"]
TEMPERATURE = 0.1          # same as the pipeline evaluator node


def plan_tasks(posts: dict, runs: int, conditions: str = "both"):
    tasks = []
    for pid, p in posts.items():
        if conditions in ("both", "truncated"):
            tasks += [(pid, "truncated", r) for r in range(1, runs + 1)]
        if conditions in ("both", "full") and len(p["content"]) > LEGACY_EVAL_MAX_CHARS:
            tasks += [(pid, "full", r) for r in range(1, runs + 1)]
    # Run 1 for every post first, then run 2, then run 3: if a daily cap stops the job, every post
    # still has at least one score instead of a few posts having all three.
    return sorted(tasks, key=lambda t: t[2])


async def run_task(llm, posts, task, sem, judge_model, limiter, csv_path):
    pid, condition, run = task
    p = posts[pid]
    prompt = build_evaluator_prompt(p["topic"], p["audience"], p["tone"], p["content"],
                                    max_chars=LEGACY_EVAL_MAX_CHARS if condition == "truncated" else None, warn=False)
    async with sem:
        if QUOTA["exhausted"]:
            return                      # daily cap reached: leave undone, rerun later
        await limiter.wait(estimate_tokens(prompt))
        row = {"post_id": pid, "condition": condition, "run": run, "score": "", "is_pass": "",
               "parse_ok": False, "judge_model": judge_model}
        try:
            response = await call_with_retry(llm, prompt)
            ev = parse_evaluation(response.content)
            row.update(score=ev.score, is_pass=ev.is_pass, parse_ok=True)
        except Exception as e:
            if is_quota_error(e):
                print("  rate limit / quota reached, stopping; rerun later to resume", flush=True)
                return
            print(f"  {pid} {condition} run{run}: failed ({type(e).__name__}: {str(e)[:80]})", flush=True)
        append_row(csv_path, FIELDS, row)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", required=True, help="e.g. groq/llama-3.3-70b-versatile or ollama/llama3.2:3b")
    ap.add_argument("--tag", required=True, help="folder name for this judge under results/ and runs/")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conditions", choices=["both", "truncated", "full"], default="both")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between call starts (request caps)")
    ap.add_argument("--tpm", type=int, default=0, help="max estimated tokens per minute (token caps), 0 = off")
    ap.add_argument("--yes", action="store_true", help="confirm that the printed number of calls is OK")
    args = ap.parse_args()

    results_dir, runs_dir = judge_dirs(args.tag)
    csv_path = os.path.join(results_dir, "pointwise_raw.csv")
    posts = load_posts()
    done = {(r["post_id"], r["condition"], int(r["run"])) for r in read_rows(csv_path) if r["parse_ok"] == "True"}
    tasks = [t for t in plan_tasks(posts, args.runs, args.conditions) if t not in done]
    print(f"{len(posts)} posts; {len(tasks)} pointwise judge calls to make with {args.judge_model} "
          f"({len(done)} already done), conditions={args.conditions}")
    if not args.yes:
        print("Not running. Re-run with --yes to start.")
        return

    record_invocation(runs_dir, {"script": "judge_pointwise", "judge_model": args.judge_model,
                                 "temperature": TEMPERATURE, "runs": args.runs, "conditions": args.conditions,
                                 "eval_max_chars": LEGACY_EVAL_MAX_CHARS, "n_calls_planned": len(tasks),
                                 "min_interval": args.min_interval, "tpm": args.tpm, "concurrency": args.concurrency,
                                 "date": datetime.date.today().isoformat()})
    llm = judge_llm(args.judge_model, TEMPERATURE, os.path.join(runs_dir, "calls.jsonl"))
    sem = asyncio.Semaphore(args.concurrency)
    limiter = RateLimiter(args.min_interval, args.tpm)
    await asyncio.gather(*(run_task(llm, posts, t, sem, args.judge_model, limiter, csv_path) for t in tasks))
    print("done ->", csv_path)


if __name__ == "__main__":
    asyncio.run(main())
