"""
Before/after impact of the evaluator full-text fix, computed ONLY from judge replies that already exist
in evaluation/results/<tag>/pointwise_raw.csv. No LLM calls are made and the live pipeline is not re-run.

"Before" is the old view (the judge saw the first 4000 characters: condition "truncated").
"After" is the fixed view (the judge sees the whole post: condition "full").
Both come from the same posts, the same judge and the same prompt, 3 runs each.
A post "passes" when the mean of its runs is >= 7 (the pipeline's pass threshold).

Usage: python -m evaluation.fix_impact [--tag groq-gpt-oss-120b]
Output: evaluation/results/fix_impact.json
"""

import argparse
import csv
import datetime
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.evaluator import LEGACY_EVAL_MAX_CHARS                      # noqa: E402
from evaluation.metrics import mean_bootstrap_ci, wilson_ci             # noqa: E402

EVAL_DIR = os.path.join(ROOT, "evaluation")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
PASS_THRESHOLD = 7


def prop(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "p": (k / n if n else None), "wilson95": [lo, hi]}


def fix_impact_stats(old: dict, new: dict, threshold: int = PASS_THRESHOLD) -> dict:
    """old/new: post_id -> list of scores for the same posts. Pass = mean score >= threshold."""
    ids = sorted(set(old) & set(new))
    n = len(ids)
    old_mean = {i: float(np.mean(old[i])) for i in ids}
    new_mean = {i: float(np.mean(new[i])) for i in ids}
    old_pass = {i: old_mean[i] >= threshold for i in ids}
    new_pass = {i: new_mean[i] >= threshold for i in ids}
    diffs = [new_mean[i] - old_mean[i] for i in ids]
    fail_to_pass = sum((not old_pass[i]) and new_pass[i] for i in ids)
    pass_to_fail = sum(old_pass[i] and (not new_pass[i]) for i in ids)
    reply_old = [s for i in ids for s in old[i]]
    reply_new = [s for i in ids for s in new[i]]
    return {
        "n_posts": n,
        "pass_rate_before_4000_chars": prop(sum(old_pass.values()), n),
        "pass_rate_after_full_text": prop(sum(new_pass.values()), n),
        "mean_score_before": float(np.mean(list(old_mean.values()))),
        "mean_score_after": float(np.mean(list(new_mean.values()))),
        "mean_score_change": float(np.mean(diffs)),
        "mean_score_change_ci95": list(mean_bootstrap_ci(diffs)),
        "verdict_changes": {"fail_to_pass": int(fail_to_pass), "pass_to_fail": int(pass_to_fail),
                            "unchanged": int(n - fail_to_pass - pass_to_fail)},
        "per_reply_pass_rate_before": prop(sum(s >= threshold for s in reply_old), len(reply_old)),
        "per_reply_pass_rate_after": prop(sum(s >= threshold for s in reply_new), len(reply_new)),
        "runs_per_post_before": [min(len(old[i]) for i in ids), max(len(old[i]) for i in ids)] if ids else None,
        "runs_per_post_after": [min(len(new[i]) for i in ids), max(len(new[i]) for i in ids)] if ids else None,
    }


def main(tag: str):
    path = os.path.join(RESULTS_DIR, tag, "pointwise_raw.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["parse_ok"] == "True"]
    old, new = defaultdict(list), defaultdict(list)
    for r in rows:
        (old if r["condition"] == "truncated" else new)[r["post_id"]].append(int(r["score"]))

    long_posts = set()
    for meta_path in glob.glob(os.path.join(EVAL_DIR, "posts", "*.json")):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta["chars"] > LEGACY_EVAL_MAX_CHARS:
            long_posts.add(meta["post_id"])

    stats = fix_impact_stats({p: old[p] for p in long_posts if p in old}, {p: new[p] for p in long_posts if p in new})
    out = {
        "what": "Before/after of the evaluator full-text fix on the same posts, the same judge and the same prompt.",
        "judge_tag": tag,
        "before": f"judge sees the first {LEGACY_EVAL_MAX_CHARS} characters (old evaluator behaviour)",
        "after": "judge sees the whole post (new default cap is far above these posts)",
        "pass_rule": f"post passes when the mean of its runs is >= {PASS_THRESHOLD}",
        "source": f"existing replies in evaluation/results/{tag}/pointwise_raw.csv; no LLM calls were made",
        "live_pipeline_rerun": False,
        "long_posts_in_study": len(long_posts),
        **stats,
        "date": datetime.date.today().isoformat(),
    }
    with open(os.path.join(RESULTS_DIR, "fix_impact.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    pb, pa = out["pass_rate_before_4000_chars"], out["pass_rate_after_full_text"]
    print(f"{out['n_posts']} long posts: pass rate {pb['p']:.1%} ({pb['k']}/{pb['n']}) before, "
          f"{pa['p']:.1%} ({pa['k']}/{pa['n']}) after; mean score change {out['mean_score_change']:+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="groq-gpt-oss-120b")
    main(ap.parse_args().tag)
