"""
Analysis of the judge audit. Reads evaluation/results/<tag>/*.csv,
writes evaluation/results/<tag>/analysis.json and prints a summary.

Agreement with a human reader was not measured and is out of scope; nothing here measures accuracy.

Usage: python -m evaluation.analyze [--tag groq-gpt-oss-120b]
"""

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.metrics import mean_bootstrap_ci, wilson_ci  # noqa: E402

EVAL_DIR = os.path.join(ROOT, "evaluation")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
NOT_MEASURED = "not yet measured"
PASS_THRESHOLD = 7


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prop(k, n):
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "p": (k / n if n else None), "wilson95": [lo, hi]}


# ---------------------------------------------------------------- pure functions (unit-tested)

def retest_stats(scores_by_post: dict) -> dict:
    """scores_by_post: post_id -> list of scores from repeated runs (>= 2)."""
    mads, all_same, pair_same, pair_n = [], 0, 0, 0
    for scores in scores_by_post.values():
        diffs = [abs(a - b) for a, b in combinations(scores, 2)]
        mads.append(float(np.mean(diffs)))
        all_same += int(len(set(scores)) == 1)
        pair_same += sum(d == 0 for d in diffs)
        pair_n += len(diffs)
    n = len(scores_by_post)
    return {"n_posts": n, "mean_abs_diff": float(np.mean(mads)),
            "mean_abs_diff_ci95": list(mean_bootstrap_ci(mads)),
            "share_posts_all_runs_identical": prop(all_same, n),
            "share_run_pairs_identical": prop(pair_same, pair_n)}


def truncation_stats(trunc: dict, full: dict) -> dict:
    """trunc/full: post_id -> list of scores; only posts present in both."""
    ids = sorted(set(trunc) & set(full))
    if not ids:
        return {"n_long_posts": 0}
    diffs = [float(np.mean(full[i]) - np.mean(trunc[i])) for i in ids]
    changed = sum(abs(d) >= 1.0 for d in diffs)
    pass_flips = sum((np.mean(full[i]) >= PASS_THRESHOLD) != (np.mean(trunc[i]) >= PASS_THRESHOLD) for i in ids)
    return {"n_long_posts": len(ids),
            "mean_diff_full_minus_truncated": float(np.mean(diffs)),
            "mean_diff_ci95": list(mean_bootstrap_ci(diffs)),
            "mean_abs_diff": float(np.mean(np.abs(diffs))),
            "share_changed_by_1_or_more_points": prop(int(changed), len(ids)),
            "share_pass_fail_flipped": prop(int(pass_flips), len(ids))}


def pairwise_stats(verdicts: dict, slot_verdicts: dict) -> dict:
    """
    verdicts:      topic -> {'AB': variant verdict, 'BA': variant verdict}
    slot_verdicts: topic -> {'AB': 'first'|'second'|'tie', 'BA': ...}
    """
    topics = sorted(t for t in verdicts if set(verdicts[t]) == {"AB", "BA"})
    n = len(topics)
    inconsistent = [t for t in topics if verdicts[t]["AB"] != verdicts[t]["BA"]]
    both_first = [t for t in inconsistent
                  if slot_verdicts[t]["AB"] == "first" and slot_verdicts[t]["BA"] == "first"]
    both_second = [t for t in inconsistent
                   if slot_verdicts[t]["AB"] == "second" and slot_verdicts[t]["BA"] == "second"]
    all_slots = [slot_verdicts[t][o] for t in topics for o in ("AB", "BA")]
    decisive = [s for s in all_slots if s != "tie"]
    return {
        "n_topics": n,
        "inconsistency_rate": prop(len(inconsistent), n),
        "among_inconsistent": {
            "n": len(inconsistent),
            "first_shown_won_in_both_orders": prop(len(both_first), len(inconsistent)),
            "second_shown_won_in_both_orders": prop(len(both_second), len(inconsistent)),
            "involved_a_tie_in_one_order": prop(len(inconsistent) - len(both_first) - len(both_second),
                                                len(inconsistent))},
        "first_shown_share_of_all_decisive_verdicts": prop(sum(s == "first" for s in decisive), len(decisive)),
        "verdict_counts_by_order": {o: dict(Counter(verdicts[t][o] for t in topics)) for o in ("AB", "BA")},
    }


# ---------------------------------------------------------------- main

def main(tag: str):
    results_dir = os.path.join(RESULTS_DIR, tag)
    out = {}
    posts_meta = [json.load(open(f, encoding="utf-8")) for f in
                  sorted(glob.glob(os.path.join(EVAL_DIR, "posts", "*.json")))]
    n_posts = len(posts_meta)
    n_long = sum(p["chars"] > 4000 for p in posts_meta)
    topics_both = len({p["topic_id"] for p in posts_meta}
                      if all(f"{t}_A" in {q["post_id"] for q in posts_meta} and
                             f"{t}_B" in {q["post_id"] for q in posts_meta}
                             for t in {p["topic_id"] for p in posts_meta}) else set())
    pw = _read(os.path.join(results_dir, "pointwise_raw.csv"))
    if pw is None:
        out["study1_pointwise"] = NOT_MEASURED + " (run evaluation.judge_pointwise)"
    else:
        ok = [r for r in pw if r["parse_ok"] == "True"]
        out["study1_calls"] = {"rows": len(pw), "parsed_ok": len(ok), "failed_to_parse": len(pw) - len(ok)}
        out["score_distribution"] = {
            cond: dict(sorted(Counter(int(r["score"]) for r in ok if r["condition"] == cond).items()))
            for cond in ("truncated", "full")}
        all_scores = [int(r["score"]) for r in ok]
        if all_scores:
            mode, mode_n = Counter(all_scores).most_common(1)[0]
            out["score_distribution"]["modal_score"] = mode
            out["score_distribution"]["modal_score_share"] = prop(mode_n, len(all_scores))
        out["is_pass_vs_score"] = {
            "parsed_replies": len(ok),
            "is_pass_disagrees_with_score_ge_7": sum((r["is_pass"] == "True") != (int(r["score"]) >= PASS_THRESHOLD)
                                                     for r in ok)}
        done_keys = {(r["post_id"], r["condition"], r["run"]) for r in ok}
        failed_rows = [r for r in pw if r["parse_ok"] != "True"]
        out["failed_attempts_later_succeeded"] = sum((r["post_id"], r["condition"], r["run"]) in done_keys
                                                     for r in failed_rows)
        out["failed_replies_by_condition"] = dict(Counter(r["condition"] for r in pw if r["parse_ok"] != "True"))
        trunc, full = defaultdict(list), defaultdict(list)
        for r in ok:
            (trunc if r["condition"] == "truncated" else full)[r["post_id"]].append(int(r["score"]))
        complete = {p: s for p, s in trunc.items() if len(s) >= 2}
        out["test_retest"] = retest_stats(complete) if complete else NOT_MEASURED
        out["truncation"] = truncation_stats(
            {p: s for p, s in trunc.items() if len(s) >= 1}, {p: s for p, s in full.items() if len(s) >= 1})
        if out["truncation"].get("n_long_posts"):
            both = sorted(set(trunc) & set(full))
            out["truncation"]["runs_per_post_truncated"] = [min(len(trunc[i]) for i in both),
                                                            max(len(trunc[i]) for i in both)]
            out["truncation"]["runs_per_post_full"] = [min(len(full[i]) for i in both),
                                                       max(len(full[i]) for i in both)]

    pr = _read(os.path.join(results_dir, "pairwise_raw.csv"))
    verdicts, slots = defaultdict(dict), defaultdict(dict)
    if pr is None:
        out["position_bias"] = NOT_MEASURED + " (run evaluation.judge_pairwise)"
    else:
        for r in pr:
            if r["parse_ok"] == "True":
                verdicts[r["topic_id"]][r["order"]] = r["variant_winner"]
                slots[r["topic_id"]][r["order"]] = r["slot_winner"]
        distinct_ok = {(t, o) for t in verdicts for o in verdicts[t]}
        out["pairwise_parse"] = {"topic_orders_attempted": len({(r["topic_id"], r["order"]) for r in pr}),
                                 "topic_orders_parsed": len(distinct_ok),
                                 "log_rows_including_retries": len(pr)}
        out["position_bias"] = pairwise_stats(verdicts, slots)

    def _parsed(rows, key):
        return len({key(r) for r in rows if r["parse_ok"] == "True"}) if rows else 0

    out["completeness"] = {
        "pointwise_truncated": {"parsed": _parsed([r for r in (pw or []) if r["condition"] == "truncated"],
                                                  lambda r: (r["post_id"], r["run"])), "planned": 3 * n_posts},
        "pointwise_full": {"parsed": _parsed([r for r in (pw or []) if r["condition"] == "full"],
                                             lambda r: (r["post_id"], r["run"])), "planned": 3 * n_long},
        "pairwise": {"parsed": _parsed(pr or [], lambda r: (r["topic_id"], r["order"])),
                     "planned": 2 * topics_both},
    }
    out["completeness"]["complete"] = all(v["parsed"] >= v["planned"] for v in out["completeness"].values())

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="judge folder under evaluation/results/ (default: every judge folder)")
    a = ap.parse_args()
    tags = [a.tag] if a.tag else sorted(d for d in os.listdir(RESULTS_DIR)
                                        if os.path.isdir(os.path.join(RESULTS_DIR, d)))
    for t in tags:
        print(f"=== analysis for judge folder: {t}")
        main(t)
