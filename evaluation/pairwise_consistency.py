"""
Pairwise consistency of a judge, from replies that already exist in evaluation/results/<tag>/pairwise_raw.csv.
No LLM calls are made. This measures CONSISTENCY ONLY. Nothing here says which verdict is correct, because
agreement with a human reader was not measured.

Variant A is the short post and variant B the long post, so "B won" is confounded with length.

Usage: python -m evaluation.pairwise_consistency [--tag groq-gpt-oss-120b]
Output: evaluation/results/<tag>/pairwise_consistency.json
"""

import argparse
import csv
import datetime
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.metrics import binomial_test_two_sided, wilson_ci  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def prop(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "p": (k / n if n else None), "wilson95": [lo, hi]}


def consistency_stats(variant: dict, slot: dict) -> dict:
    """
    variant: topic -> {'AB': 'A'|'B'|'tie', 'BA': ...}   winner as a variant, per presentation order
    slot:    topic -> {'AB': 'first'|'second'|'tie', 'BA': ...}   winner as a position
    """
    topics = sorted(t for t in variant if set(variant[t]) == {"AB", "BA"})
    n = len(topics)
    consistent = [t for t in topics if variant[t]["AB"] == variant[t]["BA"]]
    flipped = [t for t in topics if variant[t]["AB"] != variant[t]["BA"]]

    winners = Counter(variant[t]["AB"] for t in consistent)
    first_both = [t for t in flipped if slot[t]["AB"] == "first" and slot[t]["BA"] == "first"]
    second_both = [t for t in flipped if slot[t]["AB"] == "second" and slot[t]["BA"] == "second"]
    one_tie = len(flipped) - len(first_both) - len(second_both)

    return {
        "n_topics": n,
        "same_winner_in_both_orders": prop(len(consistent), n),
        "among_consistent_topics": {
            "n": len(consistent),
            "winner_counts": {"A_short": winners.get("A", 0), "B_long": winners.get("B", 0), "tie": winners.get("tie", 0)},
            "B_long_won": prop(winners.get("B", 0), len(consistent)),
            "note": "Variant B is about twice as long as variant A, so 'B won' is confounded with length.",
        },
        "among_flipped_topics": {
            "n": len(flipped),
            "first_shown_won_in_both_orders": len(first_both),
            "second_shown_won_in_both_orders": len(second_both),
            "tie_in_one_order": one_tie,
            "binomial_test": {
                "question": "Among flipped topics, did the first-shown post win more often than chance (50%)?",
                "k_first_shown_won_in_both_orders": len(first_both),
                "n_flipped_topics": len(flipped),
                "null_probability": 0.5,
                "p_value_exact_two_sided": binomial_test_two_sided(len(first_both), len(flipped)),
                "significant_at_0.05": binomial_test_two_sided(len(first_both), len(flipped)) < 0.05,
            },
        },
        "agree_or_tie_rule": {
            "rule": "accept a verdict only if both orders agree, otherwise call it a tie",
            "topics_with_an_accepted_verdict": prop(len(consistent), n),
            "topics_turned_into_a_tie_by_the_rule": len(flipped),
            "note": "This measures consistency only. It does not show that the accepted verdicts are correct.",
        },
    }


def main(tag: str):
    path = os.path.join(RESULTS_DIR, tag, "pairwise_raw.csv")
    variant, slot = defaultdict(dict), defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["parse_ok"] == "True":
                variant[r["topic_id"]][r["order"]] = r["variant_winner"]
                slot[r["topic_id"]][r["order"]] = r["slot_winner"]
    out = {"judge_tag": tag, "source": f"existing replies in evaluation/results/{tag}/pairwise_raw.csv; no LLM calls were made",
           "measures": "consistency only, not accuracy", **consistency_stats(variant, slot),
           "date": datetime.date.today().isoformat()}
    with open(os.path.join(RESULTS_DIR, tag, "pairwise_consistency.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    s = out["same_winner_in_both_orders"]
    t = out["among_flipped_topics"]["binomial_test"]
    print(f"{s['k']}/{s['n']} topics ({s['p']:.1%}) have the same winner in both orders; "
          f"binomial p={t['p_value_exact_two_sided']:.3f} for {t['k_first_shown_won_in_both_orders']}/{t['n_flipped_topics']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="groq-gpt-oss-120b")
    main(ap.parse_args().tag)
