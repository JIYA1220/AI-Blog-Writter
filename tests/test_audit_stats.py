import math

import pytest

from evaluation.fix_impact import fix_impact_stats
from evaluation.metrics import binomial_test_two_sided
from evaluation.pairwise_consistency import consistency_stats


# ---- exact binomial test ------------------------------------------------------------------

def test_binomial_six_of_seven_is_not_significant():
    # outcomes no more likely than 6/7: 0, 1, 6, 7 successes -> (1 + 7 + 7 + 1) / 128 = 0.125
    assert binomial_test_two_sided(6, 7) == pytest.approx(0.125)


def test_binomial_known_values():
    assert binomial_test_two_sided(5, 10) == pytest.approx(1.0)
    assert binomial_test_two_sided(0, 5) == pytest.approx(2 / 32)
    assert binomial_test_two_sided(9, 10) == pytest.approx(22 / 1024)


def test_binomial_is_symmetric_and_handles_empty():
    assert binomial_test_two_sided(2, 9) == pytest.approx(binomial_test_two_sided(7, 9))
    assert math.isnan(binomial_test_two_sided(0, 0))


# ---- pairwise consistency -------------------------------------------------------------------

def test_consistency_stats_toy_data():
    variant = {
        "t1": {"AB": "B", "BA": "B"},        # consistent, long post wins
        "t2": {"AB": "A", "BA": "A"},        # consistent, short post wins
        "t3": {"AB": "A", "BA": "B"},        # flip: first-shown wins in both orders
        "t4": {"AB": "B", "BA": "A"},        # flip: second-shown wins in both orders
        "t5": {"AB": "A", "BA": "tie"},      # flip involving a tie
        "t6": {"AB": "tie", "BA": "tie"},    # consistent tie
    }
    slot = {
        "t1": {"AB": "second", "BA": "first"}, "t2": {"AB": "first", "BA": "second"},
        "t3": {"AB": "first", "BA": "first"}, "t4": {"AB": "second", "BA": "second"},
        "t5": {"AB": "first", "BA": "tie"}, "t6": {"AB": "tie", "BA": "tie"},
    }
    s = consistency_stats(variant, slot)
    assert s["n_topics"] == 6
    assert (s["same_winner_in_both_orders"]["k"], s["same_winner_in_both_orders"]["n"]) == (3, 6)
    assert s["among_consistent_topics"]["winner_counts"] == {"A_short": 1, "B_long": 1, "tie": 1}
    flipped = s["among_flipped_topics"]
    assert (flipped["n"], flipped["first_shown_won_in_both_orders"],
            flipped["second_shown_won_in_both_orders"], flipped["tie_in_one_order"]) == (3, 1, 1, 1)
    assert s["agree_or_tie_rule"]["topics_with_an_accepted_verdict"]["k"] == 3
    assert s["agree_or_tie_rule"]["topics_turned_into_a_tie_by_the_rule"] == 3


def test_consistency_stats_reports_binomial_test():
    variant = {f"t{i}": {"AB": "A", "BA": "B"} for i in range(7)}
    slot = {f"t{i}": {"AB": "first", "BA": "first"} for i in range(6)}
    slot["t6"] = {"AB": "second", "BA": "second"}
    test = consistency_stats(variant, slot)["among_flipped_topics"]["binomial_test"]
    assert (test["k_first_shown_won_in_both_orders"], test["n_flipped_topics"]) == (6, 7)
    assert test["p_value_exact_two_sided"] == pytest.approx(0.125)
    assert test["significant_at_0.05"] is False


# ---- fix impact -----------------------------------------------------------------------------

def test_fix_impact_stats_toy_data():
    old = {"a": [6, 6, 6], "b": [8, 8, 8], "c": [5, 5, 5], "d": [7, 7, 7]}
    new = {"a": [8, 8, 8], "b": [8, 8, 8], "c": [5, 5, 5], "d": [6, 6, 6]}
    s = fix_impact_stats(old, new)
    assert s["n_posts"] == 4
    assert (s["pass_rate_before_4000_chars"]["k"], s["pass_rate_after_full_text"]["k"]) == (2, 2)
    assert s["verdict_changes"] == {"fail_to_pass": 1, "pass_to_fail": 1, "unchanged": 2}
    assert s["mean_score_change"] == pytest.approx((2 + 0 + 0 - 1) / 4)


def test_fix_impact_uses_mean_of_runs_for_the_pass_rule():
    s = fix_impact_stats({"a": [6, 7, 7]}, {"a": [8, 8, 8]})      # mean 6.67 -> fail, then 8 -> pass
    assert s["pass_rate_before_4000_chars"]["k"] == 0 and s["pass_rate_after_full_text"]["k"] == 1


def test_fix_impact_only_compares_posts_present_in_both_views():
    s = fix_impact_stats({"a": [8], "b": [8]}, {"a": [8]})
    assert s["n_posts"] == 1
