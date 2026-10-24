import pytest

from evaluation.analyze import pairwise_stats, retest_stats, truncation_stats
from evaluation.judge_pairwise import parse_pairwise, to_variant






def test_to_variant_maps_slot_back_to_variant():
    assert to_variant("AB", "first") == "A" and to_variant("AB", "second") == "B"
    assert to_variant("BA", "first") == "B" and to_variant("BA", "second") == "A"
    assert to_variant("BA", "tie") == "tie"


def test_parse_pairwise():
    assert parse_pairwise('{"winner": "A", "reason": "x"}') == ("first", "x")
    assert parse_pairwise('```json\n{"winner": "b"}\n```')[0] == "second"
    assert parse_pairwise('{"winner": "Tie"}')[0] == "tie"
    with pytest.raises(ValueError):
        parse_pairwise('{"winner": "C"}')


def test_retest_stats_known_values():
    stats = retest_stats({"p1": [7, 7, 7], "p2": [6, 8, 7]})
    # p1: diffs 0,0,0 -> 0 ; p2: diffs 2,1,1 -> 4/3 ; mean = 2/3
    assert stats["mean_abs_diff"] == pytest.approx(2 / 3)
    assert stats["share_posts_all_runs_identical"]["k"] == 1
    assert stats["share_run_pairs_identical"]["k"] == 3 and stats["share_run_pairs_identical"]["n"] == 6


def test_truncation_stats_known_values():
    stats = truncation_stats({"a": [6, 6, 6], "b": [8, 8, 8]}, {"a": [8, 8, 8], "b": [8, 8, 8]})
    assert stats["n_long_posts"] == 2
    assert stats["mean_diff_full_minus_truncated"] == pytest.approx(1.0)
    assert stats["share_changed_by_1_or_more_points"]["k"] == 1
    assert stats["share_pass_fail_flipped"]["k"] == 1       # a: fail (6) -> pass (8)


def test_pairwise_stats_position_bias_detection():
    # t1 consistent (A wins both orders); t2 pure first-position bias; t3 pure second-position bias;
    # t4 tie in one order.
    verdicts = {"t1": {"AB": "A", "BA": "A"}, "t2": {"AB": "A", "BA": "B"},
                "t3": {"AB": "B", "BA": "A"}, "t4": {"AB": "A", "BA": "tie"}}
    slots = {"t1": {"AB": "first", "BA": "second"}, "t2": {"AB": "first", "BA": "first"},
             "t3": {"AB": "second", "BA": "second"}, "t4": {"AB": "first", "BA": "tie"}}
    s = pairwise_stats(verdicts, slots)
    assert s["inconsistency_rate"]["k"] == 3 and s["inconsistency_rate"]["n"] == 4
    assert s["among_inconsistent"]["first_shown_won_in_both_orders"]["k"] == 1
    assert s["among_inconsistent"]["second_shown_won_in_both_orders"]["k"] == 1
    assert s["among_inconsistent"]["involved_a_tie_in_one_order"]["k"] == 1
