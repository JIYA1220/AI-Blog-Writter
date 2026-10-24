import math

import numpy as np
import pytest

from evaluation.metrics import (accuracy, cohen_kappa, mean_bootstrap_ci, spearman,
                                spearman_bootstrap_ci, wilson_ci)


def test_wilson_known_value():
    # 8 of 10, 95%: textbook Wilson interval (0.4902, 0.9433)
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_edges_stay_inside_unit_interval():
    assert wilson_ci(0, 10)[0] == 0.0 and 0 < wilson_ci(0, 10)[1] < 0.35
    assert wilson_ci(10, 10)[1] == 1.0 and wilson_ci(10, 10)[0] > 0.65
    assert math.isnan(wilson_ci(0, 0)[0])


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    assert spearman([1, 2, 3, 4], [1, 4, 9, 1000]) == pytest.approx(1.0)


def test_spearman_known_value_with_ties():
    # hand-checked: ranks x = [1, 2.5, 2.5, 4], y = [1, 2, 3, 4] -> r = 0.9487
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9487, abs=1e-3)


def test_spearman_constant_input_is_nan():
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_spearman_bootstrap_ci_brackets_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = x + rng.normal(scale=0.5, size=40)
    lo, hi = spearman_bootstrap_ci(x, y, n_resamples=300, seed=1)
    assert lo <= spearman(x, y) <= hi
    assert (lo, hi) == spearman_bootstrap_ci(x, y, n_resamples=300, seed=1)


def test_kappa_perfect_agreement_is_one():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)


def test_kappa_hand_computed():
    a = ["y", "y", "y", "n", "n"]
    b = ["y", "y", "n", "n", "n"]
    # observed 4/5 = 0.8 ; expected = 0.6*0.4 + 0.4*0.6 = 0.48 ; kappa = 0.32/0.52
    assert cohen_kappa(a, b) == pytest.approx(0.32 / 0.52)


def test_kappa_no_agreement_beyond_chance_is_zero():
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_kappa_degenerate_is_nan():
    assert math.isnan(cohen_kappa(["a", "a"], ["a", "a"]))


def test_accuracy_counts():
    assert accuracy([1, 2, 3], [1, 2, 4]) == (2, 3)


def test_mean_bootstrap_ci_contains_mean():
    v = [1, 2, 3, 4, 5, 6]
    lo, hi = mean_bootstrap_ci(v, n_resamples=500)
    assert lo <= np.mean(v) <= hi
