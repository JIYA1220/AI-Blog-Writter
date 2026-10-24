"""
Metric functions for the judge-reliability study. Plain numpy, no scipy.
Unit-tested in tests/test_eval_metrics.py.
"""

import math
from typing import Callable, Sequence, Tuple

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% by default)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties get the average of the ranks they span."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks). NaN if either side is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(x, y, stat: Callable, n_resamples: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of stat(x, y) over resampled pairs (NaN resamples are skipped)."""
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        v = stat(x[idx], y[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def spearman_bootstrap_ci(x, y, n_resamples: int = 2000, seed: int = 0) -> Tuple[float, float]:
    return bootstrap_ci(x, y, spearman, n_resamples, seed)


def mean_bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_resamples)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Unweighted Cohen's kappa between two label sequences. NaN if expected agreement is 1."""
    a, b = list(a), list(b)
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    labels = sorted(set(a) | set(b), key=str)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def accuracy(a: Sequence, b: Sequence) -> Tuple[int, int]:
    """Returns (number agreeing, n) so callers can attach a Wilson CI."""
    a, b = list(a), list(b)
    return sum(x == y for x, y in zip(a, b)), len(a)


def binomial_test_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """
    Exact two-sided binomial test of k successes in n trials against success probability p0.
    p-value = sum of the probabilities of all outcomes that are no more likely than the observed one.
    """
    if n <= 0:
        return float("nan")
    pmf = [math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(n + 1)]
    observed = pmf[k]
    return float(min(1.0, sum(x for x in pmf if x <= observed * (1 + 1e-9))))
