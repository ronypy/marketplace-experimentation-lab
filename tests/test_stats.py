"""Tests for src/stats.py: SRM, CUPED unbiasedness, and A/A false-positive rate."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stats import cuped, diff_ci, holm_adjust, srm_check, standardized_mean_diff, variance_reduction

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hillstrom.csv")


def test_srm_detects_balanced_split():
    chi2, p = srm_check([21307, 21387, 21306])
    assert p > 0.001


def test_srm_detects_mismatch():
    chi2, p = srm_check([25000, 19500, 19500])
    assert p < 0.001


def test_standardized_mean_diff_zero_for_identical_groups():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert standardized_mean_diff(x, x) == 0.0


def test_holm_adjust_is_monotone_and_conservative():
    p_adj = holm_adjust([0.01, 0.04])
    assert all(p_adj[i] >= 0.01 for i in range(2))
    assert np.all(np.diff(sorted(p_adj)) >= 0)


def test_cuped_preserves_treatment_effect_in_expectation():
    """CUPED must not bias the estimated treatment effect, only reduce its variance."""
    rng = np.random.default_rng(0)
    n = 20_000
    true_effect = 2.0

    x = rng.normal(50, 10, n)  # pre-period covariate, unaffected by treatment
    treat = rng.integers(0, 2, n)
    noise = rng.normal(0, 20, n)
    y = 10 + 0.8 * x + true_effect * treat + noise

    y_adj = cuped(y, x)

    raw_diff = y[treat == 1].mean() - y[treat == 0].mean()
    adj_diff = y_adj[treat == 1].mean() - y_adj[treat == 0].mean()

    assert raw_diff == pytest.approx(true_effect, abs=0.5)
    assert adj_diff == pytest.approx(true_effect, abs=0.5)
    assert adj_diff == pytest.approx(raw_diff, abs=0.3)


def test_cuped_reduces_variance_when_covariate_correlates():
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.normal(0, 1, n)
    y = 5 * x + rng.normal(0, 1, n)  # strong correlation(y, x)

    y_adj = cuped(y, x)
    reduction = variance_reduction(y, y_adj)

    assert reduction > 0.8  # most of the variance should be removed


def test_synthetic_injected_effect_recovered_within_ci():
    """CUPED's CI for a known, injected effect should cover the true effect."""
    rng = np.random.default_rng(7)
    n = 10_000
    true_effect = 1.5

    x = rng.normal(100, 15, 2 * n)
    treat = np.concatenate([np.ones(n), np.zeros(n)])
    y = 20 + 0.5 * x + true_effect * treat + rng.normal(0, 5, 2 * n)

    y_adj = cuped(y, x)
    y_t, y_c = y_adj[treat == 1], y_adj[treat == 0]
    d, se, (lo, hi), p = diff_ci(y_t, y_c)

    assert lo <= true_effect <= hi


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="hillstrom.csv not downloaded")
def test_aa_false_positive_rate_on_control_arm():
    """A/A split of the real control arm: both raw and CUPED should have ~5% false positives."""
    df = pd.read_csv(DATA_PATH)
    control = df[df.segment == "No E-Mail"]
    y = control.spend.values
    x = control.history.values
    n = len(y)

    rng = np.random.default_rng(123)
    n_sims = 200
    p_raw, p_cuped = [], []

    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())

    for _ in range(n_sims):
        perm = rng.permutation(n)
        half = n // 2
        idx_a, idx_b = perm[:half], perm[half:]

        _, _, _, p = diff_ci(y[idx_a], y[idx_b])
        p_raw.append(p)

        _, _, _, p2 = diff_ci(y_adj[idx_a], y_adj[idx_b])
        p_cuped.append(p2)

    fpr_raw = np.mean(np.array(p_raw) < 0.05)
    fpr_cuped = np.mean(np.array(p_cuped) < 0.05)

    # 200 sims at a true 5% rate: allow a generous band so the test isn't flaky.
    assert 0.01 <= fpr_raw <= 0.12
    assert 0.01 <= fpr_cuped <= 0.12
