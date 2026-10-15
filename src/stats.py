"""Core stats helpers for the promotion A/B test: SRM, balance, CIs, power, CUPED."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisquare, norm


def srm_check(counts, expected_props=None):
    """Chi-square sample-ratio-mismatch test. counts: array-like of arm sizes."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if expected_props is None:
        expected_props = np.ones(len(counts)) / len(counts)
    expected = n * np.asarray(expected_props)
    chi2, p = chisquare(counts, f_exp=expected)
    return chi2, p


def standardized_mean_diff(x_t, x_c):
    """Standardized mean difference between two groups for a covariate."""
    x_t, x_c = np.asarray(x_t, dtype=float), np.asarray(x_c, dtype=float)
    pooled_sd = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return (x_t.mean() - x_c.mean()) / pooled_sd


def diff_ci(y_t, y_c, alpha=0.05):
    """Welch-style difference in means with a two-sided CI and p-value."""
    y_t, y_c = np.asarray(y_t, dtype=float), np.asarray(y_c, dtype=float)
    d = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    z = norm.ppf(1 - alpha / 2)
    p = 2 * (1 - norm.cdf(abs(d / se))) if se > 0 else 1.0
    return d, se, (d - z * se, d + z * se), p


def two_proportion_ci(y_t, y_c, alpha=0.05):
    """Two-proportion z-test difference with a CI, for rate metrics like visit/conversion."""
    return diff_ci(y_t, y_c, alpha=alpha)


def relative_lift(y_t, y_c):
    """Relative lift of treatment mean over control mean."""
    y_t, y_c = np.asarray(y_t, dtype=float), np.asarray(y_c, dtype=float)
    c_mean = y_c.mean()
    if c_mean == 0:
        return np.nan
    return y_t.mean() / c_mean - 1


def n_per_arm(sigma, mde_abs, alpha=0.05, power=0.8):
    """Required sample size per arm for a two-sample mean test with a given absolute MDE."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(2 * sigma ** 2 * z ** 2 / mde_abs ** 2))


def mde(sigma, n, alpha=0.05, power=0.8):
    """Minimum detectable effect (absolute) for a given sample size per arm."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * np.sqrt(2 * sigma ** 2 / n)


def cuped(y, x):
    """CUPED-adjust outcome y using pre-period covariate x. theta is fit on the pooled sample."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


def variance_reduction(y_raw, y_adj):
    """Fraction of variance removed by an adjustment, i.e. equivalent sample saved."""
    y_raw, y_adj = np.asarray(y_raw, dtype=float), np.asarray(y_adj, dtype=float)
    var_raw = y_raw.var(ddof=1)
    var_adj = y_adj.var(ddof=1)
    return 1 - var_adj / var_raw


def holm_adjust(pvals):
    """Holm-Bonferroni adjusted p-values. Returns array in the original order."""
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(pvals, method="holm")
    return p_adj
