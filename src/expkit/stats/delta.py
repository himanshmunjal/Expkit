"""Ratio metrics: delta method vs the naive mean-of-per-user-ratios.

The quantity of interest is the ratio of sums R = sum(num) / sum(den) in each
arm (e.g. accepted answers / answers). Users, not answers, are the randomised
unit, so answers within a user are correlated and a binomial variance on
answers is wrong. The delta method gives the user-level variance:

    Var(R) ~= (s_N^2 - 2 R s_ND + R^2 s_D^2) / (n * mean(D)^2)
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from expkit._types import TestResult, as_1d_float
from expkit.stats.ttest import welch_ttest


def ratio_and_variance(num, den) -> tuple[float, float]:
    num, den = as_1d_float(num, "num"), as_1d_float(den, "den")
    if num.size != den.size:
        raise ValueError("num and den must have equal length")
    if num.size < 2:
        raise ValueError("need at least 2 units")
    n = num.size
    mn, md = num.mean(), den.mean()
    if md <= 0:
        raise ValueError("mean denominator must be positive")
    r = mn / md
    cov = np.cov(num, den, ddof=1)
    var_r = (cov[0, 0] - 2.0 * r * cov[0, 1] + r * r * cov[1, 1]) / (n * md * md)
    return float(r), float(max(var_r, 0.0))


def delta_ratio_test(
    num_c, den_c, num_t, den_t,
    alpha: float = 0.05,
    relative: bool = False,
) -> TestResult:
    """Delta-method z-test on the difference (or relative lift) of ratios of sums."""
    rc, vc = ratio_and_variance(num_c, den_c)
    rt, vt = ratio_and_variance(num_t, den_t)
    if relative:
        if rc == 0:
            raise ValueError("control ratio is 0; relative lift undefined")
        est = rt / rc - 1.0
        var = vt / rc**2 + (rt**2) * vc / rc**4
    else:
        est = rt - rc
        var = vt + vc
    se = math.sqrt(var)
    q = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = float(2.0 * stats.norm.sf(abs(est) / se)) if se > 0 else float(est == 0.0)
    n_c, n_t = np.asarray(num_c).size, np.asarray(num_t).size
    return TestResult(
        "delta_relative" if relative else "delta", est, est - q * se, est + q * se, p, alpha,
        rc, rt, n_c, n_t, {"se": se, "relative": relative},
    )


def jackknife_ratio_and_variance(num, den) -> tuple[float, float, int]:
    """Ratio of sums with its leave-one-user-out jackknife variance.

    Returns (ratio, variance, effective units). Effective units are users with
    den > 0, the only ones that carry information about the ratio. Closed form,
    O(n): R_(-i) = (sum(num) - num_i) / (sum(den) - den_i).
    """
    num, den = as_1d_float(num, "num"), as_1d_float(den, "den")
    if num.size != den.size:
        raise ValueError("num and den must have equal length")
    n_eff = int((den > 0).sum())
    if n_eff < 2:
        raise ValueError("need at least 2 units with a positive denominator")
    s, t, k = num.sum(), den.sum(), num.size
    loo = (s - num) / (t - den)
    return float(s / t), float((k - 1) / k * np.sum((loo - loo.mean()) ** 2)), n_eff


def jackknife_ratio_test(num_c, den_c, num_t, den_t, alpha: float = 0.05) -> TestResult:
    """Difference of ratios of sums: jackknife variance, Student t reference.

    The delta method's linearised variance is biased low when a few users hold
    most of the denominator (heavy-tailed answer counts). On the real Stack
    Overflow data its A/A false-positive rate was 20% at n=100 and 10% at
    n=1,000 per arm. The jackknife (an HC3-like correction for high-leverage
    users) with t(df = min effective units - 1) brought this to 5-6.5%
    (DECISION D-050).
    """
    rc, vc, ec = jackknife_ratio_and_variance(num_c, den_c)
    rt, vt, et = jackknife_ratio_and_variance(num_t, den_t)
    est, se, dof = rt - rc, math.sqrt(vc + vt), min(ec, et) - 1
    q = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
    p = float(2.0 * stats.t.sf(abs(est) / se, dof)) if se > 0 else float(est == 0.0)
    return TestResult(
        "jackknife_ratio", est, est - q * se, est + q * se, p, alpha,
        rc, rt, np.asarray(num_c).size, np.asarray(num_t).size,
        {"se": se, "df": dof, "effective_units": (ec, et)},
    )


def naive_ratio_ttest(num_c, den_c, num_t, den_t, alpha: float = 0.05) -> TestResult:
    """Welch t-test on per-user ratios (users with den == 0 are dropped).

    This answers a different question (the average user's ratio, where a user
    with 1 answer counts as much as a user with 100) and is included to show
    how it diverges from the ratio of sums.
    """
    nc, dc = as_1d_float(num_c, "num_c"), as_1d_float(den_c, "den_c")
    nt, dt = as_1d_float(num_t, "num_t"), as_1d_float(den_t, "den_t")
    pc = nc[dc > 0] / dc[dc > 0]
    pt = nt[dt > 0] / dt[dt > 0]
    res = welch_ttest(pc, pt, alpha, method="naive_ratio")
    return res
