"""Mean-difference tests: Welch t, large-sample z, Mann-Whitney U."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from expkit._types import TestResult, as_1d_float


def _check_sizes(c: np.ndarray, t: np.ndarray) -> None:
    if c.size < 2 or t.size < 2:
        raise ValueError("each arm needs at least 2 observations")


def welch_ttest(control, treatment, alpha: float = 0.05, method: str = "welch") -> TestResult:
    """Welch's unequal-variance t-test on the difference in means."""
    c = as_1d_float(control, "control")
    t = as_1d_float(treatment, "treatment")
    _check_sizes(c, t)
    nc, nt = c.size, t.size
    mc, mt = float(c.mean()), float(t.mean())
    vc, vt = float(c.var(ddof=1)), float(t.var(ddof=1))
    diff = mt - mc
    se2 = vc / nc + vt / nt
    if se2 == 0.0:
        p = 1.0 if diff == 0.0 else 0.0
        return TestResult(method, diff, diff, diff, p, alpha, mc, mt, nc, nt, {"se": 0.0})
    se = math.sqrt(se2)
    dof_den = (vc / nc) ** 2 / (nc - 1) + (vt / nt) ** 2 / (nt - 1)
    # Welch-Satterthwaite; squares of tiny variances can underflow to 0, in which
    # case fall back to the pooled degrees of freedom.
    dof = se2**2 / dof_den if dof_den > 0 else float(nc + nt - 2)
    tstat = diff / se
    p = float(2.0 * stats.t.sf(abs(tstat), dof))
    q = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
    return TestResult(
        method, diff, diff - q * se, diff + q * se, p, alpha, mc, mt, nc, nt,
        {"se": se, "t": tstat, "df": dof},
    )


def z_test(control, treatment, alpha: float = 0.05, method: str = "ztest") -> TestResult:
    """Large-sample normal test on the difference in means (unpooled variance)."""
    c = as_1d_float(control, "control")
    t = as_1d_float(treatment, "treatment")
    _check_sizes(c, t)
    mc, mt = float(c.mean()), float(t.mean())
    diff = mt - mc
    se = math.sqrt(c.var(ddof=1) / c.size + t.var(ddof=1) / t.size)
    if se == 0.0:
        p = 1.0 if diff == 0.0 else 0.0
        return TestResult(method, diff, diff, diff, p, alpha, mc, mt, c.size, t.size, {"se": 0.0})
    z = diff / se
    q = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = float(2.0 * stats.norm.sf(abs(z)))
    return TestResult(
        method, diff, diff - q * se, diff + q * se, p, alpha, mc, mt, c.size, t.size,
        {"se": se, "z": z},
    )


def mann_whitney(control, treatment, alpha: float = 0.05) -> TestResult:
    """Mann-Whitney U. Tests stochastic dominance, NOT a difference in means.

    The estimate reported is still the mean difference for comparability, and
    no confidence interval is given because the test does not imply one for it.
    """
    c = as_1d_float(control, "control")
    t = as_1d_float(treatment, "treatment")
    _check_sizes(c, t)
    res = stats.mannwhitneyu(t, c, alternative="two-sided", method="asymptotic")
    mc, mt = float(c.mean()), float(t.mean())
    return TestResult(
        "mannwhitney", mt - mc, math.nan, math.nan, float(res.pvalue), alpha, mc, mt,
        c.size, t.size, {"u": float(res.statistic)},
    )
