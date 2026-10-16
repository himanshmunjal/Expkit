"""Adapters that register the built-in methods in the registry.

Each adapter maps the uniform (control, treatment) MetricSample signature onto
the plain-array function that does the maths. The maths functions stay usable
on their own; this file is the only place that knows about both.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from expkit._types import MetricSample, TestResult
from expkit.registry import register
from expkit.stats.bayes import beta_binomial
from expkit.stats.bootstrap import bootstrap_mean_diff, bootstrap_ratio_diff
from expkit.stats.cuped import cuped_ttest
from expkit.stats.delta import delta_ratio_test, jackknife_ratio_test, naive_ratio_ttest
from expkit.stats.ttest import mann_whitney, welch_ttest, z_test
from expkit.stats.winsor import winsorize_pair


@register("welch", description="Welch t-test on the difference in means")
def _welch(c: MetricSample, t: MetricSample, *, alpha, rng, **_) -> TestResult:
    return welch_ttest(c.y, t.y, alpha)


@register("ztest", description="Large-sample z-test on the difference in means")
def _ztest(c, t, *, alpha, rng, **_):
    return z_test(c.y, t.y, alpha)


@register("mannwhitney", description="Mann-Whitney U (rank test; no CI)")
def _mw(c, t, *, alpha, rng, **_):
    return mann_whitney(c.y, t.y, alpha)


@register("bootstrap", description="Percentile bootstrap of the mean difference")
def _boot(c, t, *, alpha, rng, n_boot: int = 2000, **_):
    return bootstrap_mean_diff(c.y, t.y, alpha, n_boot=n_boot, kind="percentile", rng=rng)


@register("bootstrap_bca", description="BCa bootstrap of the mean difference")
def _boot_bca(c, t, *, alpha, rng, n_boot: int = 2000, **_):
    return bootstrap_mean_diff(c.y, t.y, alpha, n_boot=n_boot, kind="bca", rng=rng)


@register("cuped", needs_covariate=True, description="CUPED-adjusted Welch t-test")
def _cuped(c, t, *, alpha, rng, zero_indicator: bool = False, **_):
    return cuped_ttest(c.y, t.y, c.x, t.x, alpha, zero_indicator=zero_indicator)


@register("cuped_zero", needs_covariate=True,
          description="CUPED with a zero-pre-activity indicator covariate")
def _cuped_zero(c, t, *, alpha, rng, **_):
    res = cuped_ttest(c.y, t.y, c.x, t.x, alpha, zero_indicator=True)
    return replace(res, method="cuped_zero")


@register("winsor_welch", description="Welch t-test after capping at a pooled quantile")
def _winsor(c, t, *, alpha, rng, quantile: float = 0.99, **_):
    wc, wt, cap = winsorize_pair(c.y, t.y, quantile)
    res = welch_ttest(wc, wt, alpha, method="winsor_welch")
    return replace(res, extra={**res.extra, "cap": cap, "quantile": quantile})


@register("delta", kinds=("ratio",), description="Delta-method test on ratio of sums")
def _delta(c, t, *, alpha, rng, **_):
    return delta_ratio_test(c.y, c.den, t.y, t.den, alpha)


@register("jackknife_ratio", kinds=("ratio",),
          description="Jackknife variance + t test on ratio of sums (default; robust to heavy users)")
def _jackknife_ratio(c, t, *, alpha, rng, **_):
    return jackknife_ratio_test(c.y, c.den, t.y, t.den, alpha)


@register("naive_ratio", kinds=("ratio",),
          description="Welch t-test on per-user ratios (for comparison only)")
def _naive_ratio(c, t, *, alpha, rng, **_):
    return naive_ratio_ttest(c.y, c.den, t.y, t.den, alpha)


@register("bootstrap_ratio", kinds=("ratio",), description="User-level bootstrap of ratio of sums")
def _boot_ratio(c, t, *, alpha, rng, n_boot: int = 2000, **_):
    return bootstrap_ratio_diff(c.y, c.den, t.y, t.den, alpha, n_boot=n_boot, rng=rng)


@register("bayes_beta", kinds=("binary",),
          description="Beta-binomial posterior; p_value = 2*min(P(better), P(worse)), not a p-value")
def _bayes(c, t, *, alpha, rng, n_draws: int = 20_000, **_):
    sc, st = int(round(c.y.sum())), int(round(t.y.sum()))
    b = beta_binomial(sc, c.y.size, st, t.y.size, alpha=alpha, n_draws=n_draws, rng=rng)
    pb = b.prob_treatment_better
    mc, mt = float(c.y.mean()), float(t.y.mean())
    return TestResult(
        "bayes_beta", mt - mc, b.lift_ci_low, b.lift_ci_high,
        float(np.clip(2 * min(pb, 1 - pb), 0, 1)), alpha, mc, mt, c.y.size, t.y.size,
        {"prob_treatment_better": pb, "expected_loss_ship": b.expected_loss_ship},
    )
