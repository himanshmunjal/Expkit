"""A/A and injected-lift simulation harness.

One run = sample 2n users from the real population, randomise them into two
arms, inject a known lift into treatment (0 for A/A), and run every requested
method on every metric. All metrics share the same split within a run, which
is what makes family-wise error across metrics measurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from expkit import registry
from expkit._types import MetricSample, MetricSpec
from expkit.effects import inject
from expkit.sim._parallel import run_many


@dataclass(frozen=True)
class SimulationSpec:
    metrics: tuple[MetricSpec, ...]
    n_per_arm: int
    n_runs: int = 1000
    lift: float = 0.0
    methods: tuple[str, ...] = ("welch",)
    alpha: float = 0.05
    seed: int = 0
    n_jobs: int = 1
    method_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "methods", tuple(self.methods))
        if self.n_per_arm < 2 or self.n_runs < 1:
            raise ValueError("n_per_arm must be >= 2 and n_runs >= 1")


def population_statistic(sample: MetricSample, kind: str) -> float:
    if kind == "ratio":
        return float(sample.y.sum() / sample.den.sum())
    return float(sample.y.mean())


def applicable_methods(metric: MetricSpec, methods: tuple[str, ...]) -> list[str]:
    out = []
    for name in methods:
        entry = registry.get(name)
        if metric.kind in entry.kinds and (not entry.needs_covariate or metric.covariates):
            out.append(name)
    return out


def run_simulation(df: pd.DataFrame, spec: SimulationSpec) -> pd.DataFrame:
    """Return one row per (run, metric, method)."""
    if 2 * spec.n_per_arm > len(df):
        raise ValueError(f"need {2 * spec.n_per_arm} users, population has {len(df)}")
    population = {m.name: m.sample(df) for m in spec.metrics}
    truth = {m.name: spec.lift * population_statistic(population[m.name], m.kind)
             for m in spec.metrics}
    plan = {m.name: applicable_methods(m, spec.methods) for m in spec.metrics}
    n_pop, n = len(df), spec.n_per_arm

    def one_run(run_id: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        idx = rng.choice(n_pop, size=2 * n, replace=False)  # random order = random split
        rows = []
        for m in spec.metrics:
            control = population[m.name].take(idx[:n])
            treatment = inject(population[m.name].take(idx[n:]), m.kind, spec.lift, rng)
            for method in plan[m.name]:
                row = {"run": run_id, "metric": m.name, "method": method, "lift": spec.lift,
                       "n_per_arm": n, "true_effect": truth[m.name], "error": None}
                try:
                    res = registry.run(method, control, treatment, kind=m.kind, alpha=spec.alpha,
                                       rng=rng, **spec.method_options.get(method, {}))
                except ValueError as e:
                    # Degenerate sample (e.g. no answers in an arm at small n). Recorded,
                    # not hidden: summarize() reports the failure rate per method.
                    row.update(estimate=math.nan, ci_low=math.nan, ci_high=math.nan,
                               p_value=math.nan, control_mean=math.nan,
                               variance_reduction=math.nan, error=str(e))
                else:
                    row.update(estimate=res.estimate, ci_low=res.ci_low, ci_high=res.ci_high,
                               p_value=res.p_value, control_mean=res.control_mean,
                               variance_reduction=res.extra.get("variance_reduction", math.nan))
                rows.append(row)
        return rows

    runs = run_many(one_run, spec.n_runs, spec.seed, spec.n_jobs)
    return pd.DataFrame([row for rows in runs for row in rows])


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def summarize(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Per (metric, method, lift, n): rejection rate with Wilson CI, p-value
    uniformity (KS, meaningful for A/A only), bias, CI coverage and width."""
    out = []
    for key, g_all in results.groupby(["metric", "method", "lift", "n_per_arm"], sort=True):
        g = g_all[g_all.p_value.notna()]
        failed = len(g_all) - len(g)
        if g.empty:
            out.append({"metric": key[0], "method": key[1], "lift": key[2], "n_per_arm": key[3],
                        "runs": 0, "failed_runs": failed})
            continue
        k, n = int((g.p_value < alpha).sum()), len(g)
        lo, hi = wilson_interval(k, n)
        truth = float(g.true_effect.iloc[0])
        has_ci = g.ci_low.notna() & g.ci_high.notna()
        covered = (g.ci_low[has_ci] <= truth) & (truth <= g.ci_high[has_ci])
        out.append({
            "metric": key[0], "method": key[1], "lift": key[2], "n_per_arm": key[3],
            "runs": n, "failed_runs": failed, "rejection_rate": k / n, "rejection_ci_low": lo, "rejection_ci_high": hi,
            "ks_uniform_p": float(stats.kstest(g.p_value, "uniform").pvalue),
            "true_effect": truth, "mean_estimate": float(g.estimate.mean()),
            "bias": float(g.estimate.mean() - truth), "sd_estimate": float(g.estimate.std(ddof=1)),
            "coverage": float(covered.mean()) if has_ci.any() else math.nan,
            "mean_ci_width": float((g.ci_high - g.ci_low)[has_ci].mean()) if has_ci.any() else math.nan,
            "mean_variance_reduction": float(g.variance_reduction.mean()),
        })
    return pd.DataFrame(out)
