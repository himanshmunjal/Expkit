"""Peeking and sequential-monitoring simulation.

Users enter the experiment in random order; at each of K looks the analyst
sees the first k/K of each arm (sample accrual over time). Per run we record
whether each monitoring rule ever rejects, and at which look it would stop.
Statistics at each look are computed from cumulative sums, so a 28-look
simulation costs about as much as a single test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from expkit._types import MetricSpec
from expkit.effects import inject
from expkit.sequential import group_sequential_boundaries, msprt_pvalues
from expkit.sim._parallel import run_many

RULES = ("fixed_horizon", "naive_peeking", "obf", "pocock", "msprt")


@dataclass(frozen=True)
class PeekingSpec:
    metric: MetricSpec
    n_per_arm: int
    looks: int = 28
    n_runs: int = 2000
    lift: float = 0.0
    alpha: float = 0.05
    # mSPRT mixing sd in units of the SE of the difference at the final look (DECISION D-051).
    # Scaling tau to the metric mean made it ~0.5 SE for answers_posted, so mSPRT never rejected.
    msprt_tau_se: float = 3.0
    seed: int = 0
    n_jobs: int = 1


def _cumulative_stats(y: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.cumsum(y)[ends - 1]
    s2 = np.cumsum(y * y)[ends - 1]
    mean = s1 / ends
    var = (s2 - ends * mean**2) / (ends - 1)
    return mean, np.maximum(var, 0.0)


def run_peeking_simulation(df: pd.DataFrame, spec: PeekingSpec) -> pd.DataFrame:
    if spec.metric.kind == "ratio":
        raise ValueError("peeking simulation supports mean/binary metrics")
    population = spec.metric.sample(df)
    n, K = spec.n_per_arm, spec.looks
    ends = np.unique(np.ceil(n * np.arange(1, K + 1) / K).astype(int))
    ends = ends[ends >= 2]
    fractions = ends / n
    bounds = {r: group_sequential_boundaries(fractions, spec.alpha, r) for r in ("obf", "pocock")}
    z_crit = stats.norm.ppf(1 - spec.alpha / 2)
    se_final2 = 2.0 * float(np.var(population.y, ddof=1)) / n   # Var(diff) at the planned horizon
    tau2 = spec.msprt_tau_se**2 * se_final2
    if tau2 <= 0:
        raise ValueError("metric has zero variance; mSPRT tau is undefined")

    def one_run(run_id: int, rng: np.random.Generator) -> dict:
        idx = rng.choice(len(df), size=2 * n, replace=False)
        yc = population.y[idx[:n]]
        yt = inject(population.take(idx[n:]), spec.metric.kind, spec.lift, rng).y
        mc, vc = _cumulative_stats(yc, ends)
        mt, vt = _cumulative_stats(yt, ends)
        diff = mt - mc
        var = vc / ends + vt / ends
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(var > 0, diff / np.sqrt(var), 0.0)
        p_av = msprt_pvalues(diff, np.maximum(var, 1e-300), tau2)
        crossings = {
            "naive_peeking": np.abs(z) >= z_crit,
            "obf": np.abs(z) >= bounds["obf"],
            "pocock": np.abs(z) >= bounds["pocock"],
            "msprt": p_av < spec.alpha,
        }
        row = {"run": run_id, "lift": spec.lift, "n_per_arm": n, "looks": len(ends),
               "fixed_horizon_reject": bool(abs(z[-1]) >= z_crit),
               "fixed_horizon_stop_fraction": 1.0}
        for rule, hit in crossings.items():
            first = int(np.argmax(hit)) if hit.any() else None
            row[f"{rule}_reject"] = first is not None
            row[f"{rule}_stop_fraction"] = float(fractions[first]) if first is not None else 1.0
        return row

    return pd.DataFrame(run_many(one_run, spec.n_runs, spec.seed, spec.n_jobs))


def summarize_peeking(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        rows.append({
            "rule": rule,
            "rejection_rate": float(results[f"{rule}_reject"].mean()),
            "mean_sample_fraction": float(results[f"{rule}_stop_fraction"].mean()),
        })
    return pd.DataFrame(rows)
