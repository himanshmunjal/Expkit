"""CUPED: does empirical variance reduction match the theoretical R^2, is the
lift estimate unbiased, and how much power / sample size does it buy?

Acceptance: empirical VR = 1 - var(cuped est)/var(welch est) within a few
points of the population R^2; bias ~ 0; power at 3% lift higher than Welch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import load_population, parser, save

from expkit.catalog import ACTIVE, ANSWERS, COMMENTS
from expkit.sim import SimulationSpec, run_simulation, summarize
from expkit.stats.cuped import cuped_theta

METRICS = (ANSWERS, COMMENTS, ACTIVE)


def theoretical_r2(df: pd.DataFrame, metric) -> float:
    s = metric.sample(df)
    theta = cuped_theta(s.y, s.x)
    x = s.x[:, None] if s.x.ndim == 1 else s.x
    resid = s.y - (x - x.mean(axis=0)) @ theta
    return float(1 - resid.var() / s.y.var())


def main() -> None:
    a = parser(__doc__, runs=2000).parse_args()
    df, source = load_population(a.data)
    n, runs = (2000, 100) if a.quick else (5000, a.runs)
    raw = pd.concat([
        run_simulation(df, SimulationSpec(METRICS, n, runs, lift,
                                          ("welch", "cuped", "cuped_zero"),
                                          seed=a.seed + k, n_jobs=a.jobs))
        for k, lift in enumerate((0.0, 0.03))
    ], ignore_index=True)
    summary = summarize(raw)
    rows = []
    for m in METRICS:
        aa = raw[(raw.metric == m.name) & (raw.lift == 0.0)]
        sd = aa.groupby("method").estimate.std()
        r2 = theoretical_r2(df, m)
        for method in ("cuped", "cuped_zero"):
            vr = 1 - (sd[method] / sd["welch"]) ** 2
            rows.append({"metric": m.name, "method": method, "theoretical_r2": r2,
                         "empirical_vr": vr, "sample_size_saving": vr,
                         "abs_gap": float(np.abs(vr - r2))})
    vr_table = pd.DataFrame(rows)
    summary = summary.merge(vr_table, on=["metric", "method"], how="left")
    save("cuped", raw, summary, source, {"runs": runs, "n_per_arm": n})


if __name__ == "__main__":
    main()
