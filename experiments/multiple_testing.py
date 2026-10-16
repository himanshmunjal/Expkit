"""Multiple comparisons across metrics x tenure segments in A/A tests.

Each "experiment" looks at 4 metrics in 5 tenure quintiles (20 tests). With no
correction the chance of at least one false positive is far above 5%; Holm
restores family-wise error; BH controls the false discovery rate (which, with
every null true, equals the FWER).
"""

from __future__ import annotations

import pandas as pd
from _common import load_population, parser, save

from expkit.catalog import CATALOG
from expkit.sim import SimulationSpec, run_simulation
from expkit.stats.multiple import adjust


def main() -> None:
    a = parser(__doc__, runs=2000).parse_args()
    df, source = load_population(a.data)
    n, runs = (500, 100) if a.quick else (2000, a.runs)
    df = df.assign(segment=pd.qcut(df.tenure_days, 5, labels=False, duplicates="drop"))
    parts = []
    for seg, sub in df.groupby("segment"):
        res = run_simulation(sub.reset_index(drop=True),
                             SimulationSpec(tuple(CATALOG.values()), n, runs, 0.0,
                                            ("welch", "delta"), seed=a.seed + int(seg),
                                            n_jobs=a.jobs))
        parts.append(res.assign(segment=seg))
    raw = pd.concat(parts, ignore_index=True)
    rows = []
    for run, g in raw.groupby("run"):
        for method in ("none", "bonferroni", "holm", "bh"):
            rows.append({"run": run, "correction": method, "n_tests": len(g),
                         "any_rejection": bool((adjust(g.p_value.to_numpy(), method) < 0.05).any())})
    fam = pd.DataFrame(rows)
    summary = fam.groupby("correction", as_index=False).agg(
        family_wise_error=("any_rejection", "mean"), n_tests=("n_tests", "first"))
    save("multiple_testing", raw, summary, source, {"runs": runs, "n_per_arm": n})


if __name__ == "__main__":
    main()
