"""Ratio metrics: delta method vs user-level bootstrap vs naive per-user ratios.

Acceptance: delta and bootstrap coverage of the true ratio-of-sums effect near
95%; the naive estimator targets a different quantity, so its "coverage" of
the ratio-of-sums effect is reported to show how it misleads.
"""

from __future__ import annotations

import pandas as pd
from _common import load_population, parser, save

from expkit.catalog import ACCEPT_RATE
from expkit.sim import SimulationSpec, run_simulation, summarize


def main() -> None:
    a = parser(__doc__, runs=2000).parse_args()
    df, source = load_population(a.data)
    sizes, runs = ((1000,), 100) if a.quick else ((500, 2000, 10_000), a.runs)
    raw = pd.concat([
        run_simulation(df, SimulationSpec((ACCEPT_RATE,), n, runs, lift,
                                          ("delta", "jackknife_ratio", "bootstrap_ratio", "naive_ratio"),
                                          seed=a.seed + i, n_jobs=a.jobs,
                                          method_options={"bootstrap_ratio": {"n_boot": 500}}))
        for i, (n, lift) in enumerate((n, lift) for n in sizes for lift in (0.0, 0.05))
    ], ignore_index=True)
    save("ratio_metrics", raw, summarize(raw), source, {"runs": runs, "sizes": list(sizes)})


if __name__ == "__main__":
    main()
