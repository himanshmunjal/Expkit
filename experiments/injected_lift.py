"""Power and recovery: inject known lifts (0.5%-10%) and measure power, bias
and CI coverage as a function of lift and sample size.

Acceptance: |bias| small relative to the true effect, coverage near 95%,
power monotone in lift and n.
"""

from __future__ import annotations

import pandas as pd
from _common import load_population, parser, save

from expkit.catalog import ACCEPT_RATE, ACTIVE, ANSWERS
from expkit.sim import SimulationSpec, run_simulation, summarize

LIFTS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
SIZES = (1_000, 5_000, 20_000)


def main() -> None:
    a = parser(__doc__, runs=2000).parse_args()
    df, source = load_population(a.data)
    lifts, sizes, runs = ((0.05,), (2000,), 100) if a.quick else (LIFTS, SIZES, a.runs)
    raw = pd.concat([
        run_simulation(df, SimulationSpec((ANSWERS, ACTIVE, ACCEPT_RATE), n, runs, lift,
                                          ("welch", "cuped", "delta", "jackknife_ratio"), seed=a.seed + i,
                                          n_jobs=a.jobs))
        for i, (lift, n) in enumerate((lift, n) for lift in lifts for n in sizes)
    ], ignore_index=True)
    summary = summarize(raw).rename(columns={"rejection_rate": "power"})
    summary["relative_bias"] = summary.bias / summary.true_effect
    save("injected_lift", raw, summary, source, {"runs": runs, "lifts": list(lifts),
                                                 "sizes": list(sizes)})


if __name__ == "__main__":
    main()
