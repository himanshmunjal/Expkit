"""SRM detection vs bug severity, using production-style hash bucketing.

Acceptance: detection rate ~0.1% (the threshold) on clean runs; detection
rises with the share of treatment users silently dropped.
"""

from __future__ import annotations

import pandas as pd
from _common import load_population, parser, save

from expkit.sim import run_srm_simulation

DROPS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)


def main() -> None:
    a = parser(__doc__, runs=1000).parse_args()
    df, source = load_population(a.data)
    n_users, runs = (5_000, 50) if a.quick else (20_000, a.runs)
    raw = pd.concat([run_srm_simulation(df.user_id.to_numpy(), n_users, d, runs,
                                        seed=a.seed + i, n_jobs=a.jobs)
                     for i, d in enumerate(DROPS)], ignore_index=True)
    summary = raw.groupby("drop_fraction", as_index=False).agg(
        detection_rate=("detected", "mean"), median_p=("p_value", "median"))
    save("srm", raw, summary, source, {"runs": runs, "n_users": n_users})


if __name__ == "__main__":
    main()
