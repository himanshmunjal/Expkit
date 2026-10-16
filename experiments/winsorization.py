"""Capping heavy tails: the bias-variance trade-off against a known lift.

Capping shrinks variance (more power) but the capped metric is a different
quantity, so its estimate of the raw lift is biased. Both are reported, as the
brief requires (capped AND uncapped).

Runs an A/A arm (false-positive rate under capping) and a +20% lift at 20k/arm,
which Welch detects ~75% of the time on the real data. The first run used +3%
at 5k/arm, where every method had ~5% power and the comparison said nothing
(DECISION D-052).
"""

from __future__ import annotations

import pandas as pd
from _common import load_population, parser, save

from expkit.catalog import ANSWERS, COMMENTS
from expkit.sim import SimulationSpec, run_simulation, summarize

QUANTILES = (0.95, 0.99, 0.999)
LIFTS = (0.0, 0.20)


def main() -> None:
    a = parser(__doc__, runs=2000).parse_args()
    df, source = load_population(a.data)
    n, runs = (2000, 100) if a.quick else (20_000, a.runs)
    parts = []
    for i, lift in enumerate(LIFTS):
        spec = dict(metrics=(ANSWERS, COMMENTS), n_per_arm=n, n_runs=runs, lift=lift,
                    seed=a.seed + i, n_jobs=a.jobs)
        parts.append(run_simulation(df, SimulationSpec(methods=("welch", "cuped"), **spec)))
        for q in QUANTILES:
            res = run_simulation(df, SimulationSpec(methods=("winsor_welch",), **spec,
                                                    method_options={"winsor_welch": {"quantile": q}}))
            parts.append(res.assign(method=f"winsor_{q}"))
    raw = pd.concat(parts, ignore_index=True)
    summary = summarize(raw)
    summary["relative_bias"] = summary.bias / summary.true_effect.where(summary.true_effect != 0)
    save("winsorization", raw, summary, source, {"runs": runs, "n_per_arm": n, "lifts": list(LIFTS)})


if __name__ == "__main__":
    main()
