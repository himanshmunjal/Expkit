"""Peeking vs sequential testing over 28 daily looks, plus an mSPRT tau sweep.

Acceptance: naive daily peeking inflates the A/A false-positive rate well
above 5%; O'Brien-Fleming, Pocock and mSPRT bring it back to <= 5%. Under a
real lift, report power and the average fraction of the sample needed to stop.

Lifts are per metric and sized to be detectable at 10k/arm on the real data
(a 3% lift on answers_posted had ~6% power for every rule, so it compared
nothing). The sweep picks mSPRT's tau, in units of the final-look SE (D-051).
"""

from __future__ import annotations

import pandas as pd
from _common import load_population, parser, results_dir, save

from expkit.catalog import ACTIVE, ANSWERS
from expkit.sim import PeekingSpec, run_peeking_simulation, summarize_peeking

LIFTS = {ANSWERS.name: 0.20, ACTIVE.name: 0.05}
TAU_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


def main() -> None:
    a = parser(__doc__, runs=5000).parse_args()
    df, source = load_population(a.data)
    n, runs = (2000, 200) if a.quick else (10_000, a.runs)
    raws, sums, sweep = [], [], []
    for metric in (ANSWERS, ACTIVE):
        for k, lift in enumerate((0.0, LIFTS[metric.name])):
            raw = run_peeking_simulation(df, PeekingSpec(metric, n, 28, runs, lift,
                                                         seed=a.seed + k, n_jobs=a.jobs))
            raws.append(raw.assign(metric=metric.name))
            sums.append(summarize_peeking(raw).assign(metric=metric.name, lift=lift))
            for tau in TAU_GRID:
                s = summarize_peeking(run_peeking_simulation(df, PeekingSpec(
                    metric, n, 28, max(runs // 5, 100), lift, msprt_tau_se=tau,
                    seed=a.seed + 100 + k, n_jobs=a.jobs))).set_index("rule").loc["msprt"]
                sweep.append({"metric": metric.name, "lift": lift, "tau_se": tau,
                              "rejection_rate": s.rejection_rate,
                              "mean_sample_fraction": s.mean_sample_fraction, "data_source": source})
    save("peeking", pd.concat(raws, ignore_index=True), pd.concat(sums, ignore_index=True),
         source, {"runs": runs, "n_per_arm": n, "looks": 28, "lifts": LIFTS})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(results_dir(source) / "msprt_tau_sweep.csv", index=False)
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
