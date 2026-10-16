"""A/A calibration: with no true effect, is the false-positive rate 5% and are
p-values Uniform(0, 1)? Shows where naive tests break on heavy-tailed metrics
at small n.

Acceptance: rejection rate's Wilson CI contains 0.05 and KS p > 0.01, per
metric x method x sample size. Failures are reported, not hidden.
"""

from __future__ import annotations

import pandas as pd
from _common import load_population, parser, save

from expkit.catalog import CATALOG
from expkit.sim import SimulationSpec, run_simulation, summarize

METHODS = ("welch", "mannwhitney", "cuped", "delta", "jackknife_ratio", "naive_ratio", "bootstrap_ratio", "bootstrap")
SIZES = (100, 1_000, 10_000)


def main() -> None:
    a = parser(__doc__, runs=5000).parse_args()
    df, source = load_population(a.data)
    sizes, runs = ((100, 1000), 200) if a.quick else (SIZES, a.runs)
    raw = []
    for n in sizes:
        # The bootstrap costs n_boot * n per run; keep it to sizes where it matters.
        methods = METHODS if n <= 1_000 else tuple(m for m in METHODS if not m.startswith("bootstrap"))
        spec = SimulationSpec(tuple(CATALOG.values()), n, runs, 0.0, methods, seed=a.seed + n,
                              n_jobs=a.jobs, method_options={"bootstrap": {"n_boot": 500},
                                                             "bootstrap_ratio": {"n_boot": 500}})
        raw.append(run_simulation(df, spec))
    raw = pd.concat(raw, ignore_index=True)
    summary = summarize(raw)
    # Two verdicts (DECISION D-053). `calibrated_at_alpha` is what a ship decision depends on: the
    # false-positive rate at alpha. `calibrated` (strict) also needs Uniform p-values by KS. With
    # 5,000 runs, KS flags harmless non-uniformity far from alpha (ties at p = 1 when both arms
    # are all zeros, discrete binary p-values), so it is reported but is not the pass/fail rule.
    summary["calibrated_at_alpha"] = (summary.rejection_ci_low <= 0.05) & (summary.rejection_ci_high >= 0.05)
    summary["calibrated"] = summary.calibrated_at_alpha & (summary.ks_uniform_p > 0.01)
    save("aa_calibration", raw, summary, source, {"runs": runs, "sizes": list(sizes)})


if __name__ == "__main__":
    main()
