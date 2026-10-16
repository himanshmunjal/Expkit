"""Generate the four example readouts in docs/example_readouts/ (clearly
labelled as simulated): clear win, well-powered null, inconclusive, SRM failure.

Each scenario samples real users, hash-assigns them, injects a known effect
into answers_posted (keeping accepted <= answers and post_active consistent),
and runs the same `expkit.analyze` used for real experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import ROOT, load_population, parser, results_dir

from expkit import ExperimentConfig, analyze
from expkit.assign import hash_assign
from expkit.effects import inject_count

OUT = ROOT / "docs" / "example_readouts"

SCENARIOS = [
    # name, n_per_arm, lift, drop_fraction, relative MDE, expected decision
    # At 20k/arm an 8% lift shipped in only 28% of draws on the real data (CI ~+-0.047 vs an
    # effect of ~0.037). 80% power needs ~15% lift at 20k/arm, ~7.5% at full population (D-049).
    ("1_clear_win", 80_000, 0.20, 0.0, 0.05, "ship"),
    # On the real data the CUPED CI at 60k/arm is ~+-0.027 against a 5% MDE of ~0.023, so a
    # null can never be declared there. Use the whole population and a 10% MDE (DECISION D-049).
    ("2_well_powered_null", 80_000, 0.0, 0.0, 0.10, "do_not_ship"),
    ("3_inconclusive", 1_500, 0.03, 0.0, 0.05, "extend"),
    # 3% loss at 20k/arm gives chi2 ~ 9 (p ~ .0025): NOT caught at p < .001. 5% gives chi2 ~ 26.
    ("4_srm_failure", 20_000, 0.08, 0.05, 0.05, "invalid"),
]

CONFIG = {
    "experiment": {"arm_column": "arm", "method": "cuped", "alpha": 0.05, "mde": 0.05,
                   "simulated": True},
    "primary": {"name": "answers_posted", "kind": "mean", "column": "post_answers",
                "covariates": ["pre_answers"]},
    "guardrails": [
        {"name": "comments_posted", "kind": "mean", "column": "post_comments",
         "covariates": ["pre_comments"]},
        {"name": "accepted_answer_rate", "kind": "ratio", "column": "post_accepted",
         "denominator": "post_answers"},
    ],
}


def build(df: pd.DataFrame, n: int, lift: float, drop: float, rng, salt: str) -> pd.DataFrame:
    sample = df.iloc[rng.choice(len(df), size=int(2.2 * n), replace=False)].copy()
    arm = hash_assign(sample.user_id.to_numpy(), salt)
    sample["arm"] = np.where(arm == 1, "treatment", "control")
    ctrl = sample[sample.arm == "control"].head(n)
    trt = sample[sample.arm == "treatment"].head(n).copy()
    if lift:
        accept_rate = df.post_accepted.sum() / max(df.post_answers.sum(), 1)
        new = inject_count(trt.post_answers.to_numpy(float), lift, rng)
        extra = (new - trt.post_answers.to_numpy()).astype(int)
        trt["post_answers"] = new.astype(int)
        trt["post_accepted"] += rng.binomial(np.maximum(extra, 0), accept_rate)
        trt["post_active"] = ((trt.post_answers + trt.post_comments) > 0).astype(int)
    if drop:
        trt = trt[rng.random(len(trt)) >= drop]   # simulated logging bug
    return pd.concat([ctrl, trt], ignore_index=True)


def main() -> None:
    # One draw of a scenario can hit a 5% guardrail false alarm (guardrails are uncorrected by
    # design, D-033), so each scenario is replicated and the published readout states how often
    # each decision occurred. The published draw is the first that matches the scenario's
    # intent, and the readout says so (DECISION D-049).
    a = parser(__doc__, runs=200).parse_args()
    runs = min(a.runs, 20) if a.quick else a.runs
    df, source = load_population(a.data)
    out = OUT if source == "real" else results_dir(source) / "example_readouts"   # D-055
    out.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(a.seed).spawn(len(SCENARIOS))   # scenarios do not share draws
    rows = []
    for (name, n, lift, drop, mde, expected), ss in zip(SCENARIOS, seeds, strict=True):
        n = min(n, len(df) // 5 if a.quick else int(len(df) / 2.2))   # build() samples 2.2 n users
        cfg = ExperimentConfig.from_dict({**CONFIG, "experiment": {**CONFIG["experiment"],
                                                                   "name": name, "mde": mde}})
        decisions, shown, shown_at = [], None, None
        for i, child in enumerate(ss.spawn(runs)):
            readout = analyze(build(df, n, lift, drop, np.random.default_rng(child), f"example-{name}-{i}"), cfg)
            decisions.append(readout.decision.value)
            if shown is None and readout.decision.value == expected:
                shown, shown_at = readout, i
        freq = pd.Series(decisions).value_counts(normalize=True)
        dist = ", ".join(f"{d} {p:.0%}" for d, p in freq.items())
        if shown is None:
            shown, shown_at = readout, runs - 1
        text = shown.to_markdown() + (
            f"\n_Data source: {source}. Injected lift: {lift:+.1%} on answers_posted; "
            f"treatment users dropped: {drop:.0%}; n per arm: {n:,}._\n\n"
            f"_Across {runs} independent draws of this scenario the decisions were: {dist}. "
            f"This readout is draw #{shown_at}, the first whose decision matches the scenario "
            f"({expected})._\n")
        (out / f"{name}.md").write_text(text)
        hit = freq.get(expected, 0.0)
        flag = "ok" if hit >= 0.5 else f"UNEXPECTED (wanted {expected} in most draws)"
        print(f"{name:24s} n={n:>6,}  {expected:12s} in {hit:5.0%} of {runs} draws  [{dist}]  {flag}")
        rows += [{"scenario": name, "n_per_arm": n, "lift": lift, "drop_fraction": drop, "mde": mde,
                  "expected": expected, "decision": d, "share": float(p), "runs": runs,
                  "data_source": source} for d, p in freq.items()]
    pd.DataFrame(rows).to_csv(results_dir(source) / "example_readouts_summary.csv", index=False)


if __name__ == "__main__":
    main()
