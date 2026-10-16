"""Charts of the validation results (needs the [viz] extra: matplotlib).

    python -m expkit.viz --results results --out docs/figures

`FIGURES` maps each figure's file name to the chart function and the result
tables it reads, so the report and the app stay in sync with one list.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from expkit.viz import charts


@dataclass(frozen=True)
class FigureSpec:
    name: str
    draw: Callable[..., object]
    inputs: tuple[str, ...]          # files under results/
    title: str = ""                  # short name, for tables of contents
    how_to_read: str = ""            # what is plotted
    takeaway: str = ""               # what it means (numbers checked against results/)


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        "01_aa_false_positive_rates", charts.aa_false_positive_rates, ("aa_calibration_summary.csv",),
        "False positives by method and sample size",
        "Each point is 5,000 A/A experiments: two random groups of real users, no change applied. A correct "
        "test at alpha = 5% should flag about 5% of them; the grey band is the range a correct test would "
        "land in by chance.",
        "On heavy-tailed counts (answers, comments) at small samples, Welch is conservative (1.9-3.5% at "
        "n <= 1,000: safe, but it loses power) and the percentile bootstrap is anti-conservative (8.5-9.2% at "
        "n = 100). Everything is calibrated by 10k users per arm. The delta method on the accepted-answer "
        "rate is the outlier: 20.6% at n = 100."),
    FigureSpec(
        "02_ratio_method_fix", charts.ratio_method_fix,
        ("aa_calibration_summary.csv", "ratio_metrics_summary.csv"),
        "Ratio metrics: delta method vs jackknife",
        "Accepted-answer rate = accepted answers / answers, summed per arm. Left: false-positive rate with no "
        "real effect. Right: bias of the estimated lift when a true +5% lift is injected.",
        "The delta method's variance formula underestimates how much one heavy answerer moves the ratio, so "
        "it flags 20.6% of null experiments at 100 users per arm and 10.4% at 1,000. The leave-one-user-out "
        "jackknife with a t reference brings this to 5.9% and 6.4% and is now the default. The naive "
        "per-user ratio looks calibrated but answers a different question and is 18% biased."),
    FigureSpec(
        "03_cuped_variance_reduction", charts.cuped_variance_reduction, ("cuped_summary.csv",),
        "CUPED variance reduction",
        "CUPED subtracts the part of each user's outcome predicted by their own pre-experiment activity. "
        "Theory says it removes rho-squared of the variance (rho = pre/post correlation).",
        "Observed reduction matches theory within 1.5 points: 63% for answers and 74% for comments, so the "
        "same test needs 2.7x and 3.8x fewer users. For 'active in 28 days' the pre-period predicts little "
        "(4%), so CUPED adds almost nothing there."),
    FigureSpec(
        "04_power_curves", charts.power_curves, ("injected_lift_summary.csv",),
        "Power curves",
        "Share of 2,000 simulated experiments that detected a known injected lift, by lift size and users "
        "per arm. Dotted line: the conventional 80% power target.",
        "Binary activity is easy: a 5% lift is found about 80% of the time at 20k users per arm. Answers are hard: "
        "even with CUPED a +10% lift at 20k per arm is found only 43% of the time (Welch: 20%), so about a "
        "15% lift is needed at that size. On the ratio metric at 1,000/arm the delta method's apparent extra "
        "'power' at tiny lifts (11% at +0.5%) is just its false-positive excess."),
    FigureSpec(
        "05_peeking", charts.peeking_inflation, ("peeking_raw.parquet", "peeking_summary.csv"),
        "Peeking and sequential tests",
        "28 daily looks at an experiment with 10k users per arm. Left panels: share of A/A experiments "
        "declared significant by each day. Right: with a real lift, power against the average share of the "
        "planned sample used before stopping.",
        "Stopping at the first p < 0.05 finds a false winner in 24-28% of null experiments. O'Brien-Fleming "
        "and Pocock keep false positives at or below ~5%, and O'Brien-Fleming keeps about 93% of fixed-horizon "
        "power (30% vs 33%; 46% vs 49%) while stopping 8-13% earlier on average. mSPRT is the most "
        "conservative (0.8-1.4%) and gives up about half the power."),
    FigureSpec(
        "06_msprt_tau_sweep", charts.msprt_tau_sweep, ("msprt_tau_sweep.csv",),
        "Tuning mSPRT",
        "mSPRT mixes over possible effect sizes with a normal prior of width tau. Here tau is expressed in "
        "standard errors of the difference at the planned final look.",
        "The original setting (tau = 5% of the metric mean) was about 0.5 SE for answers, where mSPRT can "
        "never reject. Power peaks at 3-5 SE and false positives stay under 1.5% at every tau, so the "
        "default is now 3 SE."),
    FigureSpec(
        "07_capping_tradeoff", charts.capping_tradeoff, ("winsorization_summary.csv",),
        "Capping outliers vs CUPED",
        "A +20% lift injected at 20k users per arm. Capping (winsorising) replaces values above a percentile "
        "with that percentile. Bias: how far the estimated lift is from the true one. Coverage: how often "
        "the 95% CI contains the true lift (should be 95%).",
        "Capping at p99 reaches the same power as CUPED (91% vs 92% on answers) but underestimates the lift "
        "by 56% and its CI almost never contains the truth (2%). The lift sits mostly in heavy users, which "
        "is exactly what capping removes. CUPED gets the power with 1-2% bias, so capping is not used by "
        "default."),
    FigureSpec(
        "08_srm_and_multiplicity", charts.srm_and_multiplicity,
        ("srm_summary.csv", "multiple_testing_summary.csv"),
        "SRM sensitivity and multiple testing",
        "Left: a simulated logging bug drops a share of treatment users; how often does the chi-square "
        "sample-ratio check (p < 0.001) catch it? Right: 20 metrics with no real effect; how often is at "
        "least one flagged?",
        "SRM catches a 5% loss 61% of the time and a 2% loss only 3%, so a passing check does not prove "
        "logging is clean. Without correction, 20 null metrics produce a false 'win' 63% of the time; Holm "
        "or Bonferroni bring it to 5.9%, and Benjamini-Hochberg (which controls a different error rate) "
        "to 6.2%."),
    FigureSpec(
        "09_readout_decisions", charts.readout_decisions, ("example_readouts_summary.csv",),
        "Ship decisions in repeated experiments",
        "Four scenarios, each re-run 200 times on fresh samples of real users through the full decision "
        "engine (SRM gate, guardrails, primary metric, MDE rule).",
        "The engine reaches the intended verdict 96% (clear win), 94% (well-powered null), 88% "
        "(inconclusive) and 100% (SRM failure) of the time. The misses are mostly guardrail false alarms, "
        "a deliberate choice: guardrails are not multiplicity-corrected because a missed harm costs more."),
)


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def draw(spec: FigureSpec, results: str | Path):
    results = Path(results)
    return spec.draw(*(load(results / f) for f in spec.inputs))


def build_figures(results: str | Path, out: str | Path, formats: tuple[str, ...] = ("png",),
                  require_real: bool = True) -> list[Path]:
    """Draw every figure whose inputs exist. Refuses synthetic inputs unless asked."""
    import matplotlib.pyplot as plt

    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in FIGURES:
        paths = [results / f for f in spec.inputs]
        if not all(p.exists() for p in paths):
            print(f"skip {spec.name}: missing {[p.name for p in paths if not p.exists()]}")
            continue
        if require_real and any((load(p).get("data_source", pd.Series(["real"])) != "real").any()
                                for p in paths):
            raise ValueError(f"{spec.name}: inputs are not real data; pass require_real=False")
        fig = draw(spec, results)
        for fmt in formats:
            target = out / f"{spec.name}.{fmt}"
            fig.savefig(target, dpi=160, bbox_inches="tight")
            written.append(target)
        plt.close(fig)
    return written


__all__ = ["FIGURES", "FigureSpec", "build_figures", "charts", "draw"]
