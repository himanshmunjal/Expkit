"""Charts of the validation results. Each function takes a summary table from
results/ and returns a matplotlib Figure, so the report builder and the
Streamlit app draw identical charts.

Conventions: titles state the finding, subtitles state what is plotted; each
method keeps one colour everywhere (Okabe-Ito, colour-blind safe); the 5%
false-positive target is drawn with the band of rates consistent with it
given the number of simulated runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator, PercentFormatter
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError("install the extra: pip install 'expkit[viz]'") from e

INK, MUTED, GRID, TARGET = "#1f2328", "#59636e", "#e6e8eb", "#b0b7bf"
COLORS = {
    "welch": "#0072B2", "cuped": "#009E73", "cuped_zero": "#56B4E9", "mannwhitney": "#999999",
    "bootstrap": "#E69F00", "bootstrap_ratio": "#E69F00", "delta": "#D55E00",
    "jackknife_ratio": "#009E73", "naive_ratio": "#CC79A7",
    "winsor_0.95": "#882255", "winsor_0.99": "#CC79A7", "winsor_0.999": "#E8A3C9",
    "fixed_horizon": "#59636e", "naive_peeking": "#D55E00", "obf": "#0072B2", "pocock": "#56B4E9",
    "msprt": "#009E73",
}
LABELS = {
    "welch": "Welch t-test", "cuped": "CUPED", "cuped_zero": "CUPED + zero flag",
    "mannwhitney": "Mann-Whitney", "bootstrap": "Percentile bootstrap",
    "bootstrap_ratio": "User bootstrap", "delta": "Delta method", "jackknife_ratio": "Jackknife (new default)",
    "naive_ratio": "Naive per-user ratio", "winsor_0.95": "Cap at p95", "winsor_0.99": "Cap at p99",
    "winsor_0.999": "Cap at p99.9", "fixed_horizon": "Fixed horizon (one look)",
    "naive_peeking": "Naive daily peeking", "obf": "O'Brien-Fleming", "pocock": "Pocock", "msprt": "mSPRT",
    "answers_posted": "Answers posted", "comments_posted": "Comments posted", "active_28d": "Active (28d)",
    "accepted_answer_rate": "Accepted-answer rate",
    "ship": "Ship", "do_not_ship": "Do not ship", "extend": "Extend", "invalid": "Invalid (SRM)",
}
DECISION_COLORS = {"ship": "#009E73", "do_not_ship": "#D55E00", "extend": "#E69F00", "invalid": "#59636e"}
PCT = PercentFormatter(1.0, decimals=0)
PCT1 = PercentFormatter(1.0, decimals=1)


def label(key: str) -> str:
    return LABELS.get(key, key)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 9, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.titlecolor": INK, "axes.titleweight": "bold",
    })


def _figure(title: str, subtitle: str, ncols: int = 1, nrows: int = 1, width: float = 9.0,
            height: float = 4.2, **kw) -> tuple[Figure, np.ndarray]:
    _style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False, **kw)
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    return fig, axes


def _target_band(ax, runs: int, alpha: float = 0.05, horizontal: bool = True) -> None:
    """Shade the rates a correctly calibrated test would produce in `runs` simulations."""
    lo, hi = stats.binom.ppf([0.025, 0.975], runs, alpha) / runs
    ax.axhspan(lo, hi, color=TARGET, alpha=0.25, lw=0, zorder=0)
    ax.axhline(alpha, color=TARGET, lw=1.2, ls="--", zorder=1)


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Nudge label positions apart (keeping order) so neighbours are at least min_gap apart."""
    order = np.argsort(values)
    pos = np.asarray(values, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out.tolist()


# ---------------------------------------------------------------- 1. A/A calibration
def aa_false_positive_rates(aa: pd.DataFrame) -> Figure:
    metrics = ["answers_posted", "comments_posted", "active_28d", "accepted_answer_rate"]
    runs = int(aa.runs.max())
    fig, axes = _figure(
        "With no real effect, which tests stay at 5% false positives?",
        f"A/A tests on real users ({runs:,} per point). Grey band: rates consistent with a correct 5% test. "
        "Bars: 95% CI.", ncols=4, width=12, height=4.0, sharey=True)
    for ax, metric in zip(axes[0], metrics, strict=True):
        d = aa[aa.metric == metric]
        _target_band(ax, runs)
        methods = list(dict.fromkeys(d.method))
        offsets = np.linspace(-0.12, 0.12, len(methods))
        for off, m in zip(offsets, methods, strict=True):
            g = d[d.method == m].sort_values("n_per_arm")
            x = np.log10(g.n_per_arm) + off
            ax.errorbar(x, g.rejection_rate, yerr=[g.rejection_rate - g.rejection_ci_low,
                                                   g.rejection_ci_high - g.rejection_rate],
                        fmt="o-", ms=5, lw=1.4, capsize=0, color=COLORS.get(m, MUTED), label=label(m))
        ax.set_title(label(metric))
        ax.set_xticks([2, 3, 4], ["100", "1k", "10k"])
        ax.set_xlabel("users per arm")
        ax.yaxis.set_major_formatter(PCT)
        ax.legend(loc="upper right", fontsize=8)
    axes[0, 0].set_ylabel("false-positive rate")
    axes[0, 0].set_ylim(0, max(0.22, float(aa.rejection_ci_high.max()) + 0.01))
    return fig


def ratio_method_fix(aa: pd.DataFrame, ratio: pd.DataFrame) -> Figure:
    """False positives (A/A runs from both experiments) and bias under a real lift."""
    cols = ["method", "n_per_arm", "runs", "rejection_rate"]
    a = aa[aa.metric == "accepted_answer_rate"][cols]
    r = ratio[ratio.lift == 0][cols]
    d = (pd.concat([a, r]).sort_values("runs", ascending=False)   # keep the larger run where both exist
         .drop_duplicates(["method", "n_per_arm"]).sort_values("n_per_arm"))
    fig, axes = _figure(
        "The delta method over-rejects when a few users hold most answers; the jackknife fixes it",
        "Accepted-answer rate. Only ~12% of users answer, so small samples are dominated by a handful of "
        "heavy answerers. Left: A/A (2,000-5,000 runs per point). Right: +5% true lift, 10k/arm.",
        ncols=2, width=12, height=4.4, gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0, 0]
    _target_band(ax, int(d.runs.min()))
    methods = ["delta", "bootstrap_ratio", "jackknife_ratio", "naive_ratio"]
    for m in methods:
        g = d[d.method == m]
        ax.plot(g.n_per_arm, g.rejection_rate, "o-", color=COLORS[m], lw=2.4 if m == "jackknife_ratio" else 1.5,
                ms=6, label=label(m), zorder=3 if m == "jackknife_ratio" else 2)
        first = g.iloc[0]
        ax.annotate(f"{first.rejection_rate:.1%}", (first.n_per_arm, first.rejection_rate),
                    textcoords="offset points", xytext=(-8, 0), ha="right", va="center",
                    fontsize=8.5, color=COLORS[m])
    ticks = sorted(d.n_per_arm.unique())
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{n:,}" for n in ticks])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] / 2.2, ticks[-1] * 1.3)
    ax.set_xlabel("users per arm")
    ax.set_ylabel("false-positive rate")
    ax.yaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    ax.set_title("False positives with no real effect")

    ax = axes[0, 1]
    b = ratio[(ratio.lift > 0) & (ratio.n_per_arm == ratio.n_per_arm.max())].set_index("method").loc[methods]
    rel = (b.bias / b.true_effect).to_numpy()
    y = np.arange(len(methods))
    ax.barh(y, rel, color=[COLORS[m] for m in methods], height=0.55)
    for yi, v in zip(y, rel, strict=True):
        ax.text(max(v, 0) + 0.012, yi, f"{v:+.0%}", va="center", ha="left", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [label(m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(-0.08, 0.28)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_title("Bias of the estimated lift")
    ax.grid(axis="y", visible=False)
    ax.text(0.5, -0.2, "Naive is calibrated, but it estimates the average user's\nratio, not the ratio of totals.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
    return fig


# ---------------------------------------------------------------- 2. CUPED
def cuped_variance_reduction(cuped: pd.DataFrame) -> Figure:
    d = (cuped[(cuped.method == "cuped") & (cuped.lift == 0)]
         .drop_duplicates("metric").set_index("metric")
         .loc[["comments_posted", "answers_posted", "active_28d"]])
    fig, axes = _figure(
        "CUPED removes 63-74% of the variance for activity counts, matching theory",
        "Variance reduction from adjusting for each user's pre-period activity. Theory: squared "
        "pre/post correlation (\u03c1\u00b2). Observed: simulated experiments (within 1.5 points).",
        ncols=2, width=11, height=4.0)
    ax = axes[0, 0]
    y = np.arange(len(d))
    ax.barh(y + 0.18, d.theoretical_r2, height=0.34, color="#bcd9cf", label="theory (\u03c1\u00b2)")
    ax.barh(y - 0.18, d.empirical_vr, height=0.34, color=COLORS["cuped"], label="observed")
    for yi, v in zip(y, d.empirical_vr, strict=True):
        ax.text(v + 0.01, yi - 0.18, f"{v:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [label(m) for m in d.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_title("Variance removed")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)

    ax = axes[0, 1]
    n = 1 / (1 - d.empirical_vr)
    ax.barh(y, n, color=COLORS["cuped"], height=0.5)
    for yi, v in zip(y, n, strict=True):
        ax.text(v + 0.05, yi, f"{v:.1f}x", va="center", fontsize=9, color=INK)
    ax.axvline(1, color=TARGET, ls="--", lw=1)
    ax.set_yticks(y, [label(m) for m in d.index])
    ax.invert_yaxis()
    ax.set_xlim(0, max(n) * 1.2)
    ax.set_title("Same power with this many times fewer users")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.35)   # room for the right panel's labels
    return fig


# ---------------------------------------------------------------- 3. Power
def power_curves(lift: pd.DataFrame) -> Figure:
    from matplotlib.lines import Line2D

    metrics = ["answers_posted", "active_28d", "accepted_answer_rate"]
    ns = sorted(lift.n_per_arm.unique())
    shades = dict(zip(ns, [0.3, 0.6, 1.0], strict=False))
    styles = {"cuped": "o-", "jackknife_ratio": "o-", "welch": "s--", "delta": "s--"}
    fig, axes = _figure(
        "Power: small lifts on heavy-tailed counts are hard to detect, even with CUPED",
        "Share of simulated experiments that detected a true injected lift (2,000 runs per point). "
        "Darker = more users per arm. Dotted line: the usual 80% power target.",
        ncols=3, width=12, height=4.6, sharey=True)
    for ax, metric in zip(axes[0], metrics, strict=True):
        d = lift[lift.metric == metric]
        methods = ["welch", "cuped"] if metric != "accepted_answer_rate" else ["delta", "jackknife_ratio"]
        for m in methods:
            for n in ns:
                g = d[(d.method == m) & (d.n_per_arm == n)].sort_values("lift")
                ax.plot(g.lift, g.power, styles[m], ms=4, lw=1.6, color=COLORS[m], alpha=shades[n])
        ax.axhline(0.8, color=INK, lw=0.9, ls=":")
        ax.set_title(label(metric))
        ax.set_xlabel("true relative lift")
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("power")
    handles = [Line2D([], [], color=COLORS[m], marker=styles[m][0], ls=styles[m][1:], label=label(m))
               for m in ["welch", "cuped", "delta", "jackknife_ratio"]]
    handles += [Line2D([], [], color=MUTED, alpha=shades[n], lw=3, label=f"{n:,} users/arm") for n in ns]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.885))
    return fig


# ---------------------------------------------------------------- 4. Peeking
def _cumulative_rejections(raw: pd.DataFrame, rule: str, grid: np.ndarray) -> np.ndarray:
    stop = np.where(raw[f"{rule}_reject"], raw[f"{rule}_stop_fraction"], np.inf)
    return np.array([(stop <= g + 1e-9).mean() for g in grid])


def peeking_inflation(raw: pd.DataFrame, summary: pd.DataFrame) -> Figure:
    from matplotlib.lines import Line2D

    rules = ["naive_peeking", "pocock", "obf", "msprt"]
    short = {"fixed_horizon": "Fixed", "naive_peeking": "Naive peeking", "obf": "OBF", "pocock": "Pocock",
             "msprt": "mSPRT"}
    metrics = ["answers_posted", "active_28d"]
    fig, axes = _figure(
        "Checking a test every day turns a 5% false-positive rate into 24-28%",
        "Left two panels: A/A experiments with 28 daily looks, cumulative false positives by day. "
        "Right: under a real lift, power vs how much of the planned sample each rule used.",
        ncols=3, width=12.5, height=4.5)
    grid = np.linspace(1 / 28, 1, 28)
    for ax, metric in zip(axes[0, :2], metrics, strict=True):
        a = raw[(raw.metric == metric) & (raw.lift == 0)]
        _target_band(ax, len(a))
        curves = {r: _cumulative_rejections(a, r, grid) for r in rules}
        ends = [c[-1] for c in curves.values()]
        for r, ypos in zip(rules, _spread(ends, 0.014), strict=True):
            ax.plot(grid * 28, curves[r], lw=2, color=COLORS[r], label=label(r))
            ax.annotate(f"{curves[r][-1]:.1%}", (28, ypos), textcoords="offset points", xytext=(4, 0),
                        va="center", fontsize=8.5, color=COLORS[r])
        ax.set_title(f"{label(metric)}: A/A")
        ax.set_xlabel("day of experiment")
        ax.set_xlim(0, 33)
        ax.set_ylim(0, 0.32)
        ax.yaxis.set_major_formatter(PCT)
    axes[0, 0].set_ylabel("cumulative false-positive rate")
    axes[0, 0].legend(loc="upper left")

    ax = axes[0, 2]
    s_ = summary[summary.lift > 0]
    markers = {"answers_posted": "o", "active_28d": "s"}
    for metric in metrics:
        g = s_[s_.metric == metric]
        for _, row in g.iterrows():
            ax.scatter(row.mean_sample_fraction, row.rejection_rate, s=55, marker=markers[metric],
                       color=COLORS[row.rule], zorder=3, edgecolor="white")
            ax.annotate(short[row.rule], (row.mean_sample_fraction, row.rejection_rate),
                        textcoords="offset points", xytext=(-7, 0), ha="right", va="center",
                        fontsize=7.5, color=COLORS[row.rule])
    ax.set_title("Under a real lift: power vs sample used")
    ax.set_xlabel("average share of planned sample used")
    ax.set_ylabel("power")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlim(0.45, 1.04)
    ax.set_ylim(0, 0.75)
    ax.legend(handles=[Line2D([], [], marker=markers[m], color=MUTED, ls="",
                              label=f"{label(m)} (+{s_[s_.metric == m].lift.iloc[0]:.0%})") for m in metrics],
              loc="upper right", fontsize=8)
    ax.text(0.47, 0.02, "Naive peeking's 'power' is inflated by its\nfalse positives; it is not a usable rule.",
            fontsize=7.5, color=MUTED, va="bottom")
    return fig


def msprt_tau_sweep(sweep: pd.DataFrame) -> Figure:
    fig, axes = _figure(
        "mSPRT needs tau of about 3 standard errors; the old setting (~0.5) never detected anything",
        "tau = width of mSPRT's prior on the effect, in units of the SE at the final look. "
        "False positives stay below 1.5% at every tau.", ncols=2, width=11, height=4.0)
    for ax, (is_aa, title) in zip(axes[0], [(False, "Power under a real lift"),
                                             (True, "False-positive rate (A/A)")], strict=True):
        for metric, marker in [("answers_posted", "o"), ("active_28d", "s")]:
            g = sweep[(sweep.metric == metric) & ((sweep.lift == 0) == is_aa)].sort_values("tau_se")
            lift = g.lift.iloc[0]
            ax.plot(g.tau_se, g.rejection_rate, marker + "-", color=COLORS["msprt"],
                    alpha=1.0 if metric == "active_28d" else 0.55, lw=1.8,
                    label=f"{label(metric)}" + ("" if is_aa else f" (+{lift:.0%})"))
        ax.axvline(3, color=INK, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 3, 5, 10], ["0.5", "1", "2", "3", "5", "10"])
        ax.minorticks_off()
        ax.set_xlabel("tau (SEs at the final look)")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PCT1 if is_aa else PCT)
        ax.legend(loc="upper left")
    axes[0, 0].set_ylim(0, 0.25)
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(0.05))
    axes[0, 1].axhline(0.05, color=TARGET, lw=1.2, ls="--")
    axes[0, 1].text(10, 0.051, "5% target", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    axes[0, 1].set_ylim(0, 0.06)
    for ax in axes[0]:
        ax.text(3.1, 0.003 if ax is axes[0, 1] else 0.01, "default = 3", fontsize=8.5, color=INK)
    return fig


# ---------------------------------------------------------------- 5. Capping
def capping_tradeoff(w: pd.DataFrame) -> Figure:
    order = ["welch", "cuped", "winsor_0.999", "winsor_0.99", "winsor_0.95"]
    d = w[w.lift > 0]
    fig, axes = _figure(
        "Capping outliers buys power by measuring the wrong thing; CUPED gets the power honestly",
        f"Injected +{d.lift.iloc[0]:.0%} lift, {int(d.n_per_arm.iloc[0]):,} users per arm, "
        f"{int(d.runs.iloc[0]):,} runs. Coverage = how often the 95% CI contains the true lift.",
        ncols=3, width=12, height=4.3, sharey=True)
    panels = [("rejection_rate", "Power", (0, 1.12)), ("relative_bias", "Bias of the lift estimate", (-1.02, 0.12)),
              ("coverage", "95% CI coverage of true lift", (0, 1))]
    y = np.arange(len(order))
    for ax, (col, title, lim) in zip(axes[0], panels, strict=True):
        for k, (metric, dy) in enumerate([("answers_posted", -0.18), ("comments_posted", 0.18)]):
            g = d[d.metric == metric].set_index("method").loc[order]
            vals = g[col].fillna(0).to_numpy()
            ax.barh(y + dy, vals, height=0.34, color=[COLORS[m] for m in order],
                    alpha=1.0 if k == 0 else 0.55, label=label(metric))
            for yi, v in zip(y + dy, vals, strict=True):
                ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.0%}" if col == "relative_bias" else f"{v:.0%}",
                        va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
        if col == "coverage":
            ax.axvline(0.95, color=TARGET, ls="--", lw=1.2)
        if col == "relative_bias":
            ax.axvline(0, color=INK, lw=0.8)
            ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0])
        else:
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(*lim)
        ax.xaxis.set_major_formatter(PCT)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_yticks(y, [label(m) for m in order])
    axes[0, 0].invert_yaxis()
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=a) for a in (1.0, 0.55)],
               labels=["Answers posted (upper, darker bar)", "Comments posted (lower, lighter bar)"],
               loc="lower center", ncol=2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.885))
    return fig


# ---------------------------------------------------------------- 6. Guardrails on validity
def srm_and_multiplicity(srm: pd.DataFrame, mt: pd.DataFrame) -> Figure:
    fig, axes = _figure(
        "Two safety checks: SRM only catches large losses, and 20 metrics need a correction",
        "Left: how often the p < 0.001 sample-ratio check flags a bug that drops a share of treatment "
        "users (20k users). Right: chance of at least one false 'win' among 20 null metrics.",
        ncols=2, width=12, height=4.0)
    ax = axes[0, 0]
    ax.plot(srm.drop_fraction, srm.detection_rate, "o-", color=INK, lw=2)
    for _, r in srm.iterrows():
        if r.drop_fraction >= 0.02:
            ax.annotate(f"{r.detection_rate:.0%}", (r.drop_fraction, r.detection_rate),
                        textcoords="offset points", xytext=(-6, 6), ha="right", fontsize=9)
    ax.set_xlabel("share of treatment users lost by the bug")
    ax.set_ylabel("detection rate")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1)
    ax.set_title("SRM detection vs size of the logging bug")

    ax = axes[0, 1]
    order = ["none", "bonferroni", "holm", "bh"]
    names = {"none": "None", "bonferroni": "Bonferroni", "holm": "Holm", "bh": "BH (FDR)"}
    g = mt.set_index("correction").loc[order]
    colors = ["#D55E00", "#0072B2", "#0072B2", "#56B4E9"]
    ax.barh(range(4), g.family_wise_error, color=colors, height=0.55)
    for i, v in enumerate(g.family_wise_error):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)
    ax.axvline(0.05, color=TARGET, ls="--", lw=1.2)
    ax.set_yticks(range(4), [names[c] for c in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_title("Family-wise false-positive rate, 20 metrics")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.3)   # room for the right panel's labels
    return fig


# ---------------------------------------------------------------- 7. Decisions
def readout_decisions(ex: pd.DataFrame) -> Figure:
    order = ["1_clear_win", "2_well_powered_null", "3_inconclusive", "4_srm_failure"]
    names = {"1_clear_win": "Clear win\n(+20%, 80k/arm)", "2_well_powered_null": "Well-powered null\n(0%, 80k/arm)",
             "3_inconclusive": "Inconclusive\n(+3%, 1.5k/arm)", "4_srm_failure": "SRM failure\n(5% of treatment lost)"}
    runs = int(ex.runs.max())
    fig, axes = _figure(
        "The decision rules reach the intended verdict in 88-100% of repeated experiments",
        f"Each scenario re-run {runs} times on fresh samples of real users. Misses are guardrail false "
        "alarms, which are deliberately not corrected (a missed harm costs more).", width=10, height=3.9)
    ax = axes[0, 0]
    for i, sc in enumerate(order):
        g = ex[ex.scenario == sc].set_index("decision").share
        left = 0.0
        for dec in ["ship", "do_not_ship", "extend", "invalid"]:
            v = float(g.get(dec, 0.0))
            if v <= 0:
                continue
            ax.barh(i, v, left=left, color=DECISION_COLORS[dec], height=0.6,
                    label=label(dec) if i == 0 or dec not in ax.get_legend_handles_labels()[1] else None)
            if v >= 0.06:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")
            left += v
    ax.set_yticks(range(4), [names[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("share of re-runs")
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=True))
    ax.legend(uniq.values(), uniq.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig

