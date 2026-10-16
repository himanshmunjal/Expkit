"""Build docs/report.html (the shareable one-page report) from report.src.html.

The figure explanations come from expkit.viz.FIGURES, the same text the
Streamlit app shows, and the calibration ledger and run count are computed
from results/, so the page cannot drift from the data. Run: make report
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from expkit.viz import FIGURES

DOCS = Path(__file__).resolve().parent
RESULTS = DOCS.parent / "results"

QUESTIONS = {  # section heading for each figure: the question it answers
    "01_aa_false_positive_rates": "Which tests keep their promise of 5% false positives?",
    "02_ratio_method_fix": "Fixing ratio metrics: the delta method vs the jackknife",
    "03_cuped_variance_reduction": "How much variance does pre-period data remove?",
    "04_power_curves": "How big a lift can you actually detect?",
    "05_peeking": "What happens when you check results every day?",
    "06_msprt_tau_sweep": "Tuning mSPRT so it can detect anything at all",
    "07_capping_tradeoff": "Capping outliers vs CUPED",
    "08_srm_and_multiplicity": "Two safety checks: SRM and multiple testing",
    "09_readout_decisions": "Does the decision engine make the right call?",
}
DECISIONS = {  # what the toolkit does about it, with the DECISION.md entry
    "01_aa_false_positive_rates": "Ship decisions are gated on the false-positive rate at alpha; the "
                                  "percentile bootstrap is not recommended below 10k users per arm on "
                                  "heavy-tailed metrics. (D-053, D-054)",
    "02_ratio_method_fix": "The jackknife ratio test is now the default for ratio metrics. (D-050)",
    "03_cuped_variance_reduction": "CUPED is the default for metrics with pre-period data. (D-025, D-054)",
    "04_power_curves": "Pick the primary metric with the power calculator before launch; answers posted "
                       "is a weak primary metric for small or medium experiments. (D-049)",
    "05_peeking": "If you will look early, use O'Brien-Fleming; use mSPRT only when looks can't be "
                  "planned. (D-051)",
    "06_msprt_tau_sweep": "mSPRT's prior width is set in standard errors at the planned final look, "
                          "default 3. (D-051)",
    "07_capping_tradeoff": "No capping by default; CUPED gives the power without the bias. (D-052)",
    "08_srm_and_multiplicity": "SRM gates every readout. Holm controls several decision metrics; "
                               "guardrails stay uncorrected on purpose. (D-027, D-028, D-033)",
    "09_readout_decisions": "Each example readout states how often its verdict occurred across the "
                            "re-runs, so no single lucky draw is presented as typical. (D-049)",
}
LEDGER_ROWS = [  # (metric, method) pairs shown in the calibration ledger
    ("answers_posted", "welch"), ("answers_posted", "cuped"), ("answers_posted", "bootstrap"),
    ("answers_posted", "mannwhitney"), ("comments_posted", "welch"), ("comments_posted", "bootstrap"),
    ("active_28d", "welch"), ("accepted_answer_rate", "delta"), ("accepted_answer_rate", "jackknife_ratio"),
    ("accepted_answer_rate", "naive_ratio"),
]


def ledger(aa: pd.DataFrame) -> tuple[str, str]:
    from expkit.viz.charts import label

    runs = int(aa.runs.max())
    sizes = sorted(aa.n_per_arm.unique())
    head = "".join(f'<th class="num">{n:,} / arm</th>' for n in sizes)
    rows = []
    for metric, method in LEDGER_ROWS:
        cells = []
        for n in sizes:
            hit = aa[(aa.metric == metric) & (aa.method == method) & (aa.n_per_arm == n)]
            if hit.empty:
                cells.append('<td class="num"><span class="chip na">not run</span></td>')
                continue
            row = hit.iloc[0]
            r = float(row.rejection_rate)
            # Same rule as calibrated_at_alpha (D-053): the rate's 95% Wilson CI contains 5%.
            cls = ("ok" if row.rejection_ci_low <= 0.05 <= row.rejection_ci_high
                   else "low" if row.rejection_ci_high < 0.05 else "high")
            cells.append(f'<td class="num"><span class="chip {cls}">{r:.2%}</span></td>')
        rows.append(f"<tr><td>{html.escape(label(metric))}</td><td>{html.escape(label(method))}</td>"
                    + "".join(cells) + "</tr>")
    table = (f'<table><thead><tr><th>Metric</th><th>Method</th>{head}</tr></thead>'
             f"<tbody>{''.join(rows)}</tbody></table>")
    return table, f"{runs:,}"


def findings() -> str:
    out = []
    for i, spec in enumerate(FIGURES, start=1):
        out.append(f"""
<hr>
<section class="finding wide" id="f{i}" aria-labelledby="h{i}">
  <div class="col finding-head"><span class="no">Fig. {i}</span>
    <h2 id="h{i}">{html.escape(QUESTIONS[spec.name])}</h2></div>
  <figure><img src="figures/{spec.name}.png" alt="{html.escape(spec.title)}: {html.escape(spec.takeaway)}"
       ><figcaption>Fig. {i} · {html.escape(spec.title)}</figcaption></figure>
  <div class="col">
    <div class="notes">
      <div><h3>How to read it</h3><p>{html.escape(spec.how_to_read)}</p></div>
      <div><h3>What it shows</h3><p>{html.escape(spec.takeaway)}</p></div>
    </div>
    <p class="decision"><b>What the toolkit does about it.</b> {html.escape(DECISIONS[spec.name])}</p>
  </div>
</section>""")
    return "\n".join(out)


def main() -> None:
    missing = [s.name for s in FIGURES if not (DOCS / "figures" / f"{s.name}.png").exists()]
    if missing:
        raise SystemExit(f"missing figures {missing}; run make figures first")
    aa = pd.read_csv(RESULTS / "aa_calibration_summary.csv")
    if (aa.data_source != "real").any():
        raise SystemExit("results/ is not real data; refusing to build the report")
    tests = sum(len(pd.read_parquet(p)) for p in RESULTS.glob("*_raw.parquet"))
    table, band = ledger(aa)
    page = (DOCS / "report.src.html").read_text()
    page = (page.replace("<!--LEDGER-->", table).replace("<!--BAND-->", band)
            .replace("<!--FINDINGS-->", findings())
            .replace("<b>~500,000</b>", f"<b>{tests:,}</b>"))
    (DOCS / "report.html").write_text(page)
    print(f"wrote {DOCS / 'report.html'} ({tests:,} simulated tests)")


if __name__ == "__main__":
    main()
