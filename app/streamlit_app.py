"""ExpKit Streamlit app: a thin UI over the public API (no statistics here).

    pip install 'expkit[app]'
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import streamlit as st

from expkit import ExperimentConfig, analyze
from expkit.catalog import CATALOG
from expkit.io import ParquetSource, SyntheticSource
from expkit.sim import (
    PeekingSpec,
    SimulationSpec,
    run_peeking_simulation,
    run_simulation,
    summarize,
)
from expkit.sim.peeking import summarize_peeking
from expkit.stats import power as pw

st.set_page_config(page_title="ExpKit", layout="wide")
st.title("ExpKit")
ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT / "data" / "users_metrics.parquet"
RESULTS = ROOT / "results"


@st.cache_data
def population(n: int = 100_000) -> tuple[pd.DataFrame, str]:
    """The real extract when it exists (make extract), else the synthetic twin."""
    if REAL_DATA.exists():
        return ParquetSource(REAL_DATA).load(), "real Stack Overflow extract"
    return SyntheticSource(n_users=n).load(), "SYNTHETIC data (run `make extract` for real data)"


POP, POP_SOURCE = population()
st.caption(f"Experiment design and analysis. Simulations use the {POP_SOURCE} "
           f"({len(POP):,} users); assignment and effects are simulated.")


tab_power, tab_aa, tab_peek, tab_analyze, tab_results = st.tabs(
    ["Sample size / MDE", "A/A simulator", "Peeking demo", "Analyze a CSV", "Validation results"])

with tab_power:
    c1, c2, c3 = st.columns(3)
    binary = c1.checkbox("Binary metric", value=False)
    baseline = c1.number_input("Baseline mean (or rate)", value=0.2 if binary else 1.0,
                               min_value=1e-6, format="%.4f")
    sd = pw.binary_sd(baseline) if binary and baseline < 1 else c1.number_input(
        "Standard deviation", value=3.0, min_value=1e-6)
    mde = c2.number_input("Relative MDE", value=0.03, min_value=1e-4, format="%.4f")
    alpha = c2.number_input("alpha", value=0.05, min_value=1e-4, max_value=0.5)
    power = c2.number_input("power", value=0.8, min_value=0.5, max_value=0.999)
    vr = c3.slider("Variance reduction (CUPED R^2)", 0.0, 0.9, 0.0, 0.05)
    n_c, n_t = pw.required_sample_size(sd, mde, baseline_mean=baseline, relative=True,
                                       alpha=alpha, power=power, variance_reduction=vr)
    c3.metric("Users per arm", f"{n_c:,}")
    if vr:
        base, _ = pw.required_sample_size(sd, mde, baseline_mean=baseline, relative=True,
                                          alpha=alpha, power=power)
        c3.metric("Saved by CUPED", f"{base - n_c:,} per arm", f"-{(base - n_c) / base:.0%}")

with tab_aa:
    metric = st.selectbox("Metric", list(CATALOG))
    n = st.select_slider("Users per arm", [100, 500, 1000, 5000, 10000], value=1000)
    runs = st.slider("Runs", 100, 2000, 500, 100)
    if st.button("Run A/A simulation"):
        m = CATALOG[metric]
        methods = ("jackknife_ratio", "delta", "naive_ratio") if m.kind == "ratio" else ("welch", "cuped",
                                                                      "mannwhitney")
        with st.spinner("simulating..."):
            res = run_simulation(POP, SimulationSpec((m,), n, runs, 0.0, methods))
        st.dataframe(summarize(res)[["method", "runs", "rejection_rate", "rejection_ci_low",
                                     "rejection_ci_high", "ks_uniform_p", "coverage"]])
        st.caption("A calibrated method rejects ~5% of A/A tests and has uniform p-values.")

with tab_peek:
    lift = st.select_slider("True lift", [0.0, 0.01, 0.03, 0.05], value=0.0)
    runs_p = st.slider("Runs ", 200, 3000, 1000, 100)
    if st.button("Run peeking simulation"):
        with st.spinner("simulating 28 daily looks..."):
            res = run_peeking_simulation(POP, PeekingSpec(CATALOG["answers_posted"],
                                                                   5000, 28, runs_p, lift))
        st.dataframe(summarize_peeking(res))
        st.caption("With lift 0, rejection_rate is the false-positive rate of each rule.")

with tab_analyze:
    st.write("Upload a CSV with one row per user and an `arm` column, plus a TOML config "
             "(see `configs/example_experiment.toml`).")
    data = st.file_uploader("Data (CSV)", type="csv")
    cfg = st.file_uploader("Config (TOML)", type="toml")
    if data and cfg:
        try:
            readout = analyze(pd.read_csv(data),
                              ExperimentConfig.from_dict(tomllib.loads(cfg.read().decode())))
            st.markdown(readout.to_markdown())
            st.download_button("Download JSON", readout.to_json(), "readout.json")
        except (KeyError, ValueError) as e:
            st.error(str(e))


@st.cache_data
def figure_png(name: str, stamp: float) -> bytes:
    """Render one result chart to PNG; `stamp` (newest input mtime) busts the cache on new results."""
    import io

    import matplotlib.pyplot as plt

    from expkit.viz import FIGURES, draw

    spec = next(f for f in FIGURES if f.name == name)
    fig = draw(spec, RESULTS)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


with tab_results:
    try:
        from expkit.viz import FIGURES
    except ImportError:
        st.info("Install the charts extra to see results: pip install 'expkit[viz]'")
        FIGURES = ()
    ready = [f for f in FIGURES if all((RESULTS / i).exists() for i in f.inputs)]
    if FIGURES and not ready:
        st.info("No results yet. Run `make experiments` and `make readouts` on the real extract.")
    for spec in ready:
        sources = {pd.read_csv(RESULTS / i).get("data_source", pd.Series(["?"])).iloc[0]
                   for i in spec.inputs if i.endswith(".csv")}
        st.subheader(spec.title)
        if sources != {"real"}:
            st.warning(f"Built from {', '.join(sorted(sources))} data, not the real extract.")
        st.image(figure_png(spec.name, max((RESULTS / i).stat().st_mtime for i in spec.inputs)))
        st.markdown(f"**How to read it.** {spec.how_to_read}\n\n**What it shows.** {spec.takeaway}")
        st.divider()
