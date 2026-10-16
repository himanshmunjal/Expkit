"""Public facade. Every integration (CLI, REST, Streamlit, notebooks, other
services) calls these functions rather than reaching into submodules, so the
internals can change without breaking callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from expkit import registry
from expkit._types import MetricSpec, TestResult
from expkit.config import ExperimentConfig
from expkit.report import MetricOutcome, Readout, decide
from expkit.stats.srm import srm_test

# jackknife_ratio, not delta: delta over-rejects on heavy-tailed denominators (DECISION D-050).
DEFAULT_METHOD = {"mean": "welch", "binary": "welch", "ratio": "jackknife_ratio"}


def resolve_method(metric: MetricSpec, preferred: str) -> str:
    """Use ``preferred`` if it fits the metric, else the safe default for its kind."""
    entry = registry.get(preferred)
    if metric.kind in entry.kinds and (not entry.needs_covariate or metric.covariates):
        return preferred
    return DEFAULT_METHOD[metric.kind]


def analyze_metric(
    df: pd.DataFrame,
    metric: MetricSpec,
    *,
    arm_column: str = "arm",
    control: str = "control",
    treatment: str = "treatment",
    method: str = "welch",
    alpha: float = 0.05,
    seed: int | None = 0,
    **options: Any,
) -> TestResult:
    """Run one method on one metric of an assigned frame."""
    arms = df[arm_column]
    c_idx = np.flatnonzero((arms == control).to_numpy())
    t_idx = np.flatnonzero((arms == treatment).to_numpy())
    if c_idx.size == 0 or t_idx.size == 0:
        raise ValueError(f"need rows for both {control!r} and {treatment!r} in {arm_column!r}")
    name = resolve_method(metric, method)
    return registry.run(name, metric.sample(df, c_idx), metric.sample(df, t_idx),
                        kind=metric.kind, alpha=alpha, rng=np.random.default_rng(seed), **options)


def analyze(df: pd.DataFrame, config: ExperimentConfig, seed: int | None = 0) -> Readout:
    """Full readout: SRM check, primary metric, guardrails, decision."""
    if config.arm_column not in df.columns:
        raise KeyError(f"assignment column {config.arm_column!r} not in data")
    arms = df[config.arm_column]
    unknown = set(arms.unique()) - {config.control, config.treatment}
    if unknown:
        raise ValueError(f"unexpected arm labels {sorted(map(str, unknown))}")
    counts = [int((arms == config.control).sum()), int((arms == config.treatment).sum())]
    srm = srm_test(counts, config.expected_split, threshold=config.srm_threshold)

    def run(metric: MetricSpec) -> MetricOutcome:
        method = resolve_method(metric, config.method)
        res = analyze_metric(df, metric, arm_column=config.arm_column, control=config.control,
                          treatment=config.treatment, method=method, alpha=config.alpha,
                          seed=seed, **config.method_options.get(method, {}))
        return MetricOutcome(metric, res)

    primary = run(config.primary)
    guardrails = tuple(run(g) for g in config.guardrails)
    mde_abs = None if config.mde is None else abs(config.mde * primary.result.control_mean)
    decision, reasons = decide(primary, guardrails, srm, mde_abs)
    notes = []
    if primary.result.method != config.method:
        notes.append(f"Primary used {primary.result.method!r}: {config.method!r} does not fit "
                     f"a {config.primary.kind} metric without covariates.")
    return Readout(config.name, decision, reasons, srm, primary, guardrails, mde_abs,
                   config.simulated, tuple(notes))
