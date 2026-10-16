"""Shared plumbing for the validation experiments.

Every experiment: loads the population, runs a harness, writes
results/<name>_raw.parquet (every run) and results/<name>_summary.csv, and
prints the summary. Results are labelled with the data source so synthetic
numbers can never be mistaken for findings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from expkit.io import ParquetSource, SyntheticSource

ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT / "data" / "users_metrics.parquet"
RESULTS = ROOT / "results"


def parser(description: str, runs: int) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--runs", type=int, default=runs)
    p.add_argument("--jobs", type=int, default=1, help="-1 = all cores (needs joblib)")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--data", default=str(REAL_DATA),
                   help="parquet with the per-user table, or 'synthetic'")
    p.add_argument("--quick", action="store_true", help="tiny run for smoke tests / CI")
    return p


def load_population(data: str) -> tuple[pd.DataFrame, str]:
    if data != "synthetic" and Path(data).exists():
        return ParquetSource(data).load(), "real"
    print("WARNING: real extract not found; using SYNTHETIC data. "
          "Numbers below are NOT findings.", file=sys.stderr)
    return SyntheticSource(n_users=200_000).load(), "synthetic"


def results_dir(source: str) -> Path:
    """Real results go to results/; anything else to results/synthetic/, so a smoke run
    can never overwrite real findings (DECISION D-055)."""
    out = RESULTS if source == "real" else RESULTS / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save(name: str, raw: pd.DataFrame, summary: pd.DataFrame, source: str, meta: dict) -> None:
    out = results_dir(source)
    raw.assign(data_source=source).to_parquet(out / f"{name}_raw.parquet", index=False)
    summary.assign(data_source=source).to_csv(out / f"{name}_summary.csv", index=False)
    (out / f"{name}_meta.json").write_text(json.dumps({"data_source": source, **meta}, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.to_string(index=False))
