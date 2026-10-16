"""Command-line interface: ``expkit <command> --help``.

Every command writes machine-readable output (JSON / CSV / parquet) so it can
be chained from shell scripts, Airflow/cron jobs, or non-Python services.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from expkit import registry
from expkit.api import analyze
from expkit.catalog import CATALOG, get_metric
from expkit.config import ExperimentConfig
from expkit.io import BigQuerySource, SyntheticSource, source_from_uri
from expkit.stats import power as pw


def _write(df: pd.DataFrame, out: str | None) -> None:
    if out is None:
        print(df.to_string(index=False))
        return
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    (df.to_parquet if out.endswith(".parquet") else df.to_csv)(out, index=False)
    print(f"wrote {len(df)} rows to {out}", file=sys.stderr)


def cmd_methods(a) -> None:
    for name in registry.available(a.kind):
        e = registry.get(name)
        print(f"{name:16s} kinds={','.join(sorted(e.kinds)):18s} {e.description}")


def _read_table(path: str) -> pd.DataFrame:
    """Any assigned table works for analysis; only the config's columns are required."""
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)


def cmd_analyze(a) -> None:
    df = _read_table(a.data)
    readout = analyze(df, ExperimentConfig.from_toml(a.config), seed=a.seed)
    text = readout.to_markdown() if a.format == "md" else readout.to_json()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    else:
        print(text)


def cmd_power(a) -> None:
    kw = dict(baseline_mean=a.baseline_mean, relative=a.relative, alpha=a.alpha,
              ratio=a.ratio, variance_reduction=a.variance_reduction)
    sd = a.sd if a.sd is not None else pw.binary_sd(a.baseline_mean)
    out: dict = {"sd": sd, **kw}
    if a.mde is not None:
        out["n_control"], out["n_treatment"] = pw.required_sample_size(sd, a.mde, power=a.power, **kw)
    if a.n is not None:
        out["mde"] = pw.minimum_detectable_effect(sd, a.n, power=a.power, **kw)
    print(json.dumps(out, indent=2))


def cmd_simulate(a) -> None:
    from expkit.sim import SimulationSpec, run_simulation, summarize

    df = source_from_uri(a.data).load()
    spec = SimulationSpec(
        metrics=tuple(get_metric(m) for m in a.metrics.split(",")),
        n_per_arm=a.n_per_arm, n_runs=a.runs, lift=a.lift,
        methods=tuple(a.methods.split(",")), alpha=a.alpha, seed=a.seed, n_jobs=a.jobs,
    )
    results = run_simulation(df, spec)
    _write(summarize(results, a.alpha), a.out)


def cmd_peek(a) -> None:
    from expkit.sim import PeekingSpec, run_peeking_simulation, summarize_peeking

    df = source_from_uri(a.data).load()
    spec = PeekingSpec(get_metric(a.metric), a.n_per_arm, a.looks, a.runs, a.lift,
                       a.alpha, seed=a.seed, n_jobs=a.jobs)
    _write(summarize_peeking(run_peeking_simulation(df, spec)), a.out)


def cmd_synth(a) -> None:
    _write(SyntheticSource(n_users=a.n, seed=a.seed).load(), a.out)


# The public Stack Overflow dataset is a frozen snapshot: answers, comments and
# acceptance votes all end on 2022-09-25 (checked 2026-10-16, DECISION D-048).
DATA_END = date(2022, 9, 25)
PRE_DAYS, POST_DAYS = 56, 28   # must match data/extract.sql


def cmd_extract(a) -> None:
    t0 = date.fromisoformat(a.t0)
    post_end = t0 + timedelta(days=POST_DAYS)
    if post_end > DATA_END and not a.allow_truncated:
        sys.exit(f"T0={t0} puts the post-period end at {post_end}, after the data ends ({DATA_END}); "
                 f"outcomes would be truncated. Use T0 <= {DATA_END - timedelta(days=POST_DAYS)} "
                 "or pass --allow-truncated.")
    out = Path(a.out)
    if out.exists():   # the source would otherwise return the cached file, whatever T0 it was built for
        print(f"replacing existing {out}", file=sys.stderr)
        out.unlink()
    src = BigQuerySource(a.sql, t0, cache_path=out, project=a.project)
    df = src.load()
    print(f"extracted {len(df)} users to {a.out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="expkit", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("methods", help="list registered statistical methods")
    s.add_argument("--kind", choices=["mean", "binary", "ratio"])
    s.set_defaults(fn=cmd_methods)

    s = sub.add_parser("analyze", help="readout for an assigned dataset + TOML config")
    s.add_argument("--data", required=True, help="parquet/csv with an arm column")
    s.add_argument("--config", required=True)
    s.add_argument("--format", choices=["md", "json"], default="md")
    s.add_argument("--out")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("power", help="sample size and/or MDE")
    s.add_argument("--baseline-mean", type=float, required=True)
    s.add_argument("--sd", type=float, help="omit for a binary metric (uses sqrt(p(1-p)))")
    s.add_argument("--mde", type=float)
    s.add_argument("--n", type=int, help="control sample size, to compute the MDE")
    s.add_argument("--relative", action="store_true", help="mde is relative to the baseline")
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--power", type=float, default=0.8)
    s.add_argument("--ratio", type=float, default=1.0)
    s.add_argument("--variance-reduction", type=float, default=0.0)
    s.set_defaults(fn=cmd_power)

    for name, fn, helptext in [("simulate", cmd_simulate, "A/A or injected-lift simulation"),
                               ("peek", cmd_peek, "peeking vs sequential monitoring")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--data", default="synthetic", help="synthetic[:n] | file.parquet | file.csv")
        if name == "simulate":
            s.add_argument("--metrics", default="answers_posted",
                           help=f"comma list from {sorted(CATALOG)}")
            s.add_argument("--methods", default="welch")
        else:
            s.add_argument("--metric", default="answers_posted")
            s.add_argument("--looks", type=int, default=28)
        s.add_argument("--n-per-arm", type=int, default=5000)
        s.add_argument("--runs", type=int, default=1000)
        s.add_argument("--lift", type=float, default=0.0)
        s.add_argument("--alpha", type=float, default=0.05)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--jobs", type=int, default=1)
        s.add_argument("--out")
        s.set_defaults(fn=fn)

    s = sub.add_parser("synth", help="write a synthetic users_metrics table")
    s.add_argument("--n", type=int, default=100_000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="data/synthetic_users_metrics.parquet")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("extract", help="run data/extract.sql on BigQuery (needs [bigquery])")
    s.add_argument("--t0", required=True, help="experiment start date, YYYY-MM-DD")
    s.add_argument("--sql", default="data/extract.sql")
    s.add_argument("--out", default="data/users_metrics.parquet")
    s.add_argument("--project")
    s.add_argument("--allow-truncated", action="store_true",
                   help="allow a post-period that runs past the end of the data")
    s.set_defaults(fn=cmd_extract)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    main()
