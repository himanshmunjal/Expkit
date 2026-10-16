"""Data sources. Each one implements ``load() -> pd.DataFrame`` (the DataSource
protocol) and returns a frame satisfying ``expkit.io.schema``.

Adding a warehouse (Snowflake, Redshift, Postgres, DuckDB...) means writing one
small class with a ``load`` method; nothing else in the package changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from expkit.io.schema import validate_user_metrics


@runtime_checkable
class DataSource(Protocol):
    def load(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ParquetSource:
    path: str | Path
    validate: bool = True

    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        return validate_user_metrics(df) if self.validate else df


@dataclass(frozen=True)
class CSVSource:
    path: str | Path
    validate: bool = True

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        return validate_user_metrics(df) if self.validate else df


@dataclass(frozen=True)
class BigQuerySource:
    """Runs ``data/extract.sql`` against the public Stack Overflow dataset.

    The result is cached to ``cache_path`` (parquet) so that the warehouse is
    hit once; every later run reads the local file. Requires the ``bigquery``
    extra and application-default credentials.
    """

    sql_path: str | Path
    t0: date
    cache_path: str | Path | None = None
    project: str | None = None
    max_bytes_billed: int = 50 * 1024**3  # guard against an accidental full scan bill

    def load(self) -> pd.DataFrame:
        if self.cache_path and Path(self.cache_path).exists():
            return validate_user_metrics(pd.read_parquet(self.cache_path))
        try:
            from google.cloud import bigquery
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("install the extra: pip install 'expkit[bigquery]'") from e
        client = bigquery.Client(project=self.project)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("t0", "DATE", self.t0)],
            maximum_bytes_billed=self.max_bytes_billed,
        )
        sql = Path(self.sql_path).read_text()
        df = client.query(sql, job_config=job_config).to_dataframe()
        df = validate_user_metrics(df)
        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.cache_path, index=False)
        return df


@dataclass(frozen=True)
class SyntheticSource:
    """Offline stand-in with the same schema and similar pathologies as the real
    extract: zero inflation, heavy right tail (gamma-mixed Poisson), and
    pre/post correlation through a persistent per-user propensity.

    Used for tests, CI and demos without cloud credentials. Results on it are
    NOT the project's findings; those come from the real extract.
    """

    n_users: int = 100_000
    seed: int = 7
    params: dict = field(default_factory=dict)

    def load(self) -> pd.DataFrame:
        p = {"shape": 0.3, "scale": 3.0, "engaged": 0.6, "retain": 0.7,
             "reactivate": 0.1, "drift_sd": 0.5, **self.params}
        rng = np.random.default_rng(self.seed)
        # ~40% of generated users are pre-active; oversample 5x, then keep the first n_users.
        n = int(self.n_users * 5) + 1000
        lam = rng.gamma(p["shape"], p["scale"], n)          # answers per 28 days
        engaged = rng.random(n) < p["engaged"]
        drift = rng.lognormal(-p["drift_sd"] ** 2 / 2, p["drift_sd"], n)
        post_engaged = np.where(engaged, rng.random(n) < p["retain"], rng.random(n) < p["reactivate"])
        accept_p = rng.beta(2.0, 5.0, n)
        comment_rate = 0.8 * lam + rng.gamma(0.5, 0.6, n)

        pre_answers = rng.poisson(2.0 * lam * engaged)        # 56-day pre-period
        post_answers = rng.poisson(lam * drift * post_engaged)
        pre_comments = rng.poisson(2.0 * comment_rate * engaged)
        post_comments = rng.poisson(comment_rate * drift * post_engaged)
        df = pd.DataFrame({
            "user_id": np.arange(1, n + 1),
            "tenure_days": rng.integers(56, 5000, n),
            "pre_answers": pre_answers,
            "pre_accepted": rng.binomial(pre_answers, accept_p),
            "pre_comments": pre_comments,
            "post_answers": post_answers,
            "post_accepted": rng.binomial(post_answers, accept_p),
            "post_comments": post_comments,
        })
        df["pre_active"] = ((df.pre_answers + df.pre_comments) > 0).astype(np.int8)
        df["post_active"] = ((df.post_answers + df.post_comments) > 0).astype(np.int8)
        df = df[df.pre_active == 1].head(self.n_users).reset_index(drop=True)
        if len(df) < self.n_users:
            raise ValueError(f"synthetic params yield too few pre-active users ({len(df)})")
        return validate_user_metrics(df)


def source_from_uri(uri: str) -> DataSource:
    """Resolve 'synthetic[:n]', '*.parquet' or '*.csv' to a DataSource."""
    if uri.startswith("synthetic"):
        _, _, n = uri.partition(":")
        return SyntheticSource(n_users=int(n) if n else 100_000)
    suffix = Path(uri).suffix.lower()
    if suffix == ".parquet":
        return ParquetSource(uri)
    if suffix == ".csv":
        return CSVSource(uri)
    raise ValueError(f"cannot infer a data source from {uri!r}")
