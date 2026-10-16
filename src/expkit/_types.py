"""Core data contracts shared by every module.

Everything that crosses a module boundary is one of these frozen dataclasses.
The stats layer only sees numpy arrays wrapped in ``MetricSample``; it never sees
pandas, BigQuery or file paths. That keeps the statistics reusable from any
caller (CLI, notebook, REST service, Streamlit, another package).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

MetricKind = Literal["mean", "binary", "ratio"]
METRIC_KINDS: tuple[str, ...] = ("mean", "binary", "ratio")


def as_1d_float(values: Any, name: str = "values") -> np.ndarray:
    """Convert to a 1-d float array and fail loudly on NaN/inf or empty input."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


@dataclass(frozen=True)
class MetricSample:
    """Per-unit values for one arm of one metric.

    y   : outcome (for ratio metrics: the numerator)
    den : denominator, only for ratio metrics
    x   : pre-experiment covariate(s), shape (n,) or (n, k), for CUPED
    """

    y: np.ndarray
    den: np.ndarray | None = None
    x: np.ndarray | None = None

    def __post_init__(self) -> None:
        y = as_1d_float(self.y, "y")
        object.__setattr__(self, "y", y)
        if self.den is not None:
            den = as_1d_float(self.den, "den")
            if den.size != y.size:
                raise ValueError("den must have the same length as y")
            object.__setattr__(self, "den", den)
        if self.x is not None:
            x = np.asarray(self.x, dtype=float)
            if x.shape[0] != y.size or x.ndim not in (1, 2):
                raise ValueError("x must have shape (n,) or (n, k) with n == len(y)")
            if not np.all(np.isfinite(x)):
                raise ValueError("x contains NaN or infinite values")
            object.__setattr__(self, "x", x)

    def __len__(self) -> int:
        return int(self.y.size)

    def take(self, idx: np.ndarray) -> MetricSample:
        return MetricSample(
            y=self.y[idx],
            den=None if self.den is None else self.den[idx],
            x=None if self.x is None else self.x[idx],
        )


@dataclass(frozen=True)
class MetricSpec:
    """Declarative description of a metric and how to read it from a frame."""

    name: str
    kind: MetricKind
    column: str
    denominator: str | None = None
    covariates: tuple[str, ...] = ()
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if self.kind not in METRIC_KINDS:
            raise ValueError(f"kind must be one of {METRIC_KINDS}, got {self.kind!r}")
        if self.kind == "ratio" and not self.denominator:
            raise ValueError(f"ratio metric {self.name!r} needs a denominator column")
        if self.kind != "ratio" and self.denominator:
            raise ValueError(f"only ratio metrics take a denominator ({self.name!r})")
        object.__setattr__(self, "covariates", tuple(self.covariates))

    @property
    def columns(self) -> tuple[str, ...]:
        cols = [self.column, *self.covariates]
        if self.denominator:
            cols.append(self.denominator)
        return tuple(cols)

    def sample(self, df: pd.DataFrame, idx: np.ndarray | None = None) -> MetricSample:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise KeyError(f"metric {self.name!r}: missing columns {missing}")
        part = df if idx is None else df.iloc[idx]
        x = None
        if self.covariates:
            x = part[list(self.covariates)].to_numpy(dtype=float)
            if x.shape[1] == 1:
                x = x[:, 0]
        return MetricSample(
            y=part[self.column].to_numpy(dtype=float),
            den=None if self.denominator is None else part[self.denominator].to_numpy(dtype=float),
            x=x,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetricSpec:
        return cls(
            name=d["name"],
            kind=d["kind"],
            column=d["column"],
            denominator=d.get("denominator"),
            covariates=tuple(d.get("covariates", ())),
            higher_is_better=d.get("higher_is_better", True),
        )


@dataclass(frozen=True)
class TestResult:
    """Uniform output of every frequentist method.

    ``estimate`` is always the absolute difference (treatment - control) of the
    metric's natural statistic (mean, rate, or ratio of sums).
    """

    __test__ = False  # stop pytest from collecting this class

    method: str
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float
    control_mean: float
    treatment_mean: float
    n_control: int
    n_treatment: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_lift(self) -> float:
        return self.estimate / self.control_mean if self.control_mean != 0 else math.nan

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["relative_lift"] = self.relative_lift
        d["significant"] = self.significant
        return _jsonable(d)


@dataclass(frozen=True)
class SRMResult:
    observed: tuple[int, ...]
    expected_share: tuple[float, ...]
    chi2: float
    p_value: float
    threshold: float

    @property
    def mismatch(self) -> bool:
        return self.p_value < self.threshold

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mismatch"] = self.mismatch
        return _jsonable(d)


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
