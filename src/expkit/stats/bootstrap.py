"""Bootstrap intervals for mean differences and ratio-of-sums differences.

Resampling is chunked so memory stays bounded at roughly ``chunk * n`` indices.
The p-value is the two-sided bootstrap tail probability of zero, an
approximation (documented in DECISION.md) that is checked by the A/A harness.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from expkit._types import TestResult, as_1d_float

_MAX_INDICES_PER_CHUNK = 5_000_000  # ~40 MB of int64 indices


def _resampled_stat(rng, arrays: tuple[np.ndarray, ...], stat, n_boot: int) -> np.ndarray:
    n = arrays[0].size
    chunk = max(1, min(n_boot, _MAX_INDICES_PER_CHUNK // n))
    out = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        out[done : done + b] = stat(*(a[idx] for a in arrays))
        done += b
    return out


def _p_from_boot(boot: np.ndarray) -> float:
    lo = np.mean(boot <= 0.0)
    hi = np.mean(boot >= 0.0)
    return float(min(1.0, max(2.0 * min(lo, hi), 1.0 / boot.size)))


def _bca_interval(boot, observed, jackknife, alpha) -> tuple[float, float]:
    prop = np.mean(boot < observed)
    prop = min(max(prop, 1.0 / boot.size), 1.0 - 1.0 / boot.size)
    z0 = stats.norm.ppf(prop)
    u = jackknife.mean() - jackknife
    denom = 6.0 * (np.sum(u**2) ** 1.5)
    a = float(np.sum(u**3) / denom) if denom > 0 else 0.0
    zs = stats.norm.ppf([alpha / 2.0, 1.0 - alpha / 2.0])
    adj = stats.norm.cdf(z0 + (z0 + zs) / (1.0 - a * (z0 + zs)))
    lo, hi = np.quantile(boot, adj)
    return float(lo), float(hi)


def bootstrap_mean_diff(
    control,
    treatment,
    alpha: float = 0.05,
    n_boot: int = 2000,
    kind: str = "percentile",
    rng: np.random.Generator | None = None,
) -> TestResult:
    """Two-sample bootstrap of the mean difference (percentile or BCa interval)."""
    if kind not in ("percentile", "bca"):
        raise ValueError("kind must be 'percentile' or 'bca'")
    rng = rng or np.random.default_rng()
    c = as_1d_float(control, "control")
    t = as_1d_float(treatment, "treatment")
    mc, mt = float(c.mean()), float(t.mean())
    observed = mt - mc
    bc = _resampled_stat(rng, (c,), lambda a: a.mean(axis=1), n_boot)
    bt = _resampled_stat(rng, (t,), lambda a: a.mean(axis=1), n_boot)
    boot = bt - bc
    if kind == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    else:
        # Leave-one-out means for each observation of each arm.
        jk_c = mt - (c.sum() - c) / (c.size - 1)
        jk_t = (t.sum() - t) / (t.size - 1) - mc
        lo, hi = _bca_interval(boot, observed, np.concatenate([jk_c, jk_t]), alpha)
    return TestResult(
        f"bootstrap_{kind}", observed, float(lo), float(hi), _p_from_boot(boot), alpha,
        mc, mt, c.size, t.size, {"n_boot": n_boot, "se": float(boot.std(ddof=1))},
    )


def bootstrap_ratio_diff(
    num_c, den_c, num_t, den_t,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """User-level bootstrap of sum(num)/sum(den) difference (percentile interval)."""
    rng = rng or np.random.default_rng()
    nc, dc = as_1d_float(num_c, "num_c"), as_1d_float(den_c, "den_c")
    nt, dt = as_1d_float(num_t, "num_t"), as_1d_float(den_t, "den_t")
    if dc.sum() <= 0 or dt.sum() <= 0:
        raise ValueError("denominator sums must be positive in both arms")
    rc, rt = nc.sum() / dc.sum(), nt.sum() / dt.sum()

    def ratio(num, den):
        d = den.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d > 0, num.sum(axis=1) / d, math.nan)

    bc = _resampled_stat(rng, (nc, dc), ratio, n_boot)
    bt = _resampled_stat(rng, (nt, dt), ratio, n_boot)
    boot = bt - bc
    boot = boot[np.isfinite(boot)]
    if boot.size < 0.5 * n_boot:
        raise ValueError("most resamples had a zero denominator; sample too sparse")
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return TestResult(
        "bootstrap_ratio", float(rt - rc), float(lo), float(hi), _p_from_boot(boot), alpha,
        float(rc), float(rt), nc.size, nt.size, {"n_boot": int(boot.size)},
    )
