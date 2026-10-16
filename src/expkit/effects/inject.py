"""Inject a known effect into real outcome data.

All injections are multiplicative on the expected value, keep the data type
(counts stay non-negative integers, binaries stay 0/1, accepted <= answers),
and keep zeros where a user had no activity to scale. ``lift`` may be a scalar
or a per-unit array (heterogeneous / segment effects).

  counts, lift < 0 : binomial thinning      y' ~ Binomial(y, 1 + lift)
  counts, lift > 0 : Poisson thickening     y' = y + Poisson(lift * y)
  continuous       : scaling                y' = y * (1 + lift)
  binary           : flip 0 -> 1 with prob p*lift/(1-p)  (lift > 0)
                     flip 1 -> 0 with prob -lift          (lift < 0)
  ratio            : convert non-accepted -> accepted (or the reverse) on the
                     numerator so that sum(num)/sum(den) moves by (1 + lift)
"""

from __future__ import annotations

import numpy as np

from expkit._types import MetricSample


def _lift_array(lift, n: int) -> np.ndarray:
    arr = np.broadcast_to(np.asarray(lift, dtype=float), (n,)).astype(float)
    if np.any(arr <= -1.0):
        raise ValueError("lift must be > -1 (cannot remove more than 100%)")
    return arr


def _is_integer(y: np.ndarray) -> bool:
    return bool(np.all(y >= 0) and np.all(np.equal(np.mod(y, 1), 0)))


def inject_count(y: np.ndarray, lift, rng: np.random.Generator) -> np.ndarray:
    lifts = _lift_array(lift, y.size)
    if not _is_integer(y):
        return y * (1.0 + lifts)
    out = y.copy()
    neg, pos = lifts < 0, lifts > 0
    if neg.any():
        out[neg] = rng.binomial(y[neg].astype(np.int64), 1.0 + lifts[neg])
    if pos.any():
        out[pos] = y[pos] + rng.poisson(lifts[pos] * y[pos])
    return out


def inject_binary(y: np.ndarray, lift, rng: np.random.Generator) -> np.ndarray:
    lifts = _lift_array(lift, y.size)
    p = float(y.mean())
    out = y.copy()
    u = rng.random(y.size)
    if p < 1.0:
        up = (lifts > 0) & (y == 0)
        q = np.clip(p * lifts / (1.0 - p), 0.0, 1.0)
        out[up & (u < q)] = 1.0
    down = (lifts < 0) & (y == 1)
    out[down & (u < -lifts)] = 0.0
    return out


def inject_ratio(num: np.ndarray, den: np.ndarray, lift, rng: np.random.Generator) -> np.ndarray:
    lifts = _lift_array(lift, num.size)
    out = num.copy()
    total_num = num.sum()
    headroom = den - num
    pos, neg = lifts > 0, lifts < 0
    if pos.any() and headroom[pos].sum() > 0:
        # per-unit conversion prob so the expected numerator gain is lift * num
        q = np.clip(lifts * total_num / headroom.sum(), 0.0, 1.0)
        out[pos] = num[pos] + rng.binomial(headroom[pos].astype(np.int64), q[pos])
    if neg.any():
        out[neg] = rng.binomial(num[neg].astype(np.int64), 1.0 + lifts[neg])
    return out


def inject(sample: MetricSample, kind: str, lift, rng: np.random.Generator) -> MetricSample:
    """Return a new MetricSample with the effect applied to the outcome only.

    Covariates (pre-period) are never touched: a real treatment cannot change
    the past, and touching them would bias CUPED.
    """
    if np.all(np.asarray(lift) == 0):
        return sample
    if kind == "mean":
        y = inject_count(sample.y, lift, rng)
    elif kind == "binary":
        y = inject_binary(sample.y, lift, rng)
    elif kind == "ratio":
        y = inject_ratio(sample.y, sample.den, lift, rng)
    else:
        raise ValueError(f"unknown metric kind {kind!r}")
    return MetricSample(y=y, den=sample.den, x=sample.x)
