"""Lan-DeMets alpha spending with two-sided symmetric boundaries.

Boundaries are found by Monte Carlo on the canonical joint distribution of
the sequential z-statistics: Z_k = S_k / sqrt(t_k), where S is Brownian motion
in information time. At each look k, c_k is chosen so that the probability of
first crossing at look k equals alpha(t_k) - alpha(t_{k-1}). With the default
400k paths, boundary error is about +-0.01 at typical looks (see DECISION.md).
Results are cached, so a 5,000-run simulation computes boundaries once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy import stats


def obrien_fleming_spending(t: np.ndarray | float, alpha: float) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 1e-12, 1.0)
    return 2.0 - 2.0 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t))


def pocock_spending(t: np.ndarray | float, alpha: float) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return alpha * np.log(1.0 + (math.e - 1.0) * t)


SPENDING_FUNCTIONS = {"obf": obrien_fleming_spending, "pocock": pocock_spending}


def _validate_fractions(info_fractions: Sequence[float]) -> tuple[float, ...]:
    t = tuple(float(x) for x in info_fractions)
    if not t or any(b <= a for a, b in zip(t, t[1:], strict=False)) or t[0] <= 0 or t[-1] > 1:
        raise ValueError("info_fractions must be strictly increasing in (0, 1]")
    return t


@lru_cache(maxsize=64)
def _boundaries_cached(
    t: tuple[float, ...], alpha: float, spending: str, n_paths: int, seed: int
) -> tuple[float, ...]:
    spend = SPENDING_FUNCTIONS[spending](np.asarray(t), alpha)
    increments = np.diff(np.concatenate([[0.0], spend]))
    rng = np.random.default_rng(seed)
    s = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    prev_t = 0.0
    bounds = []
    for tk, a_k in zip(t, increments, strict=True):
        s += rng.standard_normal(n_paths) * math.sqrt(tk - prev_t)
        prev_t = tk
        z = np.abs(s / math.sqrt(tk))
        m = int(round(a_k * n_paths))
        alive_z = z[alive]
        if m <= 0 or alive_z.size == 0:
            c = math.inf
        elif m >= alive_z.size:
            c = 0.0
        else:
            # c such that exactly m alive paths have |Z_k| >= c
            c = float(np.partition(alive_z, alive_z.size - m)[alive_z.size - m])
        bounds.append(c)
        alive &= z < c
    return tuple(bounds)


def group_sequential_boundaries(
    info_fractions: Sequence[float],
    alpha: float = 0.05,
    spending: str = "obf",
    n_paths: int = 400_000,
    seed: int = 20240101,
) -> np.ndarray:
    """Two-sided |z| critical values for each look."""
    if spending not in SPENDING_FUNCTIONS:
        raise ValueError(f"spending must be one of {list(SPENDING_FUNCTIONS)}")
    t = _validate_fractions(info_fractions)
    return np.asarray(_boundaries_cached(t, float(alpha), spending, int(n_paths), int(seed)))


def first_crossing(z_stats: Sequence[float], boundaries: Sequence[float]) -> int | None:
    """Index of the first look where |z| >= boundary, or None."""
    z = np.abs(np.asarray(z_stats, dtype=float))
    b = np.asarray(boundaries, dtype=float)[: z.size]
    hits = np.nonzero(z >= b)[0]
    return int(hits[0]) if hits.size else None
