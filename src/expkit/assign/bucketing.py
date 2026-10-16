"""Assignment strategies.

``hash_assign`` mirrors a production platform: deterministic, stateless,
reproducible from (salt, user_id) alone, so any service can recompute a user's
arm without a lookup table. ``random_assign`` and ``stratified_assign`` are
fast RNG-based alternatives used by the simulation harness.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np

N_BUCKETS = 10_000


def _normalise(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size < 2 or np.any(w <= 0):
        raise ValueError("weights must be >= 2 positive numbers")
    return w / w.sum()


def bucket_of(user_id: object, salt: str, n_buckets: int = N_BUCKETS) -> int:
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_buckets


def hash_assign(
    user_ids: Iterable[object],
    salt: str,
    weights: Sequence[float] = (0.5, 0.5),
    n_buckets: int = N_BUCKETS,
) -> np.ndarray:
    """Arm index per user from sha256(salt:user_id) % n_buckets.

    Buckets are split into contiguous ranges proportional to ``weights``. The
    salt must be unique per experiment so that assignments are independent
    across experiments.
    """
    w = _normalise(weights)
    edges = np.round(np.cumsum(w) * n_buckets).astype(int)
    buckets = np.fromiter((bucket_of(u, salt, n_buckets) for u in user_ids), dtype=np.int64)
    return np.searchsorted(edges, buckets, side="right").astype(np.int8)


def random_assign(n: int, rng: np.random.Generator, weights: Sequence[float] = (0.5, 0.5),
                  exact: bool = True) -> np.ndarray:
    """Random arm per unit. ``exact`` gives fixed arm sizes (complete randomisation)."""
    w = _normalise(weights)
    if not exact:
        return rng.choice(w.size, size=n, p=w).astype(np.int8)
    counts = np.floor(w * n).astype(int)
    counts[: n - counts.sum()] += 1
    arms = np.repeat(np.arange(w.size, dtype=np.int8), counts)
    rng.shuffle(arms)
    return arms


def stratified_assign(strata: Sequence[object], rng: np.random.Generator,
                      weights: Sequence[float] = (0.5, 0.5)) -> np.ndarray:
    """Complete randomisation within each stratum (e.g. activity tier)."""
    s = np.asarray(strata)
    arms = np.empty(s.size, dtype=np.int8)
    for value in np.unique(s):
        idx = np.nonzero(s == value)[0]
        arms[idx] = random_assign(idx.size, rng, weights, exact=True)
    return arms
