"""Deterministic, optionally parallel execution of independent simulation runs.

Each run gets its own child of one SeedSequence, so results are identical
whether runs execute serially, on 8 cores, or in a different order.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np


def run_many(fn: Callable[[int, np.random.Generator], Any], n_runs: int, seed: int,
             n_jobs: int = 1) -> list[Any]:
    children = np.random.SeedSequence(seed).spawn(n_runs)

    def chunk(ids: list[int]) -> list[Any]:
        return [fn(i, np.random.default_rng(children[i])) for i in ids]

    if n_jobs == 1:
        return chunk(list(range(n_runs)))
    try:
        from joblib import Parallel, delayed
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("n_jobs != 1 needs joblib: pip install 'expkit[parallel]'") from e
    workers = n_jobs if n_jobs > 0 else max(1, os.cpu_count() or 1)
    batches = [b.tolist() for b in np.array_split(np.arange(n_runs), workers * 4) if b.size]
    out = Parallel(n_jobs=n_jobs)(delayed(chunk)(b) for b in batches)
    return [r for batch in out for r in batch]
