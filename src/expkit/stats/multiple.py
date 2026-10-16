"""Multiple-comparison corrections. All return adjusted p-values in input order."""

from __future__ import annotations

import numpy as np


def _validate(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p-values must be 1-dimensional")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("p-values must lie in [0, 1]")
    return p


def bonferroni(p) -> np.ndarray:
    p = _validate(p)
    return np.minimum(p * p.size, 1.0)


def holm(p) -> np.ndarray:
    """Holm step-down: controls family-wise error rate, uniformly beats Bonferroni."""
    p = _validate(p)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def benjamini_hochberg(p) -> np.ndarray:
    """Benjamini-Hochberg step-up: controls the false discovery rate."""
    p = _validate(p)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


CORRECTIONS = {"none": lambda p: _validate(p).copy(), "bonferroni": bonferroni,
               "holm": holm, "bh": benjamini_hochberg}


def adjust(p, method: str = "holm") -> np.ndarray:
    try:
        return CORRECTIONS[method](p)
    except KeyError:
        raise ValueError(f"unknown correction {method!r}; use one of {list(CORRECTIONS)}") from None
