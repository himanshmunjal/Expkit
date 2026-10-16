"""Mixture sequential probability ratio test (always-valid p-values).

Normal-mixture mSPRT (Johari, Pekelis, Walsh 2017) on the difference in means.
With estimated difference d_n, its variance v_n, and a N(0, tau^2) mixing
distribution over the true effect:

    Lambda_n = sqrt(v_n / (v_n + tau^2)) * exp(tau^2 d_n^2 / (2 v_n (v_n + tau^2)))
    p_n      = min(p_{n-1}, 1 / Lambda_n)

The p-value sequence is valid at any data-dependent stopping time. tau^2 should
be on the scale of the effects you care about, e.g. (expected absolute lift)^2.
"""

from __future__ import annotations

import numpy as np


def msprt_log_lambda(diffs, variances, tau2: float) -> np.ndarray:
    d = np.asarray(diffs, dtype=float)
    v = np.asarray(variances, dtype=float)
    if tau2 <= 0:
        raise ValueError("tau2 must be positive")
    if np.any(v <= 0):
        raise ValueError("variances must be positive")
    return 0.5 * np.log(v / (v + tau2)) + tau2 * d**2 / (2.0 * v * (v + tau2))


def msprt_pvalues(diffs, variances, tau2: float) -> np.ndarray:
    """Always-valid p-values for a sequence of looks."""
    log_lam = msprt_log_lambda(diffs, variances, tau2)
    p = np.minimum(1.0, np.exp(-log_lam))
    return np.minimum.accumulate(p)
