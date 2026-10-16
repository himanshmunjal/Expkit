"""Capping heavy tails.

The cap MUST be computed from data that does not depend on assignment: the
pooled sample (both arms) or a pre-period distribution. Capping each arm at its
own percentile would cap away part of the treatment effect asymmetrically.
"""

from __future__ import annotations

import numpy as np

from expkit._types import as_1d_float


def cap_value(reference, quantile: float) -> float:
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    return float(np.quantile(as_1d_float(reference, "reference"), quantile))


def winsorize_pair(control, treatment, quantile: float = 0.99) -> tuple[np.ndarray, np.ndarray, float]:
    c = as_1d_float(control, "control")
    t = as_1d_float(treatment, "treatment")
    cap = cap_value(np.concatenate([c, t]), quantile)
    return np.minimum(c, cap), np.minimum(t, cap), cap
