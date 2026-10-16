"""Sample-ratio mismatch: chi-square goodness of fit on assignment counts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from expkit._types import SRMResult

# A strict threshold is the industry norm: SRM checks run on every experiment,
# so a 5% threshold would raise false alarms constantly.
DEFAULT_SRM_THRESHOLD = 1e-3


def srm_test(
    counts: Sequence[int],
    expected_share: Sequence[float] | None = None,
    threshold: float = DEFAULT_SRM_THRESHOLD,
) -> SRMResult:
    obs = np.asarray(counts, dtype=float)
    if obs.ndim != 1 or obs.size < 2:
        raise ValueError("need counts for at least 2 arms")
    if np.any(obs < 0):
        raise ValueError("counts must be non-negative")
    share = (
        np.full(obs.size, 1.0 / obs.size)
        if expected_share is None
        else np.asarray(expected_share, dtype=float)
    )
    if share.size != obs.size or np.any(share <= 0):
        raise ValueError("expected_share must be positive and match counts")
    share = share / share.sum()
    total = obs.sum()
    if total == 0:
        raise ValueError("no units assigned")
    expected = share * total
    chi2 = float(np.sum((obs - expected) ** 2 / expected))
    p = float(stats.chi2.sf(chi2, df=obs.size - 1))
    return SRMResult(
        tuple(int(c) for c in obs), tuple(float(s) for s in share), chi2, p, threshold
    )
