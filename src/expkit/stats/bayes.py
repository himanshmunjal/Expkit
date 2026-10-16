"""Bayesian beta-binomial comparison for binary metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from expkit._types import _jsonable


@dataclass(frozen=True)
class BayesResult:
    prob_treatment_better: float
    expected_loss_ship: float      # expected shortfall if we ship and treatment is worse
    expected_loss_hold: float      # expected shortfall if we hold and treatment is better
    lift_ci_low: float             # credible interval of the absolute difference
    lift_ci_high: float
    posterior_control: tuple[float, float]
    posterior_treatment: tuple[float, float]

    def to_dict(self):
        return _jsonable(asdict(self))


def beta_binomial(
    successes_c: int, n_c: int, successes_t: int, n_t: int,
    prior: tuple[float, float] = (1.0, 1.0),
    alpha: float = 0.05,
    n_draws: int = 100_000,
    rng: np.random.Generator | None = None,
) -> BayesResult:
    if not (0 <= successes_c <= n_c and 0 <= successes_t <= n_t):
        raise ValueError("successes must lie in [0, n]")
    rng = rng or np.random.default_rng()
    a0, b0 = prior
    pc = (a0 + successes_c, b0 + n_c - successes_c)
    pt = (a0 + successes_t, b0 + n_t - successes_t)
    dc = rng.beta(*pc, size=n_draws)
    dt = rng.beta(*pt, size=n_draws)
    diff = dt - dc
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return BayesResult(
        prob_treatment_better=float(np.mean(diff > 0)),
        expected_loss_ship=float(np.mean(np.maximum(-diff, 0))),
        expected_loss_hold=float(np.mean(np.maximum(diff, 0))),
        lift_ci_low=float(lo),
        lift_ci_high=float(hi),
        posterior_control=pc,
        posterior_treatment=pt,
    )
