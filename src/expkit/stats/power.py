"""Sample size, MDE and power for a two-sided two-sample comparison of means.

Normal approximation throughout. ``ratio`` = n_treatment / n_control.
``variance_reduction`` (e.g. from CUPED, = rho^2) shrinks the variance before
the calculation, which is how CUPED buys sample size.
"""

from __future__ import annotations

import math

from scipy import stats


def _z(alpha: float, power: float) -> tuple[float, float]:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    return float(stats.norm.ppf(1 - alpha / 2)), float(stats.norm.ppf(power))


def _var(sd: float, variance_reduction: float) -> float:
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not 0 <= variance_reduction < 1:
        raise ValueError("variance_reduction must be in [0, 1)")
    return sd * sd * (1.0 - variance_reduction)


def _abs_effect(effect: float, baseline_mean: float | None, relative: bool) -> float:
    if relative:
        if baseline_mean is None or baseline_mean == 0:
            raise ValueError("relative effects need a non-zero baseline_mean")
        return abs(effect * baseline_mean)
    return abs(effect)


def binary_sd(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    return math.sqrt(p * (1 - p))


def required_sample_size(
    sd: float,
    mde: float,
    *,
    baseline_mean: float | None = None,
    relative: bool = False,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    variance_reduction: float = 0.0,
) -> tuple[int, int]:
    """Return (n_control, n_treatment) needed to detect ``mde``."""
    za, zb = _z(alpha, power)
    delta = _abs_effect(mde, baseline_mean, relative)
    if delta == 0:
        raise ValueError("mde must be non-zero")
    n_c = (za + zb) ** 2 * _var(sd, variance_reduction) * (1 + 1 / ratio) / delta**2
    n_c = math.ceil(n_c)
    return n_c, math.ceil(n_c * ratio)


def minimum_detectable_effect(
    sd: float,
    n_control: int,
    *,
    baseline_mean: float | None = None,
    relative: bool = False,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    variance_reduction: float = 0.0,
) -> float:
    """Smallest effect detectable with the given power (absolute, or relative)."""
    za, zb = _z(alpha, power)
    se = math.sqrt(_var(sd, variance_reduction) * (1 / n_control + 1 / (n_control * ratio)))
    mde = (za + zb) * se
    if relative:
        if not baseline_mean:
            raise ValueError("relative MDE needs a non-zero baseline_mean")
        return mde / abs(baseline_mean)
    return mde


def power_at(
    sd: float,
    n_control: int,
    effect: float,
    *,
    baseline_mean: float | None = None,
    relative: bool = False,
    alpha: float = 0.05,
    ratio: float = 1.0,
    variance_reduction: float = 0.0,
) -> float:
    """Power of a two-sided test to detect ``effect``."""
    za = float(stats.norm.ppf(1 - alpha / 2))
    delta = _abs_effect(effect, baseline_mean, relative)
    se = math.sqrt(_var(sd, variance_reduction) * (1 / n_control + 1 / (n_control * ratio)))
    shift = delta / se
    return float(stats.norm.cdf(shift - za) + stats.norm.cdf(-shift - za))
