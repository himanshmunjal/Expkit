"""CUPED / regression adjustment with pre-experiment covariates.

theta is estimated on the POOLED sample (both arms) so the adjustment does not
depend on treatment, and the covariate is centred on the pooled mean so the
adjusted means keep the original scale. With one covariate this is classic
CUPED (theta = cov(Y, X) / var(X)); with several it is OLS regression
adjustment. Expected variance reduction is R^2 (= rho^2 for one covariate).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from expkit._types import TestResult, as_1d_float
from expkit.stats.ttest import welch_ttest


def _as_matrix(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def add_zero_indicator(x) -> np.ndarray:
    """Append a 'no pre-period activity' dummy per covariate column.

    Users with zero pre-activity often behave differently from what a linear fit
    through the active users predicts; the dummy lets the adjustment give them
    their own intercept.
    """
    m = _as_matrix(x)
    return np.hstack([m, (m == 0).astype(float)])


def cuped_theta(y, x) -> np.ndarray:
    """OLS slope(s) of y on centred x. Zero-variance columns get theta 0."""
    y = as_1d_float(y, "y")
    m = _as_matrix(x)
    xc = m - m.mean(axis=0)
    keep = xc.std(axis=0) > 0
    theta = np.zeros(m.shape[1])
    if keep.any():
        sol, *_ = np.linalg.lstsq(xc[:, keep], y - y.mean(), rcond=None)
        theta[keep] = sol
    return theta


def cuped_adjust(y, x, theta: np.ndarray, x_mean: np.ndarray) -> np.ndarray:
    return as_1d_float(y, "y") - (_as_matrix(x) - x_mean) @ theta


def cuped_ttest(
    y_control, y_treatment, x_control, x_treatment,
    alpha: float = 0.05,
    zero_indicator: bool = False,
) -> TestResult:
    """Welch t-test on CUPED-adjusted outcomes."""
    yc, yt = as_1d_float(y_control, "y_control"), as_1d_float(y_treatment, "y_treatment")
    xc, xt = _as_matrix(x_control), _as_matrix(x_treatment)
    if zero_indicator:
        xc, xt = add_zero_indicator(xc), add_zero_indicator(xt)
    y = np.concatenate([yc, yt])
    x = np.vstack([xc, xt])
    theta = cuped_theta(y, x)
    x_mean = x.mean(axis=0)
    adj_c = cuped_adjust(yc, xc, theta, x_mean)
    adj_t = cuped_adjust(yt, xt, theta, x_mean)
    res = welch_ttest(adj_c, adj_t, alpha, method="cuped")

    var_raw = np.var(np.concatenate([yc - yc.mean(), yt - yt.mean()]), ddof=1)
    var_adj = np.var(np.concatenate([adj_c - adj_c.mean(), adj_t - adj_t.mean()]), ddof=1)
    vr = 1.0 - var_adj / var_raw if var_raw > 0 else 0.0
    return replace(
        res,
        control_mean=float(yc.mean()),
        treatment_mean=float(yt.mean()),
        extra={**res.extra, "theta": theta.tolist(), "variance_reduction": float(vr)},
    )
