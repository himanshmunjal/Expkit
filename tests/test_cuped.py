import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from expkit.stats import cuped_theta, cuped_ttest, welch_ttest


def correlated(rng, n, rho):
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    return y, x


def test_theta_matches_cov_over_var(rng):
    y, x = correlated(rng, 5000, 0.6)
    expected = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    assert cuped_theta(y, x)[0] == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_variance_reduction_equals_rho_squared(rng, rho):
    y, x = correlated(rng, 40_000, rho)
    r = cuped_ttest(y[:20_000], y[20_000:], x[:20_000], x[20_000:])
    assert r.extra["variance_reduction"] == pytest.approx(rho**2, abs=0.02)


def test_unbiased_lift_recovery(rng):
    ests = []
    for _ in range(300):
        y, x = correlated(rng, 2000, 0.7)
        y[1000:] += 0.1
        ests.append(cuped_ttest(y[:1000], y[1000:], x[:1000], x[1000:]).estimate)
    assert np.mean(ests) == pytest.approx(0.1, abs=0.005)


def test_zero_variance_covariate_equals_welch(rng):
    y = rng.normal(size=200)
    r = cuped_ttest(y[:100], y[100:], np.zeros(100), np.zeros(100))
    w = welch_ttest(y[:100], y[100:])
    assert r.estimate == pytest.approx(w.estimate)
    assert r.p_value == pytest.approx(w.p_value)


def test_multi_covariate_and_zero_indicator(rng):
    n = 4000
    x = rng.poisson(rng.gamma(0.3, 3, (n, 2)))
    y = x @ np.array([0.5, 0.2]) + rng.normal(size=n)
    r = cuped_ttest(y[:2000], y[2000:], x[:2000], x[2000:], zero_indicator=True)
    assert len(r.extra["theta"]) == 4
    assert r.extra["variance_reduction"] > 0.3


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000))
def test_property_uncorrelated_covariate_gives_same_estimate(seed):
    rng = np.random.default_rng(seed)
    y, x = rng.normal(size=4000), rng.normal(size=4000)
    r = cuped_ttest(y[:2000], y[2000:], x[:2000], x[2000:])
    w = welch_ttest(y[:2000], y[2000:])
    assert abs(r.estimate - w.estimate) < 4 * w.extra["se"] * 0.1 + 0.01
