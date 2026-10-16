import numpy as np
import pytest

from expkit.stats import (
    bootstrap_mean_diff,
    bootstrap_ratio_diff,
    delta_ratio_test,
    naive_ratio_ttest,
    ratio_and_variance,
)


def clustered(rng, n, rate):
    den = rng.poisson(rng.gamma(0.5, 4, n))
    return rng.binomial(den, rate).astype(float), den.astype(float)


def test_ratio_variance_matches_bootstrap(rng):
    num, den = clustered(rng, 3000, 0.3)
    r, v = ratio_and_variance(num, den)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, num.size, num.size)
        boots.append(num[i].sum() / den[i].sum())
    assert r == pytest.approx(num.sum() / den.sum())
    assert np.sqrt(v) == pytest.approx(np.std(boots), rel=0.1)


def test_delta_coverage_near_nominal(rng):
    covered = 0
    runs = 400
    for _ in range(runs):
        nc, dc = clustered(rng, 1500, 0.3)
        nt, dt = clustered(rng, 1500, 0.3)
        r = delta_ratio_test(nc, dc, nt, dt)
        covered += r.ci_low <= 0 <= r.ci_high
    assert 0.92 <= covered / runs <= 0.98


def test_delta_relative_lift(rng):
    nc, dc = clustered(rng, 20_000, 0.30)
    nt, dt = clustered(rng, 20_000, 0.33)
    r = delta_ratio_test(nc, dc, nt, dt, relative=True)
    assert r.estimate == pytest.approx(0.1, abs=0.04)


def test_naive_ratio_drops_zero_denominators(rng):
    nc, dc = clustered(rng, 500, 0.3)
    r = naive_ratio_ttest(nc, dc, nc, dc)
    assert r.n_control == int((dc > 0).sum())


def test_bootstrap_percentile_and_bca(rng):
    c, t = rng.exponential(1, 400), rng.exponential(1.5, 400)
    for kind in ("percentile", "bca"):
        r = bootstrap_mean_diff(c, t, n_boot=1000, kind=kind, rng=rng)
        assert r.ci_low < r.estimate < r.ci_high
        assert r.p_value < 0.05


def test_bootstrap_ratio_runs(rng):
    nc, dc = clustered(rng, 300, 0.3)
    nt, dt = clustered(rng, 300, 0.3)
    r = bootstrap_ratio_diff(nc, dc, nt, dt, n_boot=300, rng=rng)
    assert r.ci_low <= r.estimate <= r.ci_high


def test_delta_rejects_zero_denominator():
    with pytest.raises(ValueError):
        delta_ratio_test([0, 0], [0, 0], [1, 1], [1, 1])


def test_jackknife_matches_delta_on_light_tails(rng):
    from expkit.stats import jackknife_ratio_and_variance
    num, den = clustered(rng, 5000, 0.3)
    r, v = ratio_and_variance(num, den)
    rj, vj, n_eff = jackknife_ratio_and_variance(num, den)
    assert rj == pytest.approx(r)
    assert vj == pytest.approx(v, rel=0.05)
    assert n_eff == int((den > 0).sum())


def test_jackknife_calibrated_where_delta_overrejects(rng):
    """Heavy-tailed denominators, small n: the regime where delta failed on real data."""
    from expkit.stats import jackknife_ratio_test

    def heavy(n):
        den = np.where(rng.random(n) < 0.12, np.ceil(rng.pareto(1.1, n) + 1), 0)
        return rng.binomial(den.astype(int), 0.3).astype(float), den

    runs, rej_delta, rej_jk = 1500, 0, 0
    for _ in range(runs):
        (nc, dc), (nt, dt) = heavy(150), heavy(150)
        if (dc > 0).sum() < 2 or (dt > 0).sum() < 2:
            continue
        rej_delta += delta_ratio_test(nc, dc, nt, dt).significant
        rej_jk += jackknife_ratio_test(nc, dc, nt, dt).significant
    assert rej_jk / runs < 0.08
    assert rej_jk < rej_delta


def test_jackknife_needs_two_effective_units():
    from expkit.stats import jackknife_ratio_test
    with pytest.raises(ValueError, match="positive denominator"):
        jackknife_ratio_test([1, 0, 0], [2, 0, 0], [1, 1], [2, 3])
