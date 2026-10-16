import numpy as np
import pytest

from expkit._types import MetricSample
from expkit.assign import bucket_of, hash_assign, random_assign, stratified_assign
from expkit.effects import inject, inject_binary, inject_count, inject_ratio


def test_hash_assign_deterministic_and_balanced():
    ids = np.arange(100_000)
    a1, a2 = hash_assign(ids, "exp-1"), hash_assign(ids, "exp-1")
    np.testing.assert_array_equal(a1, a2)
    assert abs(a1.mean() - 0.5) < 0.01


def test_hash_salt_decorrelates_experiments():
    ids = np.arange(50_000)
    a, b = hash_assign(ids, "exp-1"), hash_assign(ids, "exp-2")
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.02


def test_hash_unequal_weights_and_bucket_range():
    arms = hash_assign(range(50_000), "x", weights=(0.9, 0.1))
    assert abs(arms.mean() - 0.1) < 0.01
    assert 0 <= bucket_of("u", "s") < 10_000


def test_random_and_stratified_assign_exact(rng):
    assert random_assign(1001, rng).sum() in (500, 501)
    strata = np.repeat([0, 1, 2], 100)
    arms = stratified_assign(strata, rng)
    for s in range(3):
        assert arms[strata == s].sum() == 50


@pytest.mark.parametrize("lift", [-0.2, -0.03, 0.03, 0.2])
def test_count_injection_hits_expected_lift(rng, lift):
    y = rng.poisson(rng.gamma(0.3, 3, 200_000)).astype(float)
    out = inject_count(y, lift, rng)
    assert out.mean() / y.mean() - 1 == pytest.approx(lift, abs=0.01)
    assert np.all(out >= 0) and np.all(out % 1 == 0)
    assert np.all(out[y == 0] == 0)


@pytest.mark.parametrize("lift", [-0.1, 0.1])
def test_binary_injection(rng, lift):
    y = (rng.random(200_000) < 0.3).astype(float)
    out = inject_binary(y, lift, rng)
    assert out.mean() / y.mean() - 1 == pytest.approx(lift, abs=0.01)
    assert set(np.unique(out)) <= {0.0, 1.0}


@pytest.mark.parametrize("lift", [-0.1, 0.1])
def test_ratio_injection_keeps_num_le_den(rng, lift):
    den = rng.poisson(rng.gamma(0.5, 4, 100_000)).astype(float)
    num = rng.binomial(den.astype(int), 0.3).astype(float)
    out = inject_ratio(num, den, lift, rng)
    assert np.all(out <= den)
    assert (out.sum() / den.sum()) / (num.sum() / den.sum()) - 1 == pytest.approx(lift, abs=0.01)


def test_heterogeneous_lift(rng):
    y = rng.poisson(2.0, 100_000).astype(float)
    lifts = np.where(np.arange(y.size) < 50_000, 0.0, 0.2)
    out = inject_count(y, lifts, rng)
    assert out[:50_000].sum() == y[:50_000].sum()
    assert out[50_000:].mean() / y[50_000:].mean() == pytest.approx(1.2, abs=0.02)


def test_inject_never_touches_covariate(rng):
    s = MetricSample(y=np.ones(10), x=np.arange(10.0))
    out = inject(s, "mean", 0.5, rng)
    np.testing.assert_array_equal(out.x, s.x)


def test_lift_below_minus_one_rejected(rng):
    with pytest.raises(ValueError):
        inject_count(np.ones(5), -1.0, rng)
