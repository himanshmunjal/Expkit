import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from expkit.stats import (
    benjamini_hochberg,
    bonferroni,
    holm,
    minimum_detectable_effect,
    power_at,
    required_sample_size,
    srm_test,
)


@given(st.integers(1, 10_000_000))
def test_srm_never_fires_on_exact_split(n):
    assert not srm_test([n, n]).mismatch


def test_srm_detects_large_imbalance():
    r = srm_test([50_000, 48_500])
    # chi2 = 2 * 750^2 / 49,250 = 22.84 -> p = 1.76e-6
    assert r.mismatch and r.chi2 == pytest.approx(22.84, abs=0.01)


def test_srm_respects_unequal_design():
    assert not srm_test([90_000, 10_000], [0.9, 0.1]).mismatch
    assert srm_test([90_000, 10_000]).mismatch


def test_srm_three_arms():
    assert not srm_test([1000, 1000, 1000]).mismatch


def test_holm_textbook_example():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_bh_textbook_example():
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_bonferroni_caps_at_one():
    np.testing.assert_allclose(bonferroni([0.3, 0.6]), [0.6, 1.0])


@given(st.lists(st.floats(0, 1), min_size=1, max_size=30))
def test_property_correction_ordering(p):
    p = np.array(p)
    assert np.all(benjamini_hochberg(p) <= holm(p) + 1e-12)
    assert np.all(holm(p) <= bonferroni(p) + 1e-12)
    assert np.all(holm(p) >= p - 1e-12)


def test_sample_size_textbook():
    # n = 2 (z_.975 + z_.8)^2 sd^2 / delta^2 = 2 * 7.849 * 1 / 0.01 = 1570
    n_c, n_t = required_sample_size(1.0, 0.1)
    assert n_c == n_t == 1570


def test_power_and_mde_roundtrip():
    n_c, _ = required_sample_size(2.0, 0.05, baseline_mean=1.0, relative=True)
    assert power_at(2.0, n_c, 0.05, baseline_mean=1.0, relative=True) == pytest.approx(0.8, abs=0.01)
    mde = minimum_detectable_effect(2.0, n_c, baseline_mean=1.0, relative=True)
    assert mde == pytest.approx(0.05, rel=0.01)


def test_variance_reduction_shrinks_n():
    base, _ = required_sample_size(1.0, 0.1)
    cuped, _ = required_sample_size(1.0, 0.1, variance_reduction=0.5)
    assert cuped == pytest.approx(base * 0.5, rel=0.01)
