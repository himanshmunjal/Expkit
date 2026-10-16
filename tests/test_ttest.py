import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats

from expkit.stats import mann_whitney, welch_ttest, z_test


def test_welch_matches_scipy(rng):
    c, t = rng.normal(0, 1, 300), rng.normal(0.2, 2, 500)
    ours = welch_ttest(c, t)
    ref = stats.ttest_ind(t, c, equal_var=False)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    assert ours.estimate == pytest.approx(t.mean() - c.mean())
    assert ours.extra["t"] == pytest.approx(ref.statistic, rel=1e-10)


def test_welch_ci_excludes_zero_iff_significant(rng):
    for _ in range(50):
        r = welch_ttest(rng.normal(0, 1, 50), rng.normal(0.3, 1, 50))
        assert r.significant == (r.ci_low > 0 or r.ci_high < 0)


def test_ztest_close_to_welch_for_large_n(rng):
    c, t = rng.exponential(1, 20_000), rng.exponential(1.02, 20_000)
    assert z_test(c, t).p_value == pytest.approx(welch_ttest(c, t).p_value, abs=1e-3)


def test_constant_arms():
    r = welch_ttest([1, 1, 1], [1, 1, 1])
    assert r.p_value == 1.0 and r.estimate == 0.0
    assert welch_ttest([1, 1], [2, 2]).p_value == 0.0


@pytest.mark.parametrize("bad", [[1.0], [], [1.0, np.nan]])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        welch_ttest(bad, [1.0, 2.0, 3.0])


def test_mann_whitney_has_no_ci(rng):
    r = mann_whitney(rng.normal(size=100), rng.normal(size=100))
    assert np.isnan(r.ci_low) and 0 <= r.p_value <= 1


finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None)
@given(arrays(float, st.integers(2, 40), elements=finite),
       arrays(float, st.integers(2, 40), elements=finite))
def test_property_pvalue_in_unit_interval_and_swap_symmetry(c, t):
    a, b = welch_ttest(c, t), welch_ttest(t, c)
    assert 0.0 <= a.p_value <= 1.0
    assert a.p_value == pytest.approx(b.p_value, abs=1e-12)
    assert a.estimate == pytest.approx(-b.estimate, abs=1e-6)


@settings(max_examples=30, deadline=None)
@given(arrays(float, st.integers(3, 30), elements=st.floats(-100, 100)),
       st.floats(-50, 50))
def test_property_shift_invariance(x, shift):
    y = x[::-1] + 0.5
    a, b = welch_ttest(x, y), welch_ttest(x + shift, y + shift)
    assert a.p_value == pytest.approx(b.p_value, abs=1e-6)
