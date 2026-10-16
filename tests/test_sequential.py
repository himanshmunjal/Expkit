import numpy as np
import pytest

from expkit.sequential import (
    first_crossing,
    group_sequential_boundaries,
    msprt_pvalues,
    obrien_fleming_spending,
    pocock_spending,
)


def test_spending_functions_end_at_alpha():
    assert obrien_fleming_spending(1.0, 0.05) == pytest.approx(0.05)
    assert pocock_spending(1.0, 0.05) == pytest.approx(0.05)
    assert obrien_fleming_spending(0.2, 0.05) < pocock_spending(0.2, 0.05)


def test_single_look_is_fixed_horizon():
    b = group_sequential_boundaries([1.0], 0.05, "obf")
    assert b[0] == pytest.approx(1.96, abs=0.02)


def test_obf_five_looks_matches_published_values():
    # Lan-DeMets OBF, 5 equally spaced looks, two-sided alpha .05:
    # final boundary ~2.04 (Jennison & Turnbull, Table 2.x / gsDesign)
    b = group_sequential_boundaries([0.2, 0.4, 0.6, 0.8, 1.0], 0.05, "obf")
    assert b[-1] == pytest.approx(2.04, abs=0.04)
    assert b[1] > b[2] > b[3] > b[4]


def test_boundaries_control_type_one_error():
    t = np.linspace(0.1, 1.0, 10)
    rng = np.random.default_rng(99)
    s = np.cumsum(rng.standard_normal((50_000, 10)) * np.sqrt(np.diff(np.r_[0, t])), axis=1)
    z = s / np.sqrt(t)
    for spending in ("obf", "pocock"):
        b = group_sequential_boundaries(t, 0.05, spending)
        fpr = np.mean((np.abs(z) >= b).any(axis=1))
        assert fpr == pytest.approx(0.05, abs=0.006)
    naive = np.mean((np.abs(z) >= 1.96).any(axis=1))
    assert naive > 0.15


def test_first_crossing():
    assert first_crossing([1, 2, 3], [2.5, 2.5, 2.5]) == 2
    assert first_crossing([1, 1], [2, 2]) is None


def test_msprt_pvalues_monotone_and_valid():
    rng = np.random.default_rng(0)
    rejections = 0
    for _ in range(2000):
        x = rng.normal(size=(2, 400))
        n = np.arange(1, 401)
        diff = np.cumsum(x[1]) / n - np.cumsum(x[0]) / n
        p = msprt_pvalues(diff, 2.0 / n, tau2=0.05)
        assert np.all(np.diff(p) <= 0)
        rejections += p[-1] < 0.05
    assert rejections / 2000 <= 0.05


def test_rejects_bad_fractions():
    with pytest.raises(ValueError):
        group_sequential_boundaries([0.5, 0.4, 1.0])
