import numpy as np
import pytest

from expkit.catalog import ACCEPT_RATE, ACTIVE, ANSWERS
from expkit.sim import (
    PeekingSpec,
    SimulationSpec,
    run_peeking_simulation,
    run_simulation,
    run_srm_simulation,
    summarize,
    summarize_peeking,
    wilson_interval,
)


def test_aa_false_positive_rate_is_nominal(population):
    spec = SimulationSpec((ANSWERS, ACTIVE, ACCEPT_RATE), 2000, 400, 0.0,
                          ("welch", "cuped", "delta"), seed=1)
    s = summarize(run_simulation(population, spec))
    assert set(s.method) == {"welch", "cuped", "delta"}
    for _, row in s.iterrows():
        assert 0.02 <= row.rejection_rate <= 0.085, row.to_dict()


def test_injected_lift_recovered_with_power(population):
    spec = SimulationSpec((ANSWERS,), 3000, 200, 0.15, ("welch", "cuped"), seed=2)
    s = summarize(run_simulation(population, spec)).set_index("method")
    assert abs(s.loc["cuped", "bias"]) < 0.25 * s.loc["cuped", "true_effect"]
    assert s.loc["cuped", "rejection_rate"] >= s.loc["welch", "rejection_rate"]
    assert 0.88 <= s.loc["cuped", "coverage"] <= 0.99


def test_simulation_is_reproducible(population):
    spec = SimulationSpec((ANSWERS,), 500, 20, 0.0, ("welch",), seed=3)
    a, b = run_simulation(population, spec), run_simulation(population, spec)
    np.testing.assert_array_equal(a.p_value, b.p_value)


def test_parallel_equals_serial(population):
    pytest.importorskip("joblib")
    base = dict(metrics=(ANSWERS,), n_per_arm=300, n_runs=16, methods=("welch",), seed=4)
    a = run_simulation(population, SimulationSpec(**base, n_jobs=1))
    b = run_simulation(population, SimulationSpec(**base, n_jobs=2))
    np.testing.assert_allclose(a.p_value, b.p_value)


def test_too_large_sample_rejected(population):
    with pytest.raises(ValueError):
        run_simulation(population, SimulationSpec((ANSWERS,), 50_000, 1))


def test_peeking_inflates_and_sequential_controls(population):
    spec = PeekingSpec(ANSWERS, 2000, looks=20, n_runs=600, seed=5)
    s = summarize_peeking(run_peeking_simulation(population, spec)).set_index("rule")
    assert s.loc["naive_peeking", "rejection_rate"] > 2 * s.loc["fixed_horizon", "rejection_rate"]
    for rule in ("obf", "pocock", "msprt"):
        assert s.loc[rule, "rejection_rate"] <= 0.08


def test_srm_simulation_detects_bug(population):
    ids = population.user_id.to_numpy()
    clean = run_srm_simulation(ids, 10_000, 0.0, n_runs=40, seed=6)
    # 8% loss on 10k users sits at the edge of p < 0.001 (expected chi2 ~ 17); 15% does not.
    edge = run_srm_simulation(ids, 10_000, 0.08, n_runs=40, seed=6)
    buggy = run_srm_simulation(ids, 10_000, 0.15, n_runs=40, seed=6)
    assert clean.detected.mean() <= 0.05
    assert clean.detected.mean() < edge.detected.mean() <= buggy.detected.mean()
    assert buggy.detected.mean() == 1.0


def test_wilson_interval():
    lo, hi = wilson_interval(50, 1000)
    assert lo < 0.05 < hi
