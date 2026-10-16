import json
import math

import numpy as np
import pandas as pd
import pytest

from expkit import Decision, ExperimentConfig, MetricSpec, analyze, analyze_metric, registry
from expkit._types import MetricSample, SRMResult, TestResult
from expkit.report import MetricOutcome, decide

PRIMARY = MetricSpec("answers", "mean", "post_answers", covariates=("pre_answers",))
RATIO = MetricSpec("acc", "ratio", "post_accepted", denominator="post_answers")


def assigned(population, lift=0.0, n=8000, seed=0):
    rng = np.random.default_rng(seed)
    df = population.sample(2 * n, random_state=seed).reset_index(drop=True)
    df["arm"] = np.where(np.arange(2 * n) < n, "control", "treatment")
    if lift:
        # Extra answers are accepted at the base rate; otherwise the accepted-answer-rate
        # guardrail genuinely drops and the engine (correctly) refuses to ship.
        t = df.arm == "treatment"
        extra = rng.poisson(lift * df.loc[t, "post_answers"])
        rate = df.post_accepted.sum() / df.post_answers.sum()
        df.loc[t, "post_answers"] += extra
        df.loc[t, "post_accepted"] += rng.binomial(extra, rate)
    return df


def test_guardrail_blocks_ship_when_quality_drops(population):
    df = assigned(population, n=8000)
    t = df.arm == "treatment"
    df.loc[t, "post_answers"] += np.random.default_rng(1).poisson(0.3 * df.loc[t, "post_answers"])
    r = analyze(df, cfg())
    assert r.decision is Decision.DO_NOT_SHIP
    assert "acc" in r.reasons[0]


def cfg(**kw):
    base = dict(name="t", primary=PRIMARY, guardrails=(RATIO,), method="cuped", mde=0.02)
    return ExperimentConfig(**{**base, **kw})


def test_analyze_ship_on_big_lift(population):
    r = analyze(assigned(population, lift=0.3), cfg())
    assert r.decision is Decision.SHIP
    assert r.primary.result.method == "cuped"
    assert r.guardrails[0].result.method == "jackknife_ratio"  # fell back for the ratio metric


def test_analyze_invalid_on_srm(population):
    df = assigned(population, lift=0.3)
    df = df.drop(df[df.arm == "treatment"].index[:600])
    assert analyze(df, cfg()).decision is Decision.INVALID


def test_readout_serialises(population):
    r = analyze(assigned(population), cfg())
    d = json.loads(r.to_json())
    assert d["decision"] in {x.value for x in Decision}
    assert "| primary | answers |" in r.to_markdown()


def test_unknown_arm_label_rejected(population):
    df = assigned(population)
    df.loc[0, "arm"] = "oops"
    with pytest.raises(ValueError):
        analyze(df, cfg())


def _res(est, lo, hi, p):
    return TestResult("welch", est, lo, hi, p, 0.05, 1.0, 1.0 + est, 100, 100)


SRM_OK = SRMResult((100, 100), (0.5, 0.5), 0.0, 1.0, 1e-3)
SRM_BAD = SRMResult((100, 80), (0.5, 0.5), 9.0, 1e-4, 1e-3)


@pytest.mark.parametrize("primary,guard,srm,mde,expected", [
    (_res(0.1, 0.05, 0.15, 0.001), _res(0, -1, 1, 0.9), SRM_OK, None, Decision.SHIP),
    (_res(0.1, 0.05, 0.15, 0.001), _res(-0.2, -0.3, -0.1, 0.001), SRM_OK, None, Decision.DO_NOT_SHIP),
    (_res(-0.1, -0.15, -0.05, 0.001), _res(0, -1, 1, 0.9), SRM_OK, None, Decision.DO_NOT_SHIP),
    (_res(0.0, -0.01, 0.01, 0.9), _res(0, -1, 1, 0.9), SRM_OK, 0.05, Decision.DO_NOT_SHIP),
    (_res(0.0, -0.1, 0.1, 0.9), _res(0, -1, 1, 0.9), SRM_OK, 0.05, Decision.EXTEND),
    (_res(0.1, 0.05, 0.15, 0.001), _res(0, -1, 1, 0.9), SRM_BAD, None, Decision.INVALID),
])
def test_decision_matrix(primary, guard, srm, mde, expected):
    m = MetricSpec("m", "mean", "c")
    decision, reasons = decide(MetricOutcome(m, primary), (MetricOutcome(m, guard),), srm, mde)
    assert decision is expected and reasons


def test_lower_is_better_guardrail_direction():
    m = MetricSpec("latency", "mean", "c", higher_is_better=False)
    assert MetricOutcome(m, _res(0.2, 0.1, 0.3, 0.001)).direction == "bad"


def test_config_from_toml(tmp_path):
    p = tmp_path / "e.toml"
    p.write_text('[experiment]\nname="x"\nmde=0.01\n[primary]\nname="a"\nkind="mean"\n'
                 'column="post_answers"\n[[guardrails]]\nname="r"\nkind="ratio"\n'
                 'column="post_accepted"\ndenominator="post_answers"\n')
    c = ExperimentConfig.from_toml(p)
    assert c.name == "x" and c.guardrails[0].kind == "ratio"


def test_registry_plugin_and_errors():
    @registry.register("test_dummy", kinds=("mean",), description="dummy", overwrite=True)
    def dummy(c, t, *, alpha, rng, **_):
        return TestResult("test_dummy", 0, 0, 0, 1.0, alpha, 0, 0, len(c), len(t))

    s = MetricSample(y=np.ones(3))
    assert registry.run("test_dummy", s, s, kind="mean").p_value == 1.0
    assert "test_dummy" in registry.available("mean")
    with pytest.raises(ValueError):
        registry.run("test_dummy", s, s, kind="ratio")
    with pytest.raises(ValueError):
        registry.run("cuped", s, s, kind="mean")  # no covariate
    with pytest.raises(KeyError):
        registry.get("nope")


def test_analyze_metric_and_jsonable():
    df = pd.DataFrame({"arm": ["control"] * 50 + ["treatment"] * 50,
                       "y": np.r_[np.zeros(50), np.ones(50)] + np.arange(100) % 3})
    r = analyze_metric(df, MetricSpec("y", "mean", "y"))
    d = r.to_dict()
    assert d["significant"] and json.dumps(d)
    assert math.isfinite(d["relative_lift"]) or d["relative_lift"] is None


def test_metricspec_validation():
    with pytest.raises(ValueError):
        MetricSpec("r", "ratio", "a")
    with pytest.raises(ValueError):
        MetricSpec("r", "mean", "a", denominator="b")
    with pytest.raises(ValueError):
        MetricSpec("r", "median", "a")
