"""REST API: every endpoint, the error paths, and JSON-safety of non-finite results."""

import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from expkit.server import app  # noqa: E402

client = TestClient(app)

CONFIG = {
    "experiment": {"name": "api", "arm_column": "arm", "method": "cuped", "mde": 0.05},
    "primary": {"name": "answers", "kind": "mean", "column": "post_answers",
                "covariates": ["pre_answers"]},
    "guardrails": [{"name": "acc", "kind": "ratio", "column": "post_accepted",
                    "denominator": "post_answers"}],
}


def arms(rng, n=500, shift=0.0):
    return {"control": {"y": rng.normal(0, 1, n).tolist()},
            "treatment": {"y": rng.normal(shift, 1, n).tolist()}}


def test_health_and_methods():
    assert client.get("/health").json() == {"status": "ok"}
    names = {m["name"] for m in client.get("/methods").json()}
    assert {"welch", "cuped", "delta", "jackknife_ratio", "mannwhitney"} <= names


def test_test_endpoint_detects_shift(rng):
    r = client.post("/test", json={"method": "welch", **arms(rng, shift=0.5)})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "welch" and body["p_value"] < 1e-6
    assert body["ci_low"] < body["estimate"] < body["ci_high"]


def test_rank_test_without_ci_is_valid_json(rng):
    """Mann-Whitney has no CI (NaN); the response must still be strict JSON."""
    r = client.post("/test", json={"method": "mannwhitney", **arms(rng)})
    assert r.status_code == 200
    body = json.loads(r.text)            # strict parse: NaN would fail here
    assert body["ci_low"] is None and 0 <= body["p_value"] <= 1


def test_ratio_test_via_api(rng):
    den = rng.poisson(2, (2, 400)).astype(float)
    num = rng.binomial(den.astype(int), 0.3).astype(float)
    r = client.post("/test", json={
        "method": "jackknife_ratio", "kind": "ratio",
        "control": {"y": num[0].tolist(), "den": den[0].tolist()},
        "treatment": {"y": num[1].tolist(), "den": den[1].tolist()}})
    assert r.status_code == 200 and r.json()["method"] == "jackknife_ratio"


@pytest.mark.parametrize("payload", [
    {"method": "no_such_method"},
    {"method": "cuped"},                 # needs covariates, none given
])
def test_bad_requests_are_422(rng, payload):
    assert client.post("/test", json={**payload, **arms(rng, 50)}).status_code == 422


def test_analyze_end_to_end(population):
    df = population.sample(4000, random_state=3).reset_index(drop=True)
    df["arm"] = np.where(np.arange(len(df)) < 2000, "control", "treatment")
    records = json.loads(df.to_json(orient="records"))
    r = client.post("/analyze", json={"config": CONFIG, "records": records})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"ship", "do_not_ship", "extend", "invalid"}
    assert body["primary"]["method"] == "cuped"
    assert body["guardrails"][0]["method"] == "jackknife_ratio"


def test_analyze_rejects_missing_arm_column(population):
    records = json.loads(population.head(50).to_json(orient="records"))
    assert client.post("/analyze", json={"config": CONFIG, "records": records}).status_code == 422


def test_power_round_trip():
    n = client.post("/power", json={"sd": 3.0, "mde": 0.1}).json()
    assert n["n_control"] == n["n_treatment"] > 0
    mde = client.post("/power", json={"sd": 3.0, "n_control": n["n_control"]}).json()["mde"]
    assert mde == pytest.approx(0.1, rel=0.02)
    assert client.post("/power", json={"sd": -1.0, "mde": 0.1}).status_code == 422
