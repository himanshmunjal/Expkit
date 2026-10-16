"""Optional REST API so non-Python services (dashboards, a Node/Go backend,
an internal experiment platform) can call the same statistics.

    pip install 'expkit[api]'
    uvicorn expkit.server:app --port 8000

All endpoints are stateless and JSON-in / JSON-out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from expkit import registry
from expkit._types import MetricSample
from expkit.api import analyze
from expkit.config import ExperimentConfig
from expkit.stats import power as pw

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError("install the extra: pip install 'expkit[api]'") from e

app = FastAPI(title="ExpKit", version="0.1.0")


class ArmData(BaseModel):
    y: list[float]
    den: list[float] | None = None
    x: list[float] | list[list[float]] | None = None


class TestRequest(BaseModel):
    method: str = "welch"
    kind: str = "mean"
    alpha: float = 0.05
    control: ArmData
    treatment: ArmData
    options: dict[str, Any] = {}
    seed: int = 0


class AnalyzeRequest(BaseModel):
    config: dict[str, Any]           # same structure as the TOML config
    records: list[dict[str, Any]]    # one row per unit, including the arm column


class PowerRequest(BaseModel):
    sd: float
    baseline_mean: float | None = None
    mde: float | None = None
    n_control: int | None = None
    relative: bool = False
    alpha: float = 0.05
    power: float = 0.8
    ratio: float = 1.0
    variance_reduction: float = 0.0


def _sample(a: ArmData) -> MetricSample:
    return MetricSample(y=np.asarray(a.y), den=None if a.den is None else np.asarray(a.den),
                        x=None if a.x is None else np.asarray(a.x))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/methods")
def methods() -> list[dict[str, Any]]:
    return [{"name": n, "kinds": sorted(registry.get(n).kinds),
             "needs_covariate": registry.get(n).needs_covariate,
             "description": registry.get(n).description} for n in registry.available()]


@app.post("/test")
def run_test(req: TestRequest) -> dict[str, Any]:
    try:
        res = registry.run(req.method, _sample(req.control), _sample(req.treatment),
                           kind=req.kind, alpha=req.alpha,
                           rng=np.random.default_rng(req.seed), **req.options)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return res.to_dict()


@app.post("/analyze")
def run_analyze(req: AnalyzeRequest) -> dict[str, Any]:
    try:
        readout = analyze(pd.DataFrame.from_records(req.records),
                          ExperimentConfig.from_dict(req.config))
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return readout.to_dict()


@app.post("/power")
def run_power(req: PowerRequest) -> dict[str, Any]:
    kw = dict(baseline_mean=req.baseline_mean, relative=req.relative, alpha=req.alpha,
              ratio=req.ratio, variance_reduction=req.variance_reduction)
    out: dict[str, Any] = {}
    try:
        if req.mde is not None:
            out["n_control"], out["n_treatment"] = pw.required_sample_size(
                req.sd, req.mde, power=req.power, **kw)
        if req.n_control is not None:
            out["mde"] = pw.minimum_detectable_effect(req.sd, req.n_control, power=req.power, **kw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return out
