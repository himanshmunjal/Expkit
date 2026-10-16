"""Experiment configuration: one TOML file per experiment, pre-registered
before the data is looked at. TOML is in the standard library (tomllib), so
configs need no extra dependency and can be produced by any language.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from expkit._types import MetricSpec


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    primary: MetricSpec
    guardrails: tuple[MetricSpec, ...] = ()
    arm_column: str = "arm"
    control: str = "control"
    treatment: str = "treatment"
    expected_split: tuple[float, float] = (0.5, 0.5)
    alpha: float = 0.05
    method: str = "welch"
    mde: float | None = None          # relative to the control mean, e.g. 0.02 = 2%
    srm_threshold: float = 1e-3
    simulated: bool = False
    method_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if len(self.expected_split) != 2 or min(self.expected_split) <= 0:
            raise ValueError("expected_split must be two positive shares")
        if self.control == self.treatment:
            raise ValueError("control and treatment labels must differ")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        exp = dict(d.get("experiment", {}))
        if "primary" not in d:
            raise ValueError("config needs a [primary] metric table")
        return cls(
            name=exp.pop("name", "experiment"),
            primary=MetricSpec.from_dict(d["primary"]),
            guardrails=tuple(MetricSpec.from_dict(g) for g in d.get("guardrails", [])),
            expected_split=tuple(exp.pop("expected_split", (0.5, 0.5))),
            method_options=d.get("method_options", {}),
            **exp,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))
