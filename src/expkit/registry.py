"""Method registry: the single extension point for statistical methods.

Every method has the same signature::

    fn(control: MetricSample, treatment: MetricSample, *, alpha: float,
       rng: np.random.Generator, **options) -> TestResult

The simulation harness, the readout generator, the CLI and the REST API only
talk to methods through this registry, so adding a method (in this repo or in a
third-party package via the ``expkit.methods`` entry-point group) makes it
available everywhere at once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

import numpy as np

from expkit._types import METRIC_KINDS, MetricSample, TestResult

MethodFn = Callable[..., TestResult]


@dataclass(frozen=True)
class MethodEntry:
    name: str
    fn: MethodFn
    kinds: frozenset[str]
    needs_covariate: bool
    description: str


_REGISTRY: dict[str, MethodEntry] = {}
_PLUGINS_LOADED = False


def register(
    name: str,
    *,
    kinds: tuple[str, ...] = ("mean", "binary"),
    needs_covariate: bool = False,
    description: str = "",
    overwrite: bool = False,
) -> Callable[[MethodFn], MethodFn]:
    bad = set(kinds) - set(METRIC_KINDS)
    if bad:
        raise ValueError(f"unknown metric kinds {bad}")

    def deco(fn: MethodFn) -> MethodFn:
        if name in _REGISTRY and not overwrite:
            raise ValueError(f"method {name!r} is already registered")
        _REGISTRY[name] = MethodEntry(
            name=name,
            fn=fn,
            kinds=frozenset(kinds),
            needs_covariate=needs_covariate,
            description=description or _first_line(fn.__doc__),
        )
        return fn

    return deco


def _first_line(doc: str | None) -> str:
    lines = (doc or "").strip().splitlines()
    return lines[0] if lines else ""


def load_plugins() -> None:
    """Import methods published by other packages under ``expkit.methods``."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for ep in entry_points(group="expkit.methods"):
        obj = ep.load()
        if callable(obj):
            obj()  # a plugin's register() function


def get(name: str) -> MethodEntry:
    load_plugins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; available: {sorted(_REGISTRY)}") from None


def available(kind: str | None = None) -> list[str]:
    load_plugins()
    return sorted(n for n, e in _REGISTRY.items() if kind is None or kind in e.kinds)


def run(
    name: str,
    control: MetricSample,
    treatment: MetricSample,
    *,
    kind: str,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    **options: Any,
) -> TestResult:
    entry = get(name)
    if kind not in entry.kinds:
        raise ValueError(f"method {name!r} does not support {kind!r} metrics")
    if entry.needs_covariate and (control.x is None or treatment.x is None):
        raise ValueError(f"method {name!r} needs a pre-period covariate")
    if rng is None:
        rng = np.random.default_rng()
    return entry.fn(control, treatment, alpha=alpha, rng=rng, **options)
