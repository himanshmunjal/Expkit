"""ExpKit: an experimentation toolkit validated by A/A and injected-lift simulations.

Stable public API (everything else is internal and may change):

    from expkit import analyze, analyze_metric, ExperimentConfig, MetricSpec
    from expkit import registry          # list / run / register methods
    from expkit.stats import ...         # pure-array statistical functions
    from expkit.sim import ...           # validation harness
"""

from expkit import registry
from expkit import stats as _stats  # noqa: F401  (registers built-in methods)
from expkit._types import MetricSample, MetricSpec, SRMResult, TestResult
from expkit.api import analyze, analyze_metric, resolve_method
from expkit.config import ExperimentConfig
from expkit.report import Decision, Readout

__version__ = "0.1.0"

__all__ = [
    "registry", "MetricSample", "MetricSpec", "SRMResult", "TestResult", "analyze",
    "analyze_metric", "resolve_method", "ExperimentConfig", "Decision", "Readout", "__version__",
]
