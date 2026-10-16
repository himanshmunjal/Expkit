"""Turn test results into a ship / do-not-ship / extend / invalid decision.

The rules are applied in a fixed order (see docs/decision_framework.md):
  1. SRM fails                          -> INVALID (no other number is trusted)
  2. any guardrail significantly harmed -> DO_NOT_SHIP
  3. primary significant, good way      -> SHIP
  4. primary significant, bad way       -> DO_NOT_SHIP
  5. CI inside (-MDE, +MDE)             -> DO_NOT_SHIP (well-powered null)
  6. otherwise                          -> EXTEND (inconclusive)
Guardrails are deliberately not multiplicity-corrected: a missed harm costs
more than a false alarm, which only triggers a closer look.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from expkit._types import MetricSpec, SRMResult, TestResult, _jsonable


class Decision(StrEnum):
    SHIP = "ship"
    DO_NOT_SHIP = "do_not_ship"
    EXTEND = "extend"
    INVALID = "invalid"


@dataclass(frozen=True)
class MetricOutcome:
    metric: MetricSpec
    result: TestResult

    @property
    def direction(self) -> str:
        """'good', 'bad' or 'flat' (not significant), given higher_is_better."""
        if not self.result.significant:
            return "flat"
        up = self.result.estimate > 0
        return "good" if up == self.metric.higher_is_better else "bad"


@dataclass(frozen=True)
class Readout:
    experiment: str
    decision: Decision
    reasons: tuple[str, ...]
    srm: SRMResult
    primary: MetricOutcome
    guardrails: tuple[MetricOutcome, ...] = ()
    mde_abs: float | None = None
    simulated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        def outcome(o: MetricOutcome) -> dict[str, Any]:
            return {"metric": o.metric.name, "kind": o.metric.kind,
                    "higher_is_better": o.metric.higher_is_better,
                    "direction": o.direction, **o.result.to_dict()}

        return _jsonable({
            "experiment": self.experiment, "decision": self.decision.value,
            "reasons": list(self.reasons), "simulated": self.simulated,
            "srm": self.srm.to_dict(), "mde_abs": self.mde_abs,
            "primary": outcome(self.primary),
            "guardrails": [outcome(g) for g in self.guardrails],
            "notes": list(self.notes),
        })

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, **kw)

    def to_markdown(self) -> str:
        return render_markdown(self)


def decide(
    primary: MetricOutcome,
    guardrails: tuple[MetricOutcome, ...],
    srm: SRMResult,
    mde_abs: float | None,
) -> tuple[Decision, tuple[str, ...]]:
    if srm.mismatch:
        return Decision.INVALID, (
            f"Sample-ratio mismatch: counts {srm.observed} vs expected shares "
            f"{srm.expected_share} (p = {srm.p_value:.2g} < {srm.threshold}). Assignment or "
            "logging is broken; no metric result can be trusted. Investigate before deciding.",
        )
    harmed = [g for g in guardrails if g.direction == "bad"]
    if harmed:
        names = ", ".join(g.metric.name for g in harmed)
        return Decision.DO_NOT_SHIP, (f"Guardrail(s) significantly harmed: {names}.",)

    r = primary.result
    ci = f"[{r.ci_low:.4g}, {r.ci_high:.4g}]"
    if primary.direction == "good":
        return Decision.SHIP, (
            f"Primary metric {primary.metric.name} improved by {r.estimate:.4g} "
            f"({_pct(r.relative_lift)}), CI {ci}, p = {r.p_value:.3g}; guardrails flat.",
        )
    if primary.direction == "bad":
        return Decision.DO_NOT_SHIP, (
            f"Primary metric {primary.metric.name} got worse by {r.estimate:.4g}, CI {ci}.",
        )
    if mde_abs is not None and math.isfinite(r.ci_low) and -mde_abs < r.ci_low and r.ci_high < mde_abs:
        return Decision.DO_NOT_SHIP, (
            f"Well-powered null: CI {ci} excludes effects as large as the MDE "
            f"(+-{mde_abs:.4g}). Stop investing in this variant.",
        )
    return Decision.EXTEND, (
        f"Inconclusive: CI {ci} includes both zero and effects of MDE size. "
        "Extend the test (full weekly cycles) or redesign for more power.",
    )


def _pct(x: float) -> str:
    return "n/a" if not math.isfinite(x) else f"{100 * x:+.2f}%"


def render_markdown(r: Readout) -> str:
    lines = [f"# Experiment readout: {r.experiment}", ""]
    if r.simulated:
        lines += ["> **Simulated data.** Treatment and effect were injected; this readout "
                  "demonstrates the method and makes no claim about a real product change.", ""]
    lines += [f"**Decision: {r.decision.value.replace('_', ' ').upper()}**", ""]
    lines += [f"- {reason}" for reason in r.reasons] + [""]
    lines += ["## Checks", "",
              f"- SRM: counts {list(r.srm.observed)}, p = {r.srm.p_value:.3g} "
              f"({'FAIL' if r.srm.mismatch else 'pass'}, threshold {r.srm.threshold})"]
    if r.mde_abs is not None:
        lines.append(f"- Pre-registered MDE (absolute): {r.mde_abs:.4g}")
    lines += ["", "## Metrics", "",
              "| role | metric | method | control | treatment | diff | rel. lift | 95% CI | p | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for role, o in [("primary", r.primary), *(("guardrail", g) for g in r.guardrails)]:
        t = o.result
        lines.append(
            f"| {role} | {o.metric.name} | {t.method} | {t.control_mean:.4g} | "
            f"{t.treatment_mean:.4g} | {t.estimate:+.4g} | {_pct(t.relative_lift)} | "
            f"[{t.ci_low:.4g}, {t.ci_high:.4g}] | {t.p_value:.3g} | {o.direction} |"
        )
    if r.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in r.notes]
    return "\n".join(lines) + "\n"
