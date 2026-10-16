# Experiment readout: 1_clear_win

> **Simulated data.** Treatment and effect were injected; this readout demonstrates the method and makes no claim about a real product change.

**Decision: SHIP**

- Primary metric answers_posted improved by 0.08423 (+19.10%), CI [0.05797, 0.1105], p = 3.24e-10; guardrails flat.

## Checks

- SRM: counts [80000, 80000], p = 1 (pass, threshold 0.001)
- Pre-registered MDE (absolute): 0.02205

## Metrics

| role | metric | method | control | treatment | diff | rel. lift | 95% CI | p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| primary | answers_posted | cuped | 0.441 | 0.5495 | +0.08423 | +19.10% | [0.05797, 0.1105] | 3.24e-10 | good |
| guardrail | comments_posted | cuped | 1.417 | 1.536 | +0.04286 | +3.02% | [-0.02466, 0.1104] | 0.213 | flat |
| guardrail | accepted_answer_rate | jackknife_ratio | 0.3328 | 0.3376 | +0.004874 | +1.46% | [-0.009796, 0.01954] | 0.515 | flat |

_Data source: real. Injected lift: +20.0% on answers_posted; treatment users dropped: 0%; n per arm: 80,000._

_Across 200 independent draws of this scenario the decisions were: ship 96%, do_not_ship 4%. This readout is draw #0, the first whose decision matches the scenario (ship)._
