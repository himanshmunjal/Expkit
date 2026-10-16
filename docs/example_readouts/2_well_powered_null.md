# Experiment readout: 2_well_powered_null

> **Simulated data.** Treatment and effect were injected; this readout demonstrates the method and makes no claim about a real product change.

**Decision: DO NOT SHIP**

- Well-powered null: CI [-0.03087, 0.01676] excludes effects as large as the MDE (+-0.04504). Stop investing in this variant.

## Checks

- SRM: counts [80000, 80000], p = 1 (pass, threshold 0.001)
- Pre-registered MDE (absolute): 0.04504

## Metrics

| role | metric | method | control | treatment | diff | rel. lift | 95% CI | p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| primary | answers_posted | cuped | 0.4504 | 0.4556 | -0.007055 | -1.57% | [-0.03087, 0.01676] | 0.561 | flat |
| guardrail | comments_posted | cuped | 1.471 | 1.489 | -0.01457 | -0.99% | [-0.08455, 0.05541] | 0.683 | flat |
| guardrail | accepted_answer_rate | jackknife_ratio | 0.3397 | 0.3354 | -0.004325 | -1.27% | [-0.02035, 0.0117] | 0.597 | flat |

_Data source: real. Injected lift: +0.0% on answers_posted; treatment users dropped: 0%; n per arm: 80,000._

_Across 200 independent draws of this scenario the decisions were: do_not_ship 94%, ship 4%, extend 2%. This readout is draw #0, the first whose decision matches the scenario (do_not_ship)._
