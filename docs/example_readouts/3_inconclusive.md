# Experiment readout: 3_inconclusive

> **Simulated data.** Treatment and effect were injected; this readout demonstrates the method and makes no claim about a real product change.

**Decision: EXTEND**

- Inconclusive: CI [-0.1264, 0.09823] includes both zero and effects of MDE size. Extend the test (full weekly cycles) or redesign for more power.

## Checks

- SRM: counts [1500, 1500], p = 1 (pass, threshold 0.001)
- Pre-registered MDE (absolute): 0.01807

## Metrics

| role | metric | method | control | treatment | diff | rel. lift | 95% CI | p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| primary | answers_posted | cuped | 0.3613 | 0.2907 | -0.01407 | -3.89% | [-0.1264, 0.09823] | 0.806 | flat |
| guardrail | comments_posted | cuped | 1.211 | 0.9507 | -0.0836 | -6.91% | [-0.3475, 0.1803] | 0.535 | flat |
| guardrail | accepted_answer_rate | jackknife_ratio | 0.3173 | 0.2706 | -0.0467 | -14.72% | [-0.1178, 0.02436] | 0.196 | flat |

_Data source: real. Injected lift: +3.0% on answers_posted; treatment users dropped: 0%; n per arm: 1,500._

_Across 200 independent draws of this scenario the decisions were: extend 88%, do_not_ship 10%, ship 2%. This readout is draw #1, the first whose decision matches the scenario (extend)._
