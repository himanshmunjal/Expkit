# Experiment readout: 4_srm_failure

> **Simulated data.** Treatment and effect were injected; this readout demonstrates the method and makes no claim about a real product change.

**Decision: INVALID**

- Sample-ratio mismatch: counts (20000, 19013) vs expected shares (0.5, 0.5) (p = 5.8e-07 < 0.001). Assignment or logging is broken; no metric result can be trusted. Investigate before deciding.

## Checks

- SRM: counts [20000, 19013], p = 5.82e-07 (FAIL, threshold 0.001)
- Pre-registered MDE (absolute): 0.02292

## Metrics

| role | metric | method | control | treatment | diff | rel. lift | 95% CI | p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| primary | answers_posted | cuped | 0.4585 | 0.502 | +0.07524 | +16.41% | [0.02485, 0.1256] | 0.00343 | good |
| guardrail | comments_posted | cuped | 1.415 | 1.516 | +0.08772 | +6.20% | [-0.04611, 0.2216] | 0.199 | flat |
| guardrail | accepted_answer_rate | jackknife_ratio | 0.3469 | 0.3382 | -0.008742 | -2.52% | [-0.04144, 0.02396] | 0.6 | flat |

_Data source: real. Injected lift: +8.0% on answers_posted; treatment users dropped: 5%; n per arm: 20,000._

_Across 200 independent draws of this scenario the decisions were: invalid 100%. This readout is draw #0, the first whose decision matches the scenario (invalid)._
