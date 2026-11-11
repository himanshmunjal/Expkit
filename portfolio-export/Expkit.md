---
intro: ExpKit is a Python A/B-testing toolkit that I validated with thousands of simulated A/A tests and known-lift injections on real behaviour from 177,441 Stack Overflow users.
caption: Stack Overflow data flows through an SQL extract and Parquet cache into the simulation harness and on to the validation report, while experiment data uses the same method registry via the analysis facade.
---

## About

Most A/B-testing code assumes its statistics are right. I built ExpKit to check. The core question: **does each method actually give a 5% false-positive rate, the power it promises, and an unbiased estimate on real, heavy-tailed user data?**

The data is semi-synthetic on purpose. User behaviour is real. Assignment and effects are simulated, so I always know the true answer. Every result is a claim about methodology, never about business lift.

## Architecture

The validation path starts at **Stack Overflow on BigQuery**. An **SQL extract** writes one row per user to a **Parquet cache**, so the warehouse is queried once. The **Simulation harness** draws thousands of random splits, injects a known lift or none, and calls each test through the **Method registry** (arrays in, `TestResult` out). Summaries go to **Results files**, and the **Validation report** is built only from those files. A **Synthetic twin** with the same schema stands in for the real data in CI.

The analysis path reuses the same core. **Experiment data + config** goes to the **Analysis facade**, which runs an SRM gate, tests each metric through the **Method registry** and applies fixed decision rules. The readout goes out as Markdown or JSON through the **Interfaces**: a CLI, a FastAPI REST API and a Streamlit app.

## How it's implemented

### One registry for every method

Each method is registered once with one signature and then appears in the harness, CLI, API and app, so a new method can be put through the same A/A checks.

### A statistics core that only sees arrays

The `stats/` and `sequential/` modules take numpy arrays and return frozen dataclasses. The maths never imports pandas, files or web code, which kept 5,000-run simulations fast and results JSON-safe.

### Effect injection that keeps the data's shape

Lifts are injected by type: Poisson thickening or binomial thinning for counts, flips for binary metrics. Counts stay integers and zeros stay zero, so the heavy tails are kept. Pre-period covariates are never touched.

### Reproducible, parallel runs

Each run gets its own child seed from `SeedSequence.spawn`, so results are identical serially or across joblib workers.

### Changing defaults when the data said so

The delta method over-rejected on accepted-answer rate, because a few heavy answerers hold most of the denominator. I added a leave-one-user-out jackknife test and made it the ratio default. Each such change is logged in `DECISION.md`.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | `tomllib` in the standard library |
| Statistics | numpy, scipy | Short, tested formulas, no statsmodels |
| Data | BigQuery, pandas, pyarrow | One cached extract, then all local |
| Storage | Parquet and CSV files | Reproducible from SQL, scripts and seeds |
| Interfaces | argparse, FastAPI, Streamlit | Thin wrappers over one facade |
| Infra/CI | Docker, GitHub Actions, ruff | CI runs every experiment on synthetic data |
| Testing | pytest, hypothesis | Textbook, property and seeded statistical tests |

## Results

Real extract, 2,000 to 5,000 simulated runs per cell. False positives are A/A rates against a nominal 5%.

| Measure | Baseline | Best |
|---|---|---|
| Ratio metric false positives, 100 per arm | 20.6% (delta) | **5.9%** (jackknife) |
| Ratio metric false positives, 1,000 per arm | 10.4% (delta) | **6.4%** (jackknife) |
| False positives with 28 daily peeks | 24.1% (naive) | **4.1%** (O'Brien-Fleming) |
| Family-wise error, 20 null metrics | 62.7% (none) | **6.0%** (Holm) |
| Variance reduction, answers posted | 0% (Welch) | **62.9%** (CUPED) |
| Power, +20% lift at 20,000 per arm | 55.4% (Welch) | **92.0%** (CUPED) |
| Bias of that estimated lift | -56.0% (p99 capping) | **-1.5%** (CUPED) |

The most useful finding was negative: the delta method I planned as the default was miscalibrated here below about 10,000 users per arm. CUPED also beat capping outliers, with similar power and no bias.

## What I'd do differently

- Peeking is modelled as users arriving over time, not metrics growing within the 28 days.
- The jackknife still shows 6–7% false positives at 500 to 1,000 users per arm.
- I changed the calibration pass criterion after seeing results. Both columns are kept, but next time I'd fix it before the first run.
- Next I'd add CUPAC (a model-predicted covariate) and checks for interference between users.
