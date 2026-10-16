# FLOW.md: how ExpKit runs, how it was built, and how to study it

Four views of the same project:

1. **Execution flow**: the order in which you run things, end to end.
2. **Call flow**: which files execute, in order, inside one command.
3. **Creation flow**: the order in which files were written, and why that order.
4. **Study flow**: the order in which to read the code to understand it.

---

## 1. Execution flow (run order)

```
 ┌─────────────┐   ┌────────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌─────────────────┐
 │ 0. install  │──▶│ 1. verify data │──▶│ 2. extract    │──▶│ 3. profile       │──▶│ 4. validate     │
 │ make install│   │ data/verify.sql│   │ data/extract. │   │ data/prepare.py  │   │ experiments/*.py│
 └─────────────┘   └────────────────┘   │ sql → parquet │   └──────────────────┘   └────────┬────────┘
                                         └───────────────┘                                   │
          ┌──────────────────────────────────────────────────────────────────────────────────┘
          ▼
 ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
 │ 5. readouts      │──▶│ 6. serve / integrate │──▶│ 7. analyse a real test   │
 │ example_readouts │   │ app · API · CLI      │   │ expkit analyze --config  │
 └──────────────────┘   └──────────────────────┘   └──────────────────────────┘
```

| Step | Command | Reads | Writes |
|---|---|---|---|
| 0 | `make install` (= `pip install -e ".[all,dev]"`) | `pyproject.toml` | env |
| 0b | `make test` | `tests/*` | coverage report |
| 1 | `make verify` | `data/verify.sql` → BigQuery | stdout: tables, freshness → **choose T0** |
| 2 | `make extract T0=YYYY-MM-DD` | `data/extract.sql` | `data/users_metrics.parquet` |
| 3 | `make profile` | the parquet | `results/profile.json` (zero share, tails, ρ²) |
| 4 | `make experiments JOBS=-1` runs, in order: | the parquet | `results/<name>_{raw.parquet,summary.csv,meta.json}` |
| 4.1 | `experiments/aa_calibration.py` | | FPR and KS per metric × method × n (`calibrated_at_alpha`, strict `calibrated`) |
| 4.2 | `experiments/injected_lift.py` | | power, bias, coverage vs lift × n |
| 4.3 | `experiments/cuped.py` | | empirical VR vs theoretical R² |
| 4.4 | `experiments/ratio_metrics.py` | | delta vs jackknife vs bootstrap vs naive |
| 4.5 | `experiments/peeking.py` | | naive vs OBF/Pocock/mSPRT FPR and stop time; also `results/msprt_tau_sweep.csv` |
| 4.6 | `experiments/srm.py` | | detection rate vs bug severity |
| 4.7 | `experiments/multiple_testing.py` | | FWER: none / Bonferroni / Holm / BH |
| 4.8 | `experiments/winsorization.py` | | bias-variance of capping (A/A and +20% lift at 20k/arm) |
| 5 | `make readouts` | the parquet | `docs/example_readouts/*.md` |
| 6 | `make app` / `make api` / `docker build` | | UI on :8501 / REST on :8000 |
| 7 | `expkit analyze --data exp.parquet --config configs/x.toml` | your assigned data | readout (md/json) |

No credentials? Run everything on the synthetic twin: `make experiments-quick`, or pass
`--data synthetic` to any experiment script. Results will be labelled `synthetic` and written
to `results/synthetic/` (example readouts included), never over the real `results/` (D-055).

---

## 2. Call flow (what executes inside a command)

### 2a. `expkit analyze` (a real experiment readout)
```
cli.py:main ─▶ cli.cmd_analyze
   ├─ config.ExperimentConfig.from_toml          (TOML → MetricSpec objects)
   └─ api.analyze(df, config)
        ├─ stats/srm.srm_test                     ① SRM gate
        ├─ api.analyze_metric (primary, each guardrail)
        │    ├─ api.resolve_method               (cuped → jackknife_ratio for ratio, → welch if no covariate)
        │    ├─ _types.MetricSpec.sample          (DataFrame → MetricSample arrays)
        │    └─ registry.run ─▶ stats/methods.py adapter ─▶ stats/<cuped|delta|ttest>.py
        └─ report/readout.decide                  ② decision matrix
             └─ Readout.to_markdown / to_json
```

### 2b. `experiments/aa_calibration.py` (validation)
```
_common.load_population ─▶ io/sources.ParquetSource | SyntheticSource ─▶ io/schema.validate
sim/harness.run_simulation(spec)
   ├─ MetricSpec.sample (once, whole population)
   └─ sim/_parallel.run_many  (SeedSequence.spawn → one RNG per run; joblib if --jobs)
        └─ per run: rng.choice(2n) ─▶ effects/inject ─▶ registry.run(method) ─▶ row
sim/harness.summarize  (Wilson CI, KS, bias, coverage) ─▶ _common.save
```

### 2c. `experiments/peeking.py`
```
sim/peeking.run_peeking_simulation
   ├─ sequential/alpha_spending.group_sequential_boundaries  (MC, cached once)
   └─ per run: cumulative sums → z at each look → naive / OBF / Pocock / mSPRT crossings
               └─ sequential/msprt.msprt_pvalues
```

### 2d. Import graph (dependencies only point downward)
```
cli / server / app / experiments
            │
          api.py ── config.py
            │
   sim/ ────┼──── report/
            │
 stats/ · sequential/ · effects/ · assign/ · registry.py
            │
        _types.py
            │
     io/ · catalog.py
```

---

## 3. Creation flow (order files were written, and why)

Contracts first, then the pure maths, then orchestration, then the edges. Each layer was
written only after the layer it depends on existed.

| # | File(s) | Why at this point |
|---|---|---|
| 1 | `pyproject.toml` | defines the package, its extras and the CLI entry point |
| 2 | `src/expkit/_types.py` | data contracts that every later module depends on |
| 3 | `src/expkit/registry.py` | the extension point; methods register into it |
| 4 | `stats/ttest.py` | the baseline test that other methods reuse |
| 5 | `stats/bootstrap.py` | non-parametric comparison for skewed data |
| 6 | `stats/cuped.py` | variance reduction (reuses Welch) |
| 7 | `stats/delta.py` | ratio metrics: delta, jackknife (default), naive |
| 8 | `stats/srm.py`, `stats/multiple.py`, `stats/power.py` | gates, corrections, design |
| 9 | `stats/bayes.py`, `stats/winsor.py` | extensions from the brief |
| 10 | `stats/methods.py`, `stats/__init__.py` | register all of the above in the registry |
| 11 | `sequential/alpha_spending.py`, `sequential/msprt.py`, `sequential/__init__.py` | monitoring methods |
| 12 | `assign/bucketing.py`, `assign/__init__.py` | hash / random / stratified assignment |
| 13 | `effects/inject.py`, `effects/__init__.py` | known-truth generator |
| 14 | `io/schema.py` | the data contract |
| 15 | `io/sources.py`, `io/__init__.py` | Parquet / CSV / BigQuery / Synthetic sources |
| 16 | `catalog.py` | Stack Overflow metric definitions |
| 17 | `sim/_parallel.py` | deterministic, parallel run executor |
| 18 | `sim/harness.py` | A/A and injected-lift engine |
| 19 | `sim/peeking.py`, `sim/srm.py`, `sim/__init__.py` | specialised simulations |
| 20 | `report/readout.py`, `report/__init__.py` | decision rules and rendering |
| 21 | `config.py` | TOML experiment config |
| 22 | `api.py`, `__init__.py` | public facade |
| 23 | `cli.py` | shell integration |
| 24 | `server.py` | REST integration |
| 25 | `data/verify.sql`, `data/extract.sql`, `data/prepare.py` | the real-data path |
| 26 | `experiments/_common.py` + 8 experiment scripts | one per acceptance test |
| 27 | `experiments/example_readouts.py` | the four readouts in §8 of the brief |
| 28 | `app/streamlit_app.py` | UI |
| 29 | `configs/example_experiment.toml` | config template |
| 30 | `tests/*` (10 files) | unit, property and statistical tests |
| 31 | `Makefile`, `Dockerfile`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `.gitignore` | operations |
| 32 | `docs/decision_framework.md` | how to make a decision |
| 33 | `DECISION.md`, `FLOW.md`, `README.md` | documentation |

---

## 4. Study flow (recommended reading order)

Read bottom-up: each step only uses ideas from earlier steps.

| Stage | Read | What to understand | Check yourself |
|---|---|---|---|
| **1. Problem** | `../03_Analyst_experimentation_toolkit.md`, `README.md` | why A/A plus injected lift validates a platform | explain "semi-synthetic" in one sentence |
| **2. Decisions** | `DECISION.md` sections A-B | the layering and the data contracts | sketch the layer diagram from memory |
| **3. Contracts** | `_types.py`, `io/schema.py`, `catalog.py` | MetricSample, MetricSpec, TestResult; the table schema | why is `pre_active` not a covariate? |
| **4. Data** | `data/verify.sql` → `data/extract.sql` → `io/sources.py` | windows, population, leakage rules | why use votes for acceptance dates? |
| **5. Core stats** | `stats/ttest.py` → `stats/power.py` → `stats/srm.py` | Welch, sample size, SRM | derive n = 1570 in `test_srm_multiple_power.py` |
| **6. Variance reduction** | `stats/cuped.py`, `tests/test_cuped.py` | θ = cov/var, VR = ρ², pooled θ | why must X be pre-period? |
| **7. Ratio metrics** | `stats/delta.py`, `stats/bootstrap.py` | delta-method vs jackknife variance; ratio of sums vs mean of ratios | why does delta over-reject when a few users hold most answers? |
| **8. Multiplicity** | `stats/multiple.py` | Holm vs BH | FWER vs FDR |
| **9. Sequential** | `sequential/alpha_spending.py`, `sequential/msprt.py`, `tests/test_sequential.py` | why peeking inflates FPR; two fixes | what does OBF spend at t = 0.2? |
| **10. Extensibility** | `registry.py`, `stats/methods.py` | how one registration reaches every surface | add a toy method in a test |
| **11. Ground truth** | `assign/bucketing.py`, `effects/inject.py` | hash bucketing; type-preserving injection | why are covariates never injected? |
| **12. Validation engine** | `sim/_parallel.py` → `sim/harness.py` → `sim/peeking.py` → `sim/srm.py` | runs, seeds, summaries | why does parallel output equal serial output? |
| **13. Decisions** | `report/readout.py`, `docs/decision_framework.md` | the 6-rule matrix | why does SRM outrank everything? |
| **14. Integration** | `config.py` → `api.py` → `cli.py` → `server.py` → `app/streamlit_app.py` | thin wrappers over one facade | call `/analyze` with JSON |
| **15. Experiments** | `experiments/_common.py`, then the scripts in run order (§1 step 4) | how each acceptance test is measured | predict each result before running it |
| **16. Ops** | `Makefile`, `ci.yml`, `Dockerfile` | reproducibility and CI | run `make experiments-quick` |
