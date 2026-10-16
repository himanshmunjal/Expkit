# DECISION.md: architecture and methodology decision log

Each entry records **what** was decided, **why**, what was **rejected**, and the
**consequence**. Add new entries at the end with the next number. To reverse a
decision, add a new entry that supersedes the old one rather than editing it.

Status: `accepted` · `provisional` (revisit once real data is in) · `superseded`.

---

## A. Architecture

### D-001 · Installable `src/`-layout package · accepted
- **Decision:** ship ExpKit as a pip-installable package (`pyproject.toml`, hatchling,
  `src/expkit/`), with experiments, app and docs *outside* the package.
- **Why:** an installable package is the precondition for every integration (other repos,
  notebooks, Docker, CI). The `src/` layout forces tests to run against the installed package,
  which catches packaging bugs.
- **Rejected:** a flat folder of notebooks/scripts. It can't be imported by other
  applications, and it can't be versioned.

### D-002 · Layered architecture with one-way dependencies · accepted
```
 integrations   cli.py · server.py · app/ · experiments/
       │
 facade         api.py  (analyze, analyze_metric)      config.py
       │
 orchestration  sim/ (harness, peeking, srm)            report/ (decision rules)
       │
 domain         stats/ · sequential/ · effects/ · assign/      registry.py
       │
 contracts      _types.py (MetricSample, MetricSpec, TestResult, SRMResult)
       │
 data           io/ (DataSource protocol, schema, sources)   catalog.py
```
- **Rule:** a layer imports only from layers **below** it. `stats/` never imports pandas
  frames, files, BigQuery or the CLI.
- **Why:** each layer can be tested on its own and replaced on its own (for example, a new
  warehouse, a new UI or a new method), and the dependency graph has no cycles.

### D-003 · The statistics core works on numpy arrays, not DataFrames · accepted
- **Decision:** every function in `stats/` and `sequential/` takes plain arrays and returns
  a frozen dataclass. pandas appears only in `io/`, `MetricSpec.sample()`, `sim/` and `api.py`.
- **Why:** callers using polars, Spark `.toPandas()`, raw lists from a REST payload or another
  language through the API can all call the maths. Arrays are also about 10× faster than
  pandas in the inner loop of a 5,000-run simulation.
- **Rejected:** methods that take a DataFrame plus column names. That ties every caller to
  pandas and to our column naming.

### D-004 · Uniform, serialisable result contracts · accepted
- **Decision:** all frequentist methods return `TestResult` (estimate = treatment − control,
  CI, p, means, n, `extra` dict). All results expose `.to_dict()`, which produces JSON-safe output
  (numpy scalars converted, NaN/inf mapped to `null`).
- **Why:** the harness, decision engine, CLI, REST API and app consume results without
  per-method special cases. JSON-safe output is what makes cross-language integration cheap.

### D-005 · Method registry with entry-point plugins · accepted
- **Decision:** methods are registered by name (`@registry.register("cuped", kinds=…,
  needs_covariate=…)`) with one signature: `fn(control, treatment, *, alpha, rng, **opts)`.
  Third-party packages can add methods through the `expkit.methods` entry-point group.
- **Why:** this is the main scalability lever. A new method automatically shows up in the
  simulation harness, the CLI (`expkit methods`), the REST API (`/methods`, `/test`) and the
  config (`method = "…"`), with no changes to any of them.
- **Consequence:** `registry.run` validates metric kind and covariate requirements centrally,
  so a mis-configured method fails loudly and early.

### D-006 · Data access behind a `DataSource` protocol and a schema contract · accepted
- **Decision:** a source is anything with `load() -> DataFrame` that passes
  `io.schema.validate_user_metrics`. The source implementations are Parquet, CSV, BigQuery
  and Synthetic.
- **Why:** swapping BigQuery for Snowflake, Postgres or DuckDB is a single ~20-line class.
  The schema check (unique ids, no nulls, non-negative counts, 0/1 binaries,
  accepted ≤ answers) stops bad extracts before they reach the statistics.
- **Note:** `expkit analyze` deliberately does **not** enforce the Stack Overflow schema.
  Any assigned table that has the config's columns can be analysed. This keeps the analysis
  path generic for other products.

### D-007 · Synthetic data source as an offline twin · accepted
- **Decision:** `SyntheticSource` produces the same schema with the same pathologies:
  gamma-mixed Poisson heavy tails, zero inflation, pre/post correlation and churn.
- **Why:** tests, CI, the app and new contributors work without GCP credentials or cost.
- **Guard rail:** experiment outputs carry a `data_source` column and a stderr warning, so
  synthetic numbers can never be quoted as findings (see D-034).

### D-008 · Public facade `expkit.api` · accepted
- **Decision:** integrations call `analyze(df, config)` and `analyze_metric(df, metric, …)`.
  Everything else is internal.
- **Why:** internals can be refactored without breaking callers. `resolve_method` falls back
  to the safe default for each metric kind (ratio → `delta`, no covariate → `welch`) and
  records the fallback in the readout notes.

### D-009 · Integration surfaces are thin wrappers over the facade · accepted
| Surface | For | Entry |
|---|---|---|
| Python API | notebooks, other Python services | `from expkit import analyze` |
| CLI | shell, cron, Airflow, CI | `expkit analyze/power/simulate/peek/extract/synth/methods` |
| REST (FastAPI) | non-Python apps, dashboards, internal platforms | `uvicorn expkit.server:app` (`/test`, `/analyze`, `/power`, `/methods`) |
| Streamlit | humans | `streamlit run app/streamlit_app.py` |
| Docker | deployment | `docker run expkit` (API) |
- **Rule:** no statistics live in a wrapper. That way all surfaces give identical answers.

### D-010 · Optional dependency extras · accepted
- **Decision:** the core depends only on numpy, scipy, pandas and pyarrow. `bigquery`,
  `parallel` (joblib), `app` (streamlit), `api` (fastapi) and `dev` are extras.
- **Why:** consumers that only need the maths don't pull in a cloud SDK or a web framework.
- **Rejected:** statsmodels as a core dependency. The few formulas we need are short,
  and they are tested against textbook values instead.

### D-011 · Configuration in TOML, pre-registered · accepted
- **Decision:** one TOML file per experiment (primary, guardrails, method, alpha, MDE, split,
  SRM threshold). The REST `/analyze` endpoint takes the same structure as JSON.
- **Why:** `tomllib` is in the Python 3.11 standard library (no PyYAML dependency), configs
  are diffable, and writing the config before the readout *is* the pre-registration.

### D-012 · Python ≥ 3.11 · accepted
- **Why:** this matches the brief, and gives us `tomllib` plus faster CPython.

---

## B. Data

### D-013 · Verify the dataset before extracting · accepted
- **Decision:** `data/verify.sql` lists the tables and columns, the min/max `creation_date` per
  table, and the monthly acceptance-vote counts. It must be run before choosing T0.
- **Why:** the brief warns that the public snapshot may be stale or that its tables may differ.
  T0 must keep `[T0−56d, T0+28d)` inside the *oldest* table's max date.

### D-014 · One SQL extract, cached to parquet · accepted
- **Decision:** a single parameterised query (`@t0 DATE`) produces one row per user.
  `BigQuerySource` caches the result to `data/users_metrics.parquet` and sets
  `maximum_bytes_billed` (50 GB) as a cost guard.
- **Why:** simulations run thousands of times on local data. The warehouse is hit once.

### D-015 · Windows: 56-day pre-period, 28-day post-period · provisional
- **Why:** the pre-period is twice the outcome window, which gives stable covariates. The post
  period is 4 whole weeks, which avoids day-of-week imbalance. Revisit once the verify
  query shows how fresh the data is.

### D-016 · Population definition · accepted
- **Decision:** users whose account was created before the pre-period start **and** who
  posted at least one answer or comment in the pre-period.
- **Why:** every unit then has a full pre-period, so the CUPED covariate is defined for
  everyone. This mirrors "users active before launch".
- **Consequence:** `pre_active` = 1 for all users and carries no information. It is never
  used as a covariate (see D-019).

### D-017 · Leakage rules · accepted
- **Decision:**
  1. Never use snapshot columns (`users.reputation`, `up_votes`, `posts.score`, and so on).
     They include post-period activity.
  2. Acceptance is dated using `votes` with `vote_type_id = 1`. An answer counts as accepted
     in a window only if the acceptance vote is dated before that window ends. Pre-period
     covariates therefore cannot see acceptances that happened during the experiment.
- **Rejected:** using `posts_questions.accepted_answer_id` as-is. That is the
  status *today*, which leaks the future into the covariate.
- **To verify:** that `votes.creation_date` exists at the needed resolution (verify query step 4).

---

## C. Metrics

### D-018 · Metric catalog · accepted
| Metric | Kind | Column(s) | Covariate(s) |
|---|---|---|---|
| answers_posted | mean (count) | post_answers | pre_answers |
| active_28d | binary | post_active | pre_answers, pre_comments |
| comments_posted | mean (count) | post_comments | pre_comments |
| accepted_answer_rate | ratio | post_accepted / post_answers | none (delta method) |

### D-019 · `active_28d` uses pre-period *volume* as covariates · accepted
- **Why:** `pre_active` is constant (D-016) and would give a zero-variance covariate, so
  pre-period answer and comment counts are used instead.

### D-020 · Ratio metrics are ratios of sums · accepted
- **Decision:** the target is Σaccepted / Σanswers per arm, with variance from the delta method
  at the user level. The mean of per-user ratios (`naive_ratio`) is kept **for comparison only**.
- **Why:** the randomisation unit is the user, not the answer, so answers within a user are
  correlated. A binomial variance over answers is too small. The per-user mean weights a 1-answer
  user the same as a 100-answer user, so it estimates a different quantity.

---

## D. Assignment and effect injection

### D-021 · Production assignment: `sha256(salt:user_id) % 10,000` · accepted
- **Why:** it is deterministic and stateless, so any service can recompute a user's arm. The
  per-experiment salt makes assignments independent across experiments. 10,000 buckets allow
  0.01% allocation granularity.
- **Used in:** the SRM simulation and the example readouts.

### D-022 · The harness uses RNG permutations instead of hashing · accepted
- **Decision:** `run_simulation` draws 2n users without replacement from a per-run seeded RNG.
  The first n are control and the rest are treatment.
- **Why:** this is statistically equivalent to hashing with a fresh salt, but about 100× faster
  for 5,000 runs. Hashing is still exercised where its properties matter (SRM, readouts),
  and the hash itself is unit-tested for determinism, balance and salt independence.

### D-023 · Effect injection preserves data types · accepted
| Metric | Lift < 0 | Lift > 0 |
|---|---|---|
| counts | binomial thinning `Bin(y, 1+lift)` | Poisson thickening `y + Pois(lift·y)` |
| continuous | scale `y·(1+lift)` | scale |
| binary | flip 1→0 with p = −lift | flip 0→1 with p = p̄·lift/(1−p̄) |
| ratio | thin the numerator | convert non-accepted answers to accepted, keeping accepted ≤ answers |
- **Why:** counts stay integer, zeros stay zero (users with no activity can't be scaled), and
  the expected value moves by exactly (1+lift). The tail shape is kept, which is the
  point of using real data.
- **Supports:** scalar or per-user lift arrays (heterogeneous and segment effects).
- **Invariant:** covariates are **never** modified, because a treatment cannot change the
  past. Modifying them would bias CUPED.
- **Example readouts:** answers added by injection are accepted at the population acceptance
  rate, which keeps the accepted-answer-rate guardrail honest.

---

## E. Statistical methods

### D-024 · Welch t-test is the default · accepted
- **Why:** it does not assume equal variances, and costs almost nothing when the variances are
  equal. A large-sample z-test is available. Mann-Whitney is included but flagged as testing
  stochastic dominance, not a mean difference (no CI is reported).

### D-025 · CUPED design · accepted
- **Decision:** θ is estimated on the **pooled** sample (both arms), and X is centred on the
  pooled mean. Multiple covariates use OLS (regression adjustment). An optional
  zero-pre-activity indicator (`cuped_zero`) is available.
- **Why:** pooled θ does not depend on treatment, which keeps the estimator unbiased under
  randomisation. Centring keeps adjusted means on the original scale. The zero indicator
  handles users with no pre-activity, which the brief asks for explicitly.
- **Reported:** `variance_reduction` per run, compared against the theoretical R² (= ρ² with one covariate).

### D-026 · Bootstrap: percentile and BCa, chunked · accepted
- **p-value:** 2·min(P(boot ≤ 0), P(boot ≥ 0)), an approximation whose calibration is
  *checked* by the A/A harness rather than assumed.
- **Memory:** resampling is chunked to about 5M indices per chunk (~40 MB).
- **Simulations** use `n_boot = 500`, and the bootstrap runs only at n ≤ 1,000 in A/A
  calibration. That is where it matters (small-sample, skewed data) and where it is affordable.

### D-027 · SRM threshold p < 0.001 · accepted
- **Why:** SRM runs on every experiment. A 0.05 threshold would raise a false alarm in 1 of 20
  clean tests. 0.001 is the industry norm.

### D-028 · Multiple comparisons: Holm for FWER, BH for FDR · accepted
- **Why:** Holm dominates Bonferroni and still controls FWER. BH is used for exploratory
  segment scans.
- **Guardrails are not corrected** (D-033).

### D-029 · Sequential testing: Lan-DeMets boundaries by Monte Carlo · accepted
- **Decision:** two-sided symmetric boundaries for O'Brien-Fleming- and Pocock-type spending
  functions, found by simulating the canonical joint distribution of the z-statistics
  (400k paths, fixed seed, `lru_cache`).
- **Why:** this is exact up to Monte Carlo error (±~0.01 on z), short and easy to verify. It is
  checked against the published OBF value (final boundary ≈ 2.04 for 5 looks) and against an
  empirical type-I error test.
- **Rejected:** numerical recursive integration (Armitage-McPherson-Rowe). It is more precise
  but much more code to get right, and the extra precision does not change any conclusion here.
  Revisit if boundaries are needed to 3 decimal places.

### D-030 · mSPRT with a normal mixture · accepted
- **Decision:** Johari et al. (2017) normal-mixture mSPRT on the mean difference. τ is set
  relative to the population mean (`msprt_tau_rel = 0.05`), and p-values are always-valid
  (monotone min).
- **Why:** it allows continuous monitoring with no fixed number of looks. τ is the knob
  that trades early detection of large effects against power for small ones.

### D-031 · Winsorisation caps on the pooled sample · accepted
- **Why:** capping each arm at its own percentile would cap away part of the treatment effect
  asymmetrically. Capped and uncapped results are always reported side by side (brief §12).

### D-032 · Bayesian beta-binomial via Monte Carlo · accepted
- **Note:** its registry "p_value" is 2·min(P(better), P(worse)), clearly labelled as **not a
  p-value**. It exists so the A/A harness can compare it against frequentist calibration.

---

## F. Decisions engine

### D-033 · Fixed decision order · accepted
1. SRM fail → **INVALID** · 2. guardrail significantly harmed → **DO NOT SHIP** ·
3. primary up → **SHIP** · 4. primary down → **DO NOT SHIP** ·
5. CI inside (−MDE, +MDE) → **DO NOT SHIP** (well-powered null) · 6. else → **EXTEND**.
- **Why:** an SRM failure invalidates every number, and harm outranks benefit. Guardrails are
  not multiplicity-corrected because a missed harm costs more than a false alarm.
- **MDE** in the config is relative and converted to absolute using the observed control mean.

---

## G. Simulation engine

### D-034 · Claims policy: methodology only · accepted
- The data is semi-synthetic: behaviour is real, assignment and effects are simulated. We may
  claim calibration, power, coverage and variance-reduction results. We may **not** claim any
  business lift. Every simulated readout carries a "Simulated data" banner.

### D-035 · Reproducibility through `SeedSequence.spawn` · accepted
- **Decision:** each run gets its own child seed of one root seed.
- **Why:** results are bit-identical serially or in parallel, in any order and on any
  machine. A test checks that parallel output equals serial output.

### D-036 · Parallelism: joblib, optional and batched · accepted
- **Why:** runs are embarrassingly parallel. Batching (4 batches per worker) amortises
  process overhead. Without joblib, everything runs serially.

### D-037 · Degenerate runs are recorded, not fatal · accepted
- **Decision:** a `ValueError` from a method (for example, no answers in an arm at n = 100)
  becomes a row with NaN results and an `error` message. `summarize` reports `failed_runs`.
- **Why:** a 5,000-run job should not die on run 4,212. Hiding failures would bias the
  calibration, so they are counted and shown.

### D-038 · Peeking model = sample accrual · provisional
- **Decision:** users arrive in random order, and look k analyses the first k/K of each arm.
  Statistics at every look come from cumulative sums in O(n).
- **Why:** the extract is per-user 28-day aggregates, not daily events, so accrual is the
  honest model it supports. It is the standard group-sequential setting.
- **Limitation:** it does not capture within-user metric growth over the 28 days. Upgrade path:
  a daily-grain extract (`user_id, day, answers, comments`) and a `looks` mode that uses
  cumulative-to-day outcomes.

### D-039 · Acceptance criteria are numeric and reported honestly · accepted
- A/A: the Wilson 95% CI of the rejection rate contains 0.05, **and** KS uniformity p > 0.01.
  Failures (expected: Welch on heavy tails at n = 100) are reported, not tuned away.

---

## H. Quality

### D-040 · Testing strategy · accepted
- Unit tests against scipy and textbook values (Welch vs `scipy.stats.ttest_ind`, Holm/BH worked
  examples, the n = 1570 sample-size example, the OBF boundary).
- Property-based tests (`hypothesis`): p ∈ [0,1], swap symmetry, shift invariance, SRM silent
  on exact splits, correction ordering BH ≤ Holm ≤ Bonferroni, CUPED with an uncorrelated
  covariate ≈ Welch.
- Statistical tests with fixed seeds and explicit tolerances (A/A FPR, coverage, injected-lift
  recovery, peeking inflation).
- CI runs lint, the test suite, and **every experiment script in `--quick` mode**, so the
  pipeline can't silently rot.

### D-041 · Generated data and results are not committed · accepted
- `data/*.parquet` and `results/` are git-ignored. They are fully reproducible from
  `extract.sql` plus the scripts and seeds.

### D-042 · Charts deferred to phase 2 · provisional
- Phase 1 delivers the engine, CSV/parquet results and a table-based app. Calibration plots,
  power curves and the variance-reduction figure come next, built from `results/*_raw.parquet`.

### D-043 · Lint at line length 120; `Decision` is a `StrEnum` · accepted
- **Why:** statistical formulas read better on one line than wrapped. `StrEnum` (3.11+) makes
  decisions serialise as plain strings in JSON.

---

## I. Decisions made during first verification run (2026-10-16)

### D-044 · Welch df falls back to pooled df on underflow · accepted
- **Found by:** a hypothesis property test (arms `[0, 0]` vs `[0, 1.5e-146]`). The squared
  per-arm variances underflow to 0, so Welch-Satterthwaite divides by zero.
- **Decision:** when the denominator is 0, use `nc + nt − 2`. This shows why property tests
  are part of the testing strategy (D-040).

### D-045 · The guardrail test fixture keeps answer quality constant · accepted
- **Found by:** `test_analyze_ship_on_big_lift` returned DO_NOT_SHIP. Injecting extra answers
  with no accepted answers lowered `accepted_answer_rate`, and the guardrail correctly blocked
  the ship.
- **Decision:** test fixtures accept injected answers at the base rate (as `example_readouts.py`
  does). A dedicated test now asserts that the guardrail **does** block when quality drops.

### D-046 · SRM example uses a 5% loss, not 3% · accepted
- **Finding:** at 20k users per arm, losing 3% of treatment gives χ² ≈ 9 (p ≈ 0.0025), which is
  **not** caught at the 0.001 threshold. A 5% loss gives χ² ≈ 26 (p ≈ 6e-7).
- **Consequence:** the SRM threshold trades false alarms for blindness to small leaks.
  `experiments/srm.py` quantifies the detection curve. Report it in the README as a known
  limit of SRM checks.

### D-047 · SyntheticSource guarantees the requested size · accepted
- **Bug:** ~40% of generated users are pre-active, so oversampling by 2.5× sometimes returned
  fewer rows than requested. It now oversamples by 5× and raises an error if it is still short.

### D-048 · T0 = 2022-08-22, with a guard against truncated windows · accepted
- **Finding:** the first extract used T0 = 2022-09-01, so the post-period ran to 2022-09-29. A
  freshness query showed that answers, comments and acceptance votes all **end on 2022-09-25**,
  which cut the last ~4 of 28 outcome days.
- **Decision:** T0 = 2022-08-22 (a Monday). That gives a pre-period of [2022-06-27, 2022-08-22)
  and a post-period of [2022-08-22, 2022-09-19), with 6 days of margin before the data ends. Both
  windows are whole weeks.
- **Guard:** `expkit extract` refuses any T0 whose post-period runs past `DATA_END` unless
  `--allow-truncated` is passed. It also replaces an existing output file, because
  `BigQuerySource` otherwise returns the cached parquet whatever T0 it was built for.
- **Also:** user id −1 (the "Community" bot) is excluded in `extract.sql`.
- **Note:** acceptance votes are stored at **day** granularity (midnight timestamps). This is
  fine for the window boundaries, which are dates.

### D-049 · Example readouts are replicated and sized from real-data power · accepted
- **Finding:** on the real extract, scenario sizes chosen on synthetic data were badly
  underpowered. Real `answers_posted` is far more heavy-tailed. The CUPED 95% CI half-width is
  about ±0.047 at 20k/arm and ±0.024 at the full population (~80k/arm), against a control mean of
  ~0.46. **80% power needs ~15% lift at 20k/arm and ~7.5% even at the full population.**
  (Corrected 2026-10-17: an earlier version said ~14% at full population. That multiplied
  the power factor 2.8 by the CI half-width instead of the SE. Recomputed from the CUPED residual
  SD of 2.42; it agrees with the winsorization run, where +20% at 20k/arm was detected 92% of the time.)
- **Finding:** publishing a single draw per scenario hid how often the decision rules disagree
  with a scenario's intent. Uncorrected guardrails (D-033) plus heavy tails gave false alarms
  on single draws.
- **Decision:** each scenario runs 200 independent draws, and each scenario gets its own spawned
  seed. The readout publishes the first draw that matches the intended decision and states the
  decision frequencies, so the selection is disclosed, never hidden. The new scenarios:
  clear win = 80k/arm with a +20% lift; well-powered null = 80k/arm with a pre-registered 10% MDE.
- **Result (real data, 200 draws each):** clear win → ship 96%; well-powered null →
  do_not_ship 94%; inconclusive → extend 86% (12% do_not_ship from guardrail false alarms at
  n = 1,500); SRM → invalid 100%.
- **Rejected:** hunting for a seed that "looks right" without disclosure.

## J. Decisions from the first full real-data run (2026-10-17)

### D-050 · Ratio metrics default to a jackknife t-test, not the delta method · accepted
- **Finding (A/A, 5,000 runs per cell):** the delta method over-rejected on `accepted_answer_rate`.
  Its false-positive rate was **20.6% at n=100, 14.0% at 500, 10.4% at 1,000, and 6.1% at
  10,000** per arm. Only ~12% of users answer, and a few heavy answerers hold most of the
  denominator, so the linearised variance is biased low. The user-level bootstrap was also
  anti-conservative (7.4–7.8% at n ≤ 500).
- **Decision:** a new `jackknife_ratio` method. It computes a closed-form, O(n)
  leave-one-user-out variance and uses a Student t reference with df = (fewest users with a
  positive denominator in either arm) − 1. It is now `DEFAULT_METHOD["ratio"]`. `delta` stays
  registered for comparison.
- **Result:** false positives were **5.9% at n=100, 6.9% at 500, 6.4% at 1,000, 5.8% at 2,000 and
  5.2–5.5% at 10,000**. Power at 10k/arm drops slightly (12.9% vs 13.9% for a +5% lift), which
  is the honest price of correct error control.
- **Residual:** about 6–7% between 500 and 1,000 users per arm. Small ratio-metric
  guardrails should therefore be read as "worth a closer look", which matches D-033 anyway.
- **Rejected:** `naive_ratio`. It is calibrated (4.4–5.4%) but estimates a different quantity
  (the mean of per-user ratios), and it was **biased by +18%** relative to the true
  ratio-of-sums effect under a +5% lift.

### D-051 · mSPRT τ is set in units of the final-look SE; default 3 · accepted
- **Finding:** with τ = 5% of the metric mean, τ was ~0.5 SE for `answers_posted`, and mSPRT
  needed z ≈ 6 to reject. Its power was **0%**.
- **Decision:** `PeekingSpec.msprt_tau_se` (default **3.0**), so τ² = 3² × Var(diff at the
  planned n). The `experiments/peeking.py` sweep over {0.5, 1, 2, 3, 5, 10} writes
  `results/msprt_tau_sweep.csv`. Power peaks at 3–5 SE for both metrics, and A/A false
  positives are ≤ 1.5% at every τ.
- **Trade-off, documented:** at the planned horizon, mSPRT has about half the power of a
  fixed-horizon test (answers +20%: 10% vs 33%; active +5%: 21% vs 49%). It buys validity under
  unlimited peeking. **O'Brien-Fleming keeps ~93% of the fixed-horizon power** (30% vs 33%;
  46% vs 49%) and stops early on average. OBF is the recommended monitoring rule when the
  number of looks is planned.

### D-052 · Experiments use effects their sample size can detect · accepted
- **Finding:** the first winsorization run (+3% at 5k/arm) and the first peeking run (+3% on
  answers) gave ~5% power for every method, so they compared nothing.
- **Decision:** winsorization uses an A/A arm plus a +20% lift at 20k/arm. Peeking uses +20% for
  `answers_posted` and +5% for `active_28d` at 10k/arm.
- **Result (winsorization, 2,000 runs):** CUPED reached **92% power on answers with −1.5%
  bias and 95% coverage**. Capping at p99 reached 91% power but **−56% bias and 2% coverage**
  of the true lift. Capping at p95 had −87% bias and 0% coverage. Under a multiplicative effect,
  most of the absolute lift sits in heavy users, which is exactly what capping removes.

### D-053 · The calibration pass/fail is the false-positive rate at α · accepted
- `aa_calibration_summary.csv` now has two columns. `calibrated_at_alpha` checks that the
  Wilson 95% CI of the rejection rate contains 0.05. `calibrated` is the strict check: it also
  requires KS p > 0.01 for Uniform p-values.
- **Why:** with 5,000 runs, KS rejects harmless non-uniformity far from α. Examples are ties
  at p = 1 when both arms are all zeros, and discrete p-values for binary metrics. Welch on
  `answers_posted` at 10k had exactly 4.98% false positives but KS p = 0.004. Decisions depend on
  the rate at α, so that is the gate. KS is still reported. This criterion was changed **after**
  seeing the results; both columns are kept so the change is visible.

### D-054 · Real-data method guidance · accepted
- Welch and CUPED are **conservative at small n** on heavy-tailed counts. For example, Welch
  on comments had 1.9% false positives at n=100 and 3.3% at 1,000. That is safe but costs
  power. Both are calibrated at 10k.
- The percentile bootstrap is **anti-conservative** below 10k on heavy tails (8.5–9.2% at
  n=100). It is not recommended for heavy-tailed metrics with small samples.
- Mann-Whitney is calibrated everywhere, but it tests a shift in distribution, not in the mean.
  It has no CI for the lift, so it is not used for ship decisions.
- `docs/decision_framework.md` § "Choosing the method" carries this guidance.

### D-055 · Synthetic runs write to `results/synthetic/` · accepted
- **Bug:** `make experiments-quick` wrote synthetic summaries and example readouts over the
  real `results/` and `docs/example_readouts/`. It happened once during this session and was
  caught only because every file carries a `data_source` label.
- **Decision:** `experiments/_common.results_dir(source)` sends anything that is not real data
  to `results/synthetic/`, including the example readouts. Verified: a quick run leaves the real
  files byte-identical.

### D-056 · Every integration surface is tested, in CI too · accepted
- The REST API and Streamlit app had never been run. They now have tests:
  `tests/test_server.py` covers every endpoint, the 422 error paths and strict-JSON output
  when a result has NaN (Mann-Whitney has no CI). `tests/test_app.py` uses Streamlit's headless
  `AppTest` and clicks both simulation buttons.
- CI previously installed only `[dev]`, so those tests would have been **skipped**, not run. It
  now installs `[dev,api,app]` and lints `app/` too. `httpx` was added to `[dev]` for the
  FastAPI test client.
- Verified by hand: a live `uvicorn` server analysed a 6,000-user real-data sample over HTTP
  in 0.07 s. The Docker image builds (1.03 GB). The API container answers `/health` and `/power`,
  and the Streamlit container passes `/_stcore/health`.
- **Bug found:** the app always simulated on synthetic data. It now uses the real extract when
  present and states which data it uses.

---

## Open questions (to resolve with real data)
1. ~~**T0**~~ resolved: 2022-08-22 (D-048).
2. ~~Do `votes` acceptance timestamps exist across the window?~~ Resolved: yes, through
   2022-09-25, at day granularity (D-048). The dated-acceptance leakage rule (D-017) stands.
3. ~~Should heavy-tail capping be the default for `answers_posted`?~~ Resolved: **no**. CUPED
   gives the same power with no bias, while capping destroys the coverage of the true lift (D-052).
4. ~~**mSPRT τ tuning**~~ Resolved: τ = 3 × final-look SE (D-051).
5. Stretch goals after the acceptance tests pass: CUPAC (ML covariate), a daily-grain extract
   for peeking (D-038), and interference checks.
