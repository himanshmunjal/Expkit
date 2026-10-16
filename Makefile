.PHONY: install test test-fast lint verify extract profile experiments experiments-quick readouts figures app api docker

# Use the project venv when it exists, so `make` works without activating it.
VENV_BIN := $(wildcard .venv/bin/)
PY ?= $(if $(VENV_BIN),$(CURDIR)/.venv/bin/python,python)
PROJECT ?=
# Monday; post-period [08-22, 09-19) ends before the data does (2022-09-25), DECISION D-048
T0 ?= 2022-08-22
JOBS ?= -1

install:            ## editable install with every extra
	$(PY) -m pip install -e ".[all,dev]"
	pre-commit install || true

lint:
	$(PY) -m ruff check src tests experiments app data

test:               ## full suite
	$(PY) -m pytest --cov=expkit --cov-report=term-missing

test-fast:
	$(PY) -m pytest -x -q -k "not peeking and not parallel"

verify:             ## step 1: inspect tables & freshness on BigQuery
	bq query --use_legacy_sql=false --maximum_bytes_billed=20000000000 < data/verify.sql

extract:            ## step 2: per-user parquet (cached)
	$(PY) -m expkit.cli extract --t0 $(T0) --out data/users_metrics.parquet $(if $(PROJECT),--project $(PROJECT))

profile:            ## step 3: distribution facts
	$(PY) data/prepare.py --data data/users_metrics.parquet --out results/profile.json

experiments:        ## step 4: full validation suite on the real extract
	cd experiments && for s in aa_calibration injected_lift cuped ratio_metrics peeking srm multiple_testing winsorization; do \
		$(PY) $$s.py --jobs $(JOBS) || exit 1; done

experiments-quick:  ## smoke run of every experiment (synthetic data, tiny)
	cd experiments && for s in aa_calibration injected_lift cuped ratio_metrics peeking srm multiple_testing winsorization example_readouts; do \
		$(PY) $$s.py --quick --data synthetic || exit 1; done

readouts:           ## step 5: the four example readouts
	cd experiments && $(PY) example_readouts.py

figures:            ## step 6: charts from results/ (real data only) into docs/figures
	$(PY) -m expkit.viz --results results --out docs/figures

app:
	$(PY) -m streamlit run app/streamlit_app.py

api:
	$(PY) -m uvicorn expkit.server:app --port 8000

docker:
	docker build -t expkit .
