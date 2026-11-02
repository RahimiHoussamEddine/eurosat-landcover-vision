.PHONY: install install-dev fetch build verify quality test app api

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export XDG_CACHE_HOME := $(CURDIR)/.cache
export PIP_CACHE_DIR := $(CURDIR)/.cache/pip
export PIP_NO_CACHE_DIR := 1
export TMPDIR := $(CURDIR)/.cache/tmp

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install setuptools==75.8.0 wheel==0.45.1
	$(PYTHON) -m pip install --no-build-isolation -e .

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --no-build-isolation -e .

fetch:
	$(PYTHON) scripts/fetch_data.py

build:
	$(PYTHON) scripts/build_artifact.py

verify:
	$(PYTHON) -m eurosat_landcover_vision.verify

quality:
	$(PYTHON) -m ruff check src app api scripts tests
	$(PYTHON) -m ruff format --check src app api scripts tests

test:
	$(PYTHON) -m pytest

app:
	$(PYTHON) -m streamlit run app/main.py

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000
