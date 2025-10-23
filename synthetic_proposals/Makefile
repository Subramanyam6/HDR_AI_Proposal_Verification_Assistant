PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup build build-no-noise clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

build:
	$(PY) scripts/06_build_dataset.py

build-no-noise:
	$(PY) scripts/06_build_dataset.py --skip-noise

clean:
	rm -rf synthetic_proposals/dataset/train synthetic_proposals/dataset/dev synthetic_proposals/dataset/test
	rm -f synthetic_proposals/dataset/*.log
