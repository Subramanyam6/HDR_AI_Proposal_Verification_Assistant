PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip

.PHONY: setup build build-no-noise clean rebuild

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

build:
	$(PYTHON) scripts/06_build_dataset.py

build-no-noise:
	$(PYTHON) scripts/06_build_dataset.py --skip-noise

clean:
	rm -rf dataset/train dataset/dev dataset/test
	rm -f dataset/synthetic_proposals.json dataset/*.log

rebuild: clean build
