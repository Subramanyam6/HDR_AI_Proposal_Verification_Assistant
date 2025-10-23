# Synthetic Proposal Dataset Pipeline

This repository bootstraps a fully local pipeline for generating ~250 synthetic HDR-style proposal records as JSON, paired with YAML manifests, labels, text chunks, and metadata files. The pipeline is deterministic (seeded) and requires no paid APIs or headless browsers.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make build
```

The `make build` target runs the orchestrator (`06_build_dataset.py`), which executes every stage (YAML generation → chunking → labeling → JSON export) and prints a concise dataset summary at the end.

To reset the dataset, run:

```bash
make clean
```

## Repository Structure

```
synthetic_proposals/
  config/knobs.yaml        # Global knobs: counts, mistake rates, noise probabilities
  data/
    seeds/                 # Hand-authored seed YAML exemplars (per sector)
    dictionaries/          # Agencies, roles, tasks, compliance vocab
  templates/               # Legacy HTML/CSS assets (retained for reference)
  scripts/                 # Stage scripts (01–06) and shared utils
  dataset/                 # Output root (train/dev/test directories populated after build)
```

Each proposal directory inside `dataset/<split>/proposal_XXXX/` includes:

- `proposal.yaml` – canonical manifest following the base template schema
- `proposal.json` – JSON counterpart used directly for modeling or ingestion
- `labels.json` – ground-truth flags for injected issues
- `chunks.json` – normalized section text chunks for embeddings
- `meta.json` – provenance, styling, mistakes, pagination metrics
- `meta_history.jsonl` – append-only log of meta updates across stages

## Pipeline Stages

| Stage | Script | Description |
| --- | --- | --- |
| 01 | `01_generate_yaml.py` | Synthesizes YAML manifests from seeds, Faker data, and dictionaries while planning mistake injections and style choices. |
| 02 | `02_render_pdfs.py` | Legacy stage retained for compatibility; it now emits a notice because PDF rendering is not required. |
| 03 | `03_inject_noise.py` | Legacy stage retained for compatibility; it now emits a notice because PDF noise is not applied. |
| 04 | `04_make_chunks.py` | Generates normalized text chunks from YAML sections. |
| 05 | `05_labeler.py` | Produces `labels.json` with structural/compliance/formatting flags. |
| 06 | `06_build_dataset.py` | End-to-end orchestrator that runs stages 01–05 and writes an aggregated `synthetic_proposals.json`. |

All stage functions are also exposed via `scripts/pipeline.py` for programmatic use.

## Configuration

Tune generation parameters in `config/knobs.yaml`:

- `seed` – global random seed for reproducibility
- `counts.total` & `counts.split` – dataset size and split ratios
- `mistakes` – probability targets for each injected issue (minimum 20 occurrences enforced)
- `content_variation` – fonts, tone, margins, addenda range, etc.
- `noise` – retained for compatibility; currently only captured as metadata flags.

Adjust dictionaries in `data/dictionaries/` to extend agency lists, typical tasks, or compliance vocabulary. Add new seed YAMLs under `data/seeds/` to bias towards additional sectors or proposal archetypes.

## Output Summary

`06_build_dataset.py` prints two quick summaries at the end of a build:

1. Dataset tree counts per split (YAML/JSON totals)
2. Mistake coverage counts (ensures ≥20 per mistake type)

In addition, the run produces `dataset/synthetic_proposals.json`, a single file containing every proposal’s narrative content, metadata, labels, and text chunks for easy ingestion downstream.

## Determinism & Reproducibility

- All stochastic operations derive from the seed in `config/knobs.yaml`.
- Every proposal’s `meta_history.jsonl` captures stage-by-stage updates to support auditing and deterministic reruns.

## Custom Runs

Run stages individually if you are iterating on a specific component. For example:

```bash
source .venv/bin/activate
python synthetic_proposals/scripts/01_generate_yaml.py --config synthetic_proposals/config/knobs.yaml
python synthetic_proposals/scripts/04_make_chunks.py
python synthetic_proposals/scripts/05_labeler.py
python synthetic_proposals/scripts/06_build_dataset.py
```

You can point to an alternate dataset root via `--dataset-root` to maintain multiple experiment outputs side-by-side.
