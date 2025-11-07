# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HDR AI Proposal Verification Assistant is a complete ML pipeline that:
1. **Generates** 10,000 synthetic HDR-style proposals with intentional compliance errors
2. **Trains** multiple models (Naive Bayes, TF-IDF+LogReg, DistilBERT) for multi-label classification
3. **Deploys** a Gradio web UI on Hugging Face Spaces for real-time verification

The system detects 4 types of compliance issues: crosswalk errors, banned phrases, name inconsistencies, and date inconsistencies.

## Architecture: Three-Stage Pipeline

### Stage 1: Data Generation (`synthetic_proposals/`)
- **Entry point**: `scripts/06_build_dataset.py` (orchestrator)
- **Core logic**: `scripts/pipeline.py` (~1,071 LOC)
- **Text processing**: `scripts/util_text.py` (chunking and normalization)

**Key flow**:
```
config/knobs.yaml → pipeline.generate_yaml_batch() → proposal YAMLs
                  → pipeline.make_chunks() → chunks.json
                  → pipeline.label_dataset() → labels.json
                  → pipeline.build_dataset() → synthetic_proposals.json
```

**Critical detail**: Text goes through `gather_section_text()` → `chunk_text(max_words=180, overlap=25)` → concatenation. This preprocessing creates the distribution models are trained on.

### Stage 2: Model Training
Three models trained on `dataset/synthetic_proposals.json`:

1. **Baseline** (`07_baseline_ml.py`): Naive Bayes, no external dependencies
2. **TF-IDF** (`08_tfidf_logreg.py`): TF-IDF vectorizer + Logistic Regression (best performer)
3. **Transformer** (`09_transformer_mps_verified.py`): DistilBERT with MPS GPU acceleration

**Training data format**: Uses `chunks_text()` function (line 79-81 of transformer script) which concatenates all proposal chunks with single spaces. This is NOT raw text.

### Stage 3: Deployment (`huggingface_space/`)
- **Entry point**: `app.py` (Gradio UI)
- **Models loaded**: `model/vectorizer.pkl`, `model/classifier.pkl`, `model/distilbert/`
- **Inference**: `verify_proposal()` runs both models in parallel, aggregates with OR logic

## Hugging Face Spaces Deployment

### Repository Structure (CRITICAL)

**The git repository root contains TWO deployment targets:**
1. **HF Spaces deployment** - reads from ROOT directory
2. **Local development** - uses `huggingface_space/` subdirectory

```
HDR_AI_Proposal_Verification_Assistant/  (git root)
├── README.md                    ← HF Spaces reads THIS (must have YAML)
├── requirements.txt             ← HF Spaces reads THIS
├── .gitignore
└── huggingface_space/
    ├── app.py                   ← Main application
    ├── assets/
    │   └── styles.css
    ├── scratch_*.txt            ← Sample files (6 files)
    └── model/                   ← NOT in git (upload manually)
        ├── classifier.pkl
        ├── vectorizer.pkl
        ├── config.json
        └── distilbert/
            └── model.safetensors (255MB)
```

### HF Spaces Configuration (README.md)

**Root README.md MUST have YAML frontmatter:**
```yaml
---
title: HDR Proposal Verification Assistant
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: huggingface_space/app.py  ← CRITICAL: points to subdirectory
pinned: false
---
```

**Common mistakes:**
- ❌ Editing `huggingface_space/README.md` (HF ignores this)
- ❌ Missing YAML frontmatter in root README
- ❌ Wrong `app_file` path
- ❌ Version mismatch between README and requirements.txt

### Git Remotes

Two remotes configured:
```bash
huggingface   https://huggingface.co/spaces/Subramanyam6/HDR_AI_Proposal_Vefification_Assistant.git
origin        https://github.com/Subramanyam6/HDR_AI_Proposal_Verification_Assistant.git
```

**Push to HF Spaces:**
```bash
git push huggingface main
```

**Push to GitHub:**
```bash
git push origin main
```

**Note:** Remotes are separate - pushing to one does NOT update the other.

### Model Files (Manual Upload Required)

**These files are TOO LARGE for git and must be uploaded via HF web interface:**

Create `model/` directory in HF Spaces and upload:

**TF-IDF Model (required):**
- `model/classifier.pkl` (314 KB)
- `model/vectorizer.pkl` (405 KB)
- `model/config.json` (373 bytes)

**DistilBERT Model (required):**
- `model/distilbert/model.safetensors` (255 MB) ← LARGE FILE
- `model/distilbert/config.json` (754 bytes)
- `model/distilbert/tokenizer.json` (695 KB)
- `model/distilbert/tokenizer_config.json` (1.2 KB)
- `model/distilbert/special_tokens_map.json` (125 bytes)
- `model/distilbert/vocab.txt` (226 KB)

**Upload via HF web interface:** Files tab → Create folder → Upload files

### Sample Files Configuration

Sample text files are in `huggingface_space/`:
- `scratch_clean.txt` - Clean proposal (no errors)
- `scratch_crosswalk.txt` - Crosswalk error example
- `scratch_banned.txt` - Banned phrases example
- `scratch_name.txt` - Name inconsistency example
- `scratch_date.txt` - Date inconsistency example
- `scratch_sample.txt` - Additional sample

**Path configuration in app.py:**
```python
SYNTHETIC_DIR = BASE_DIR  # Points to huggingface_space/ directory
```

**Common issue:** If samples don't appear in dropdown, check that `SYNTHETIC_DIR` points to the correct location where `scratch_*.txt` files exist.

### Clean Repository (Post-Cleanup)

**Files on HF Spaces (essential only):**
- Root: `README.md`, `requirements.txt`, `.gitignore`
- `huggingface_space/app.py`
- `huggingface_space/assets/styles.css`
- `huggingface_space/scratch_*.txt` (6 sample files)

**Removed from HF Spaces (6000+ lines):**
- ❌ Entire `synthetic_proposals/` directory (training code, not needed for deployment)
- ❌ Documentation files (GPU_VERIFICATION_REPORT, IMPLEMENTATION_SUMMARY, etc.)
- ❌ Duplicate README/requirements in `huggingface_space/`
- ❌ Test files and deployment docs

**Rationale:** HF Spaces only needs to RUN the app, not train models or generate data.

## Common Commands

### Environment Setup
```bash
# Use existing venv at project root
source .venv/bin/activate
```

### Dataset Generation
```bash
cd synthetic_proposals
make build                    # Generate full dataset (10,000 proposals)
make clean                    # Remove generated data
make rebuild                  # Clean + build

# Manual invocation
python scripts/06_build_dataset.py
python scripts/06_build_dataset.py --skip-noise
```

### Model Training
```bash
cd synthetic_proposals

# Train all models in sequence
python scripts/07_baseline_ml.py                                    # Naive Bayes
python scripts/08_tfidf_logreg.py                                   # TF-IDF (best)
python scripts/09_transformer_mps_verified.py --max-epochs 20       # DistilBERT

# Transformer with custom hyperparameters
python scripts/09_transformer_mps_verified.py \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --max-epochs 30 \
  --early-stopping-patience 7
```

### Testing
```bash
# Unit tests for pipeline
cd synthetic_proposals
pytest tests/test_dataset_pipeline.py

# UI endpoint testing (uses trained models)
python test_samples_standalone.py     # Standalone test (no Gradio dependency)
python test_ui_endpoints.py           # Full UI test (requires gradio)
```

### Running UI Locally
```bash
cd huggingface_space
python app.py
# Opens at http://localhost:7860
```

## Critical Architecture Details

### Text Preprocessing Pipeline (MUST UNDERSTAND)

**Training expects chunked, normalized text**:
```python
# From util_text.py
chunks = chunk_text(text, max_words=180, overlap=25)  # Split into chunks
training_text = " ".join(c["text"] for c in chunks)   # Concatenate with spaces
```

**UI currently takes raw text** → Distribution shift issue documented in `TRAIN_TEST_MISMATCH_ANALYSIS.md`

**For demo samples**: Use files in `synthetic_proposals/*_sample.txt` (already preprocessed correctly as of latest update). Old samples backed up in `synthetic_proposals/old_samples_backup/`.

### Label Extraction Logic

Labels are derived from `labels.json` in each proposal:
```python
# From 09_transformer_mps_verified.py:40-44
TARGETS = (
    ("crosswalk_error", lambda y: int(len(y.get("crosswalk_errors", [])) > 0)),
    ("banned_phrases", lambda y: int(len(y.get("banned_phrases_found", [])) > 0)),
    ("name_inconsistency", lambda y: int(not y.get("name_consistency_flag", True))),
    ("date_inconsistency", lambda y: int(not y.get("date_consistency_flag", True))),
)
```

### Model Aggregation (UI)

```python
# From app.py:161-193
# Combines predictions with OR logic
is_issue = transformer_info["is_issue"] or tfidf_info["is_issue"]
score = max(transformer_info["confidence"], tfidf_info["confidence"]) / 100
```

Thresholds (app.py:30):
- Transformer: 0.50
- TF-IDF: 0.00 (any positive decision score triggers)

### File Structure for Generated Proposals

Each proposal in `dataset/{train,dev,test}/proposal_XXXX/`:
```
proposal.yaml         # Original YAML structure
proposal.json         # JSON representation
labels.json           # Ground truth labels
chunks.json           # Text chunks for embeddings
meta.json             # Metadata (sector, client, etc.)
meta_history.jsonl    # Append-only audit log
```

## Configuration

### Dataset Generation (`config/knobs.yaml`)
- `seed`: 20240510 (deterministic RNG)
- `counts.total`: 10000 (adjustable)
- `counts.split`: {train: 0.72, dev: 0.14, test: 0.14}
- `mistakes`: 15 error types with injection probabilities (0.10-0.22)

### Model Hyperparameters
Embedded in training scripts (no external config files):
- TF-IDF: 10,000 features, OneVsRest Logistic Regression
- Transformer: DistilBERT, batch_size=4, grad_accum=8, lr=5e-5

## Known Issues and Workarounds

### Issue 1: Train-Test Distribution Mismatch (RESOLVED)
- **Problem**: UI samples didn't match training format
- **Solution**: Corrected samples now in `synthetic_proposals/*_sample.txt`
- **Details**: See `TRAIN_TEST_MISMATCH_ANALYSIS.md`

### Issue 2: Transformer Model Incomplete Training
- **Problem**: Name/date inconsistency labels show 0.0 probability
- **Workaround**: Use only crosswalk_error and banned_phrases samples for demos
- **Details**: See `TEST_RESULTS_ANALYSIS.md`
- **Fix**: Retrain with `python scripts/09_transformer_mps_verified.py --max-epochs 30`

### Issue 3: TF-IDF False Positives on Crosswalk
- **Problem**: TF-IDF threshold of 0.00 flags many clean samples
- **Fix**: Change `MODEL_THRESHOLDS` in `app.py:30` to `{"transformer": 0.50, "tfidf": 0.50}`

## Data Flow Summary

```
knobs.yaml
    ↓
generate_yaml_batch() → 10,000 proposal YAMLs with mistakes
    ↓
make_chunks() → chunks.json (180-word chunks, 25-word overlap)
    ↓
label_dataset() → labels.json (ground truth from mistake injection)
    ↓
build_dataset() → synthetic_proposals.json (aggregated)
    ↓
train models → vectorizer.pkl, classifier.pkl, distilbert/
    ↓
app.py → Gradio UI (loads models, runs inference)
```

## Important Files

### Core Pipeline
- `synthetic_proposals/scripts/pipeline.py` - Main orchestration (1,071 LOC)
- `synthetic_proposals/scripts/common.py` - Shared utilities (RNG, config loading)
- `synthetic_proposals/scripts/util_text.py` - Text chunking and normalization

### Model Training
- `synthetic_proposals/scripts/09_transformer_mps_verified.py` - DistilBERT (563 LOC)
- `synthetic_proposals/scripts/08_tfidf_logreg.py` - TF-IDF model (193 LOC)

### Deployment
- `huggingface_space/app.py` - Gradio UI (671 LOC)
- `huggingface_space/model/` - Serialized models

### Documentation
- `TRAIN_TEST_MISMATCH_ANALYSIS.md` - Distribution shift investigation
- `TEST_RESULTS_ANALYSIS.md` - Model performance on corrected samples
- `README.md` - Project overview

## Debugging Tips

### Verify Dataset Integrity
```python
# Check a single proposal
import json
from pathlib import Path

data_path = Path("synthetic_proposals/dataset/synthetic_proposals.json")
with open(data_path) as f:
    data = json.load(f)

# Inspect first proposal
proposal = data[0]
print("ID:", proposal["id"])
print("Split:", proposal["split"])
print("Chunks:", len(proposal["chunks"]))
print("Labels:", proposal["labels"])
```

### Test Model Predictions
```bash
# Use standalone test (no UI dependencies)
python test_samples_standalone.py

# Should show predictions for all 5 samples
```

### Check GPU Usage (Apple Silicon)
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# In training script, check for:
# "Device Verification" output showing MPS memory allocation
```

## Label-Specific Notes

### Crosswalk Errors
- Detects requirement section mismatches (e.g., R1 expected in Schedule but cited in Letter)
- Stored in `labels.crosswalk_errors` as list of dicts

### Banned Phrases
- Checks for prohibited language ("guarantee", "unconditional", etc.)
- Stored in `labels.banned_phrases_found` as list of dicts
- TF-IDF model shows excellent performance (95% confidence)

### Name Inconsistency
- Detects PM name variations (e.g., "John Smith" vs "John S.")
- Flag: `name_consistency_flag = False`
- **Warning**: Transformer model shows 0.0 probability (needs retraining)

### Date Inconsistency
- Detects mismatched dates in letter vs schedule
- Flag: `date_consistency_flag = False`
- **Warning**: Transformer model shows ~0.0 probability (needs retraining)
