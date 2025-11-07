---
title: HDR Proposal Verification Assistant
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: huggingface_space/app.py
pinned: false
---

# HDR AI Proposal Verification Assistant

An AI-powered system for automated verification of HDR proposal compliance, detecting issues like page limit violations, banned phrases, inconsistencies, and structural errors.

## Overview

This project generates synthetic HDR-style proposals and trains machine learning models to identify common compliance issues:
- Crosswalk errors
- Banned phrases
- Name inconsistencies

The UI also runs a **rule-based date inconsistency check** that compares the anticipated submission date sentence against the signed/ sealed sentence.

## Project Structure

```
HDR_AI_Proposal_Verification_Assistant/
├── synthetic_proposals/
│   ├── config/           # Configuration (knobs.yaml)
│   ├── data/             # Seed data and dictionaries
│   ├── scripts/          # Pipeline and ML scripts
│   ├── templates/        # HTML/CSS templates
│   └── dataset/          # Generated data (gitignored)
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
cd synthetic_proposals
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
make build
```

This generates 10,000 synthetic proposals (configurable in `config/knobs.yaml`).

### 3. Train Models

```bash
# Naive Bayes baseline
python scripts/07_baseline_ml.py

# TF-IDF + Logistic Regression (best performer)
python scripts/08_tfidf_logreg.py

# DistilBERT transformer (GPU accelerated)
python scripts/09_transformer_mps_verified.py --max-epochs 10
```

## Model Performance

Based on 7,200 training samples, 1,400 dev samples:

| Model | Micro-F1 | Macro-F1 | Notes |
|-------|----------|----------|-------|
| **TF-IDF + LogReg** | **0.410** | **0.414** | Best overall, fast inference |
| Naive Bayes | 0.482 | 0.464 | No dependencies, baseline |
| DistilBERT | 0.324 | 0.225 | GPU-accelerated, needs tuning |

## Configuration

Edit `synthetic_proposals/config/knobs.yaml` to adjust:
- Dataset size (`counts.total`)
- Split ratios (train/dev/test)
- Mistake injection probabilities
- Content variation parameters

## Key Features

- **Deterministic generation:** Seeded for reproducibility
- **Multi-label classification:** Detects multiple issues per proposal
- **GPU acceleration:** MPS support for Apple Silicon
- **Extensible:** Easy to add new mistake types or sectors

## Requirements

- Python 3.8+
- See `synthetic_proposals/requirements.txt` for dependencies

## Documentation

- `IMPLEMENTATION_SUMMARY.md` - Detailed implementation notes
- `GPU_VERIFICATION_REPORT.md` - GPU acceleration verification
- `synthetic_proposals/README.md` - Dataset pipeline details
- `synthetic_proposals/TRANSFORMER_GUIDE.md` - Transformer training guide

## Future Work

- [ ] Deploy to Hugging Face Spaces
- [ ] Add PDF text extraction
- [ ] Improve transformer hyperparameters
- [ ] Add more verification rules
- [ ] Create web interface (Gradio)

## License

MIT License

## Author

Subramanyam
