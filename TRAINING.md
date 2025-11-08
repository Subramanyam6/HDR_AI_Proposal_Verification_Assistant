# Model Training Guide

This document explains how to generate synthetic data and train the ML models for the HDR Proposal Verification Assistant.

## Overview

The training pipeline consists of three stages:
1. **Data Generation**: Create 10,000 synthetic HDR-style proposals with compliance errors
2. **Model Training**: Train TF-IDF, DistilBERT, and Naive Bayes models
3. **Model Export**: Copy trained models to `backend/app/models/` for deployment

## Prerequisites

```bash
# Navigate to training directory
cd synthetic_proposals

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Stage 1: Generate Synthetic Data

Generate 10,000 synthetic proposals with intentional compliance errors:

```bash
cd synthetic_proposals
make build
```

This creates:
- `dataset/synthetic_proposals.json` (main dataset)
- `dataset/train/` (7,200 proposals, 72%)
- `dataset/dev/` (1,400 proposals, 14%)
- `dataset/test/` (1,400 proposals, 14%)

**Configuration**: Edit `config/knobs.yaml` to adjust dataset size, error injection probabilities, etc.

## Stage 2: Train Models

### 1. Naive Bayes Baseline

```bash
python scripts/07_baseline_ml.py
```

**Output**: Saves models to `dataset/model/nb_baseline/*.pkl`

**Performance**: Fast training, no dependencies, baseline metrics.

### 2. TF-IDF + Logistic Regression (Best Performer)

```bash
python scripts/08_tfidf_logreg.py
```

**Output**: Saves to `dataset/model/`:
- `vectorizer.pkl` (405 KB)
- `classifier.pkl` (314 KB)
- `config.json` (373 bytes)

**Performance**: Micro-F1: 0.410, Macro-F1: 0.414

### 3. DistilBERT Transformer

```bash
python scripts/09_transformer_mps_verified.py --max-epochs 20
```

**Output**: Saves to `dataset/model/distilbert/`:
- `model.safetensors` (255 MB)
- `config.json`
- `tokenizer.json`
- Other tokenizer files

**Performance**: Micro-F1: 0.324, GPU-accelerated (MPS on Apple Silicon).

**Training options**:
```bash
python scripts/09_transformer_mps_verified.py \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --max-epochs 30 \
  --early-stopping-patience 7
```

## Stage 3: Export Models for Deployment

After training, copy model files to the backend:

```bash
# From project root
cp -r synthetic_proposals/dataset/model/* backend/app/models/
```

**Verify structure**:
```
backend/app/models/
├── vectorizer.pkl
├── classifier.pkl
├── config.json
├── distilbert/
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
└── nb_baseline/ (optional)
    ├── crosswalk_error.pkl
    ├── banned_phrases.pkl
    └── name_inconsistency.pkl
```

## Data Flow Summary

```
config/knobs.yaml
    ↓
06_build_dataset.py → 10,000 synthetic proposals
    ↓
07_baseline_ml.py → Naive Bayes models
08_tfidf_logreg.py → TF-IDF + LogReg models
09_transformer_mps_verified.py → DistilBERT model
    ↓
Copy to backend/app/models/ → Ready for deployment
```

## Model Performance Comparison

| Model | Micro-F1 | Macro-F1 | Training Time | Inference Speed |
|-------|----------|----------|---------------|-----------------|
| **TF-IDF + LogReg** | **0.410** | **0.414** | ~2 min | Very Fast |
| Naive Bayes | 0.482 | 0.464 | ~1 min | Very Fast |
| DistilBERT | 0.324 | 0.225 | ~30 min (GPU) | Moderate |

## Troubleshooting

### Issue: Dataset generation fails
- **Solution**: Check `config/knobs.yaml` for valid parameters
- Ensure seed data exists in `data/sectors/`, `data/names/`, etc.

### Issue: Transformer training fails (GPU)
- **Solution**: For Apple Silicon, ensure PyTorch with MPS support:
  ```bash
  pip install torch==2.5.0
  ```
- For other systems, modify device selection in script

### Issue: Models not loading in backend
- **Solution**: Verify file paths and permissions
- Check `backend/app/models/README.md` for required files

## Advanced Configuration

### Adjust Error Injection Rates

Edit `config/knobs.yaml`:
```yaml
mistakes:
  crosswalk_wrong_section_r1: 0.15  # 15% chance
  banned_phrase_guaranteed_savings: 0.10  # 10% chance
  name_inconsistency_pm_first_name_only: 0.12  # 12% chance
```

### Custom Dataset Size

```yaml
counts:
  total: 5000  # Generate 5,000 proposals instead of 10,000
  split:
    train: 0.70
    dev: 0.15
    test: 0.15
```

## Testing Models

After training, test with standalone script:

```bash
python test_samples_standalone.py
```

This loads all models and runs inference on 5 sample proposals.

## Further Reading

- `IMPLEMENTATION_SUMMARY.md` - Detailed implementation notes
- `GPU_VERIFICATION_REPORT.md` - GPU acceleration verification
- `synthetic_proposals/README.md` - Dataset pipeline details
- `synthetic_proposals/TRANSFORMER_GUIDE.md` - Transformer training guide
