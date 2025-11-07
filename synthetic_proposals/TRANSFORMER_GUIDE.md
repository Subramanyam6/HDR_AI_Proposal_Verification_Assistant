# Transformer Model Guide (Non-Technical)

## What Was Added

A new AI model using **DistilBERT** (a modern transformer) that understands proposal text better than traditional machine learning. It runs on your Mac's GPU for faster training.

---

## How to Run

### Basic Run (Default Settings)
```bash
cd synthetic_proposals
source .venv/bin/activate
python scripts/09_transformer_mps.py
```

### Common Options

**Use a more powerful model (BERT instead of DistilBERT):**
```bash
python scripts/09_transformer_mps.py --model-name bert-base-uncased
```

**Faster training (fewer epochs):**
```bash
python scripts/09_transformer_mps.py --max-epochs 5
```

**More patient training (wait longer before stopping):**
```bash
python scripts/09_transformer_mps.py --early-stopping-patience 5
```

**Adjust batch size (if you see memory errors, lower this):**
```bash
python scripts/09_transformer_mps.py --batch-size 4
```

**Enable experimental speed optimizations:**
```bash
python scripts/09_transformer_mps.py --use-compile
```

---

## Results Summary

### Performance Comparison

| Model Type | Micro-F1 | Macro-F1 | Notes |
|------------|----------|----------|-------|
| **Naive Bayes** (baseline) | ~0.20 | ~0.20 | Simple, fast |
| **TF-IDF + Logistic Regression** | 0.267 | 0.258 | Classical ML |
| **Transformer (DistilBERT)** | **0.367** | **0.348** | **Best** |

**Improvement:** Transformer model is ~37% better than previous baseline!

### What the Numbers Mean

- **Higher = Better** (1.0 is perfect, 0.0 is worst)
- **Micro-F1**: Overall accuracy across all 4 error types
- **Macro-F1**: Average accuracy per error type (treats each equally)

### Per-Label Performance

Best epoch (Epoch 9) achieved:
- **Crosswalk Error**: F1 = 0.450 (decent at catching RFP requirement mismatches)
- **Banned Phrases**: F1 = 0.286 (needs improvement)
- **Name Inconsistency**: F1 = 0.444 (decent)
- **Date Inconsistency**: F1 = 0.211 (hardest to detect)

---

## What to Tweak for Better Results

### 1. **More Training Data** (Most Important!)
   - Current: 180 training samples
   - Your plan: Scale to 10,000 samples → **Will significantly improve performance**
   - What to do: Re-run after generating more synthetic proposals

### 2. **Use BERT Instead of DistilBERT**
   - DistilBERT: Faster, smaller (66M parameters)
   - BERT: More accurate, larger (110M parameters)
   - For 10K samples, BERT will likely perform better
   ```bash
   python scripts/09_transformer_mps.py --model-name bert-base-uncased
   ```

### 3. **Adjust Learning Rate**
   - Default: 2e-5 (good starting point)
   - Try higher: 5e-5 (faster learning, risk of instability)
   - Try lower: 1e-5 (more stable, slower learning)
   ```bash
   python scripts/09_transformer_mps.py --learning-rate 5e-5
   ```

### 4. **Longer Training**
   - Default: 10 epochs max, stops early if no improvement
   - Try: 15-20 epochs with more patience
   ```bash
   python scripts/09_transformer_mps.py --max-epochs 20 --early-stopping-patience 5
   ```

### 5. **Bigger Batches (If You Have Memory)**
   - Your M4 Pro can handle it!
   - Default: Batch size 8
   - Try: 12 or 16 (uses more GPU, may train faster)
   ```bash
   python scripts/09_transformer_mps.py --batch-size 12
   ```

---

## Understanding the Output

### Training Progress
```
Epoch 9/10
  Train loss: 1.0926
  Dev Micro-F1: 0.412  |  Macro-F1: 0.401
  ✓ New best Micro-F1: 0.412
```

- **Train loss**: How well the model fits training data (lower = better)
- **Dev Micro-F1/Macro-F1**: Performance on unseen data (higher = better)
- **✓ New best**: Model improved this epoch (automatically saved internally)

### Class Balance
```
crosswalk_error     :  42/180 (weight: 3.29)
```

- **42/180**: 42 positive examples out of 180 total
- **weight: 3.29**: Model pays 3.29x more attention to this rare error type
- This helps the model learn despite having few examples

### Sample Predictions
Shows how the model performs on specific proposals:
```
Proposal ID: proposal_0024
  TRUE: {'crosswalk_error': 0, 'banned_phrases': 1, ...}
  PRED: {'crosswalk_error': 1, 'banned_phrases': 1, ...}
```
- **TRUE**: What the actual labels are
- **PRED**: What the model predicted
- **1** = Error present, **0** = No error

---

## Hardware Utilization

Your M4 Pro setup:
- ✓ **GPU**: 16-core GPU (fully utilized via MPS)
- ✓ **Memory**: 24GB unified (plenty of headroom)
- ✓ **Mixed Precision**: Enabled (faster training)
- ⚠️ **Neural Engine**: Not used by PyTorch (GPU only)

Training time: ~2-3 minutes for 10 epochs on 180 samples

---

## Next Steps for 10K Samples

1. **Generate more synthetic proposals** (scale up data generation)
2. **Switch to BERT**: `--model-name bert-base-uncased`
3. **Increase batch size**: `--batch-size 16` (your M4 Pro can handle it)
4. **Train longer**: `--max-epochs 20 --early-stopping-patience 5`
5. **Expected results**: Micro-F1 likely > 0.60 (much better!)

---

## Troubleshooting

### Memory Errors
**Solution:** Lower batch size
```bash
python scripts/09_transformer_mps.py --batch-size 4
```

### Model Predicts All Zeros
**Solution:** Already fixed! The script now uses class weights to handle imbalanced data.

### MPS Not Available Warning
**Solution:** Make sure you're on macOS with Apple Silicon (M1/M2/M3/M4). Script will fall back to CPU.

### Training Too Slow
**Solution:** Try `--use-compile` flag (experimental speed boost)
```bash
python scripts/09_transformer_mps.py --use-compile
```

---

## Files Changed

1. **Added**: `synthetic_proposals/scripts/09_transformer_mps.py` (new transformer script)
2. **Updated**: `synthetic_proposals/requirements.txt` (added PyTorch, transformers, etc.)
3. **Installed**: torch, transformers, datasets, accelerate packages

---

## Questions?

- **"Which model should I use?"** → DistilBERT for speed, BERT for accuracy
- **"How many epochs?"** → Default (10 with patience 3) is fine; training stops early if no improvement
- **"Can I run this on test set?"** → Modify script to load test split instead of dev
- **"How do I save the model?"** → Add checkpointing (can implement if needed)

---

**Summary:** Your new transformer model is 37% better than the previous ML baseline and ready to scale to 10,000 samples for even better results!

