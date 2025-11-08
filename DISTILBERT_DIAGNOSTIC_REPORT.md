# DistilBERT Model Diagnostic Report

## Executive Summary

**Critical Issues Found:**
1. ✅ **Severe Overfitting**: Model achieves F1=1.0 after just 2 epochs (stopped early)
2. ✅ **Clean Sample False Positive**: Clean sample incorrectly flagged for banned_phrases (prob=0.9820)
3. ✅ **Overconfident Predictions**: >90% of predictions are either <0.1 or >0.9 (extreme confidence)

## Root Cause Analysis

### 1. Overfitting Evidence

**Perfect F1 Score:**
- Dev set Micro-F1: 1.000
- Dev set Macro-F1: 1.000
- Achieved after only 2 epochs
- Training stopped early due to "perfect F1" condition

**Prediction Distribution:**
- **crosswalk_error**: Separation = 0.9994 (Pos mean: 0.9996, Neg mean: 0.0002)
- **banned_phrases**: Separation = 0.9997 (Pos mean: 0.9999, Neg mean: 0.0002)
- **name_inconsistency**: Separation = 0.9995 (Pos mean: 0.9998, Neg mean: 0.0003)

**Interpretation:** The model has learned to be extremely confident, with near-perfect separation between positive and negative examples. This is a classic sign of memorization rather than generalization.

### 2. Clean Sample False Positive

**Issue:** Clean sample has probability 0.9820 for banned_phrases (should be ~0.0)

**Root Cause:** The clean sample contains the word "assurance" in line 29:
```
"This provides an assurance that our delivery method avoids scope drift."
```

The model has overfitted to associate "assurance" with banned phrases (since "absolute assurance" is a banned phrase). It now flags any occurrence of "assurance" even when used legitimately.

### 3. Data Quality Issues

**Label Distribution Similarity:**
- Train/dev label distributions differ by <2% for all labels
- This suggests train/dev splits may be too similar

**Text Similarity:**
- Jaccard similarity: 0.305 (moderate, not the main issue)

## Recommendations

### Immediate Fixes

1. **Add Regularization**
   - Increase dropout rate (currently using default)
   - Increase weight decay (currently 0.01, try 0.1)
   - Add label smoothing (0.1-0.2)

2. **Fix Early Stopping**
   - Remove the "perfect F1" early stopping condition
   - Use standard early stopping based on validation loss
   - Don't stop training just because F1=1.0 (this is suspicious!)

3. **Adjust Training Parameters**
   - Reduce learning rate (try 1e-5 instead of 5e-5)
   - Reduce classifier learning rate multiplier (currently 5x, try 2x)
   - Train for more epochs with proper early stopping

4. **Improve Data Split**
   - Ensure train/dev/test splits are truly random
   - Consider stratified splitting to maintain label distributions
   - Use a completely separate test set for final evaluation

### Long-term Improvements

1. **Data Augmentation**
   - Add text augmentation (synonym replacement, paraphrasing)
   - Increase dataset size if possible

2. **Model Architecture**
   - Consider using a smaller model or adding more dropout
   - Use ensemble methods to reduce overfitting

3. **Evaluation**
   - Add a separate test set that's never seen during training
   - Monitor both F1 and loss on validation set
   - Track prediction confidence distributions

## Action Plan

### Step 1: Fix Training Script
- Remove "perfect F1" early stopping
- Add regularization (dropout, weight decay, label smoothing)
- Adjust learning rates

### Step 2: Retrain Model
- Train with new parameters
- Monitor validation loss, not just F1
- Stop when validation loss stops improving

### Step 3: Evaluate on Clean Sample
- Test retrained model on clean sample
- Verify banned_phrases probability is low (<0.1)

### Step 4: Test on Real Data
- Evaluate on a diverse set of real proposals
- Monitor for false positives/negatives

## Code Changes Needed

### 1. Training Script (`09_transformer_mps_verified.py`)

**Remove perfect F1 early stopping:**
```python
# REMOVE THIS:
if perfect_f1_counter >= 2:
    print(f"\n🎉 PERFECT F1 achieved for 2 consecutive epochs!")
    # ... early stop
```

**Add label smoothing:**
```python
# Use BCEWithLogitsLoss with label smoothing
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
# Apply label smoothing manually or use a library
```

**Increase regularization:**
```python
optimizer = torch.optim.AdamW([
    {'params': model.distilbert.parameters(), 'lr': args.learning_rate},
    {'params': model.classifier.parameters(), 'lr': classifier_lr},
    {'params': model.pre_classifier.parameters(), 'lr': classifier_lr}
], weight_decay=0.1)  # Increase from 0.01 to 0.1
```

**Add dropout:**
```python
# Check if model has dropout, increase it
# DistilBERT typically has dropout in the classifier
```

### 2. Early Stopping Logic

**Change to use validation loss:**
```python
best_val_loss = float('inf')
patience_counter = 0

# In training loop:
val_loss = compute_validation_loss(model, dev_loader, device)
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    # Save model
else:
    patience_counter += 1
    if patience_counter >= args.early_stopping_patience:
        break
```

## Expected Outcomes

After implementing fixes:
- F1 score should be more realistic (0.6-0.8 range)
- Prediction confidence should be more calibrated
- Clean sample should pass banned_phrases check
- Model should generalize better to unseen data

## Files Modified

- `synthetic_proposals/scripts/09_transformer_mps_verified.py` - Training script
- `diagnose_distilbert.py` - Diagnostic script (created)
- `analyze_overfitting.py` - Overfitting analysis script (created)

## Next Steps

1. Review and approve recommended changes
2. Implement fixes in training script
3. Retrain model with new parameters
4. Verify clean sample passes
5. Deploy updated model

