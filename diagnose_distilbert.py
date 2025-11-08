#!/usr/bin/env python3
"""
Diagnostic script to investigate DistilBERT model issues:
1. Test model on clean sample
2. Check for data leakage
3. Verify evaluation metrics
4. Test on dev set
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths
REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "huggingface_space" / "model" / "distilbert"
DATA_PATH = REPO_ROOT / "synthetic_proposals" / "dataset" / "synthetic_proposals.json"
CLEAN_SAMPLE = REPO_ROOT / "huggingface_space" / "sample_clean.txt"

# Model configuration
TARGETS = [
    ("crosswalk_error", lambda y: int(len(y.get("crosswalk_errors", [])) > 0)),
    ("banned_phrases", lambda y: int(len(y.get("banned_phrases_found", [])) > 0)),
    ("name_inconsistency", lambda y: int(not y.get("name_consistency_flag", True))),
]
THRESHOLD = 0.5


def load_dataset(path: Path) -> List[Dict]:
    """Load the aggregated JSON dataset."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunks_text(record: Dict) -> str:
    """Extract and concatenate all chunk text from a proposal record."""
    return " ".join((c.get("text", "") for c in record.get("chunks", [])))


def load_model():
    """Load the DistilBERT model and tokenizer."""
    print(f"Loading model from {MODEL_DIR}...")
    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        sys.exit(1)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model, tokenizer, device


def test_sample(model, tokenizer, device, text: str, label: str = "sample"):
    """Test model on a text sample and return predictions."""
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    predictions = (probabilities > THRESHOLD).astype(int)
    
    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"{'='*70}")
    print(f"Text length: {len(text)} characters")
    print(f"\nPredictions (threshold={THRESHOLD}):")
    for idx, (name, _) in enumerate(TARGETS):
        prob = probabilities[idx]
        pred = bool(predictions[idx])
        status = "❌ FAIL" if pred else "✅ PASS"
        print(f"  {name:20s}: prob={prob:.4f} | {status}")
    
    return probabilities, predictions


def check_data_leakage(clean_text: str, train_records: List[Dict]) -> Dict:
    """Check if clean sample text appears in training data."""
    print(f"\n{'='*70}")
    print("Checking for data leakage...")
    print(f"{'='*70}")
    
    # Normalize text for comparison (lowercase, remove extra whitespace)
    clean_normalized = " ".join(clean_text.lower().split())
    clean_words = set(clean_normalized.split())
    
    matches = []
    for i, record in enumerate(train_records):
        train_text = chunks_text(record)
        train_normalized = " ".join(train_text.lower().split())
        
        # Check for exact match
        if clean_normalized == train_normalized:
            matches.append({
                "type": "exact_match",
                "record_id": record.get("id"),
                "index": i
            })
            continue
        
        # Check for high similarity (>90% word overlap)
        train_words = set(train_normalized.split())
        if len(train_words) > 0:
            overlap = len(clean_words & train_words) / len(clean_words)
            if overlap > 0.9:
                matches.append({
                    "type": "high_similarity",
                    "record_id": record.get("id"),
                    "index": i,
                    "overlap": overlap
                })
    
    if matches:
        print(f"⚠️  WARNING: Found {len(matches)} potential matches in training data!")
        for match in matches[:5]:  # Show first 5
            print(f"  - {match['type']}: record_id={match['record_id']}, index={match['index']}")
            if 'overlap' in match:
                print(f"    Word overlap: {match['overlap']:.2%}")
    else:
        print("✓ No data leakage detected - clean sample not found in training data")
    
    return {"matches": matches, "has_leakage": len(matches) > 0}


def evaluate_on_dev(model, tokenizer, device, dev_records: List[Dict]):
    """Evaluate model on dev set and compute metrics."""
    print(f"\n{'='*70}")
    print("Evaluating on dev set...")
    print(f"{'='*70}")
    
    all_probs = []
    all_labels = []
    
    for record in dev_records:
        text = chunks_text(record)
        inputs = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        all_probs.append(probs)
        
        # Extract true labels
        labels = np.zeros(len(TARGETS), dtype=np.float32)
        for j, (name, target_fn) in enumerate(TARGETS):
            labels[j] = target_fn(record.get("labels", {}))
        all_labels.append(labels)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    y_pred = (all_probs >= THRESHOLD).astype(int)
    
    print(f"\nDev set size: {len(dev_records)}")
    print(f"\nPer-label metrics:")
    
    per_label_results = {}
    for j, (name, _) in enumerate(TARGETS):
        y_true_j = all_labels[:, j]
        y_pred_j = y_pred[:, j]
        
        tp = int(((y_true_j == 1) & (y_pred_j == 1)).sum())
        fp = int(((y_true_j == 0) & (y_pred_j == 1)).sum())
        fn = int(((y_true_j == 1) & (y_pred_j == 0)).sum())
        tn = int(((y_true_j == 0) & (y_pred_j == 0)).sum())
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(y_true_j) if len(y_true_j) > 0 else 0.0
        
        per_label_results[name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "acc": acc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
        
        print(f"  {name:20s}: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} acc={acc:.3f}")
        print(f"    Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    
    # Micro F1
    micro_tp = int(((all_labels == 1) & (y_pred == 1)).sum())
    micro_fp = int(((all_labels == 0) & (y_pred == 1)).sum())
    micro_fn = int(((all_labels == 1) & (y_pred == 0)).sum())
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    
    # Macro F1
    macro_f1 = np.mean([per_label_results[name]["f1"] for name, _ in TARGETS])
    
    print(f"\nMicro-F1: {micro_f1:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}")
    
    # Check for suspicious F1=1.0
    if micro_f1 >= 0.99 or macro_f1 >= 0.99:
        print(f"\n⚠️  WARNING: Suspiciously high F1 score!")
        print(f"  This suggests possible overfitting or data leakage")
    
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_label": per_label_results
    }


def main():
    print("="*70)
    print("DistilBERT Model Diagnostic")
    print("="*70)
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Load dataset
    print(f"\nLoading dataset from {DATA_PATH}...")
    data = load_dataset(DATA_PATH)
    train_records = [r for r in data if r.get("split") == "train"]
    dev_records = [r for r in data if r.get("split") == "dev"]
    print(f"Loaded {len(train_records)} train and {len(dev_records)} dev records")
    
    # Load clean sample
    if not CLEAN_SAMPLE.exists():
        print(f"ERROR: Clean sample not found: {CLEAN_SAMPLE}")
        sys.exit(1)
    
    with CLEAN_SAMPLE.open("r", encoding="utf-8") as f:
        clean_text = f.read()
    
    # Test clean sample
    clean_probs, clean_preds = test_sample(model, tokenizer, device, clean_text, "Clean Sample")
    
    # Check for data leakage
    leakage_info = check_data_leakage(clean_text, train_records)
    
    # Evaluate on dev set
    dev_metrics = evaluate_on_dev(model, tokenizer, device, dev_records)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Clean sample predictions:")
    for idx, (name, _) in enumerate(TARGETS):
        prob = clean_probs[idx]
        pred = bool(clean_preds[idx])
        status = "❌ FAIL" if pred else "✅ PASS"
        print(f"  {name:20s}: {prob:.4f} | {status}")
    
    print(f"\nData leakage: {'⚠️  YES' if leakage_info['has_leakage'] else '✓ NO'}")
    print(f"Dev set Micro-F1: {dev_metrics['micro_f1']:.3f}")
    print(f"Dev set Macro-F1: {dev_metrics['macro_f1']:.3f}")
    
    # Check if banned_phrases is incorrectly flagged
    banned_idx = next(i for i, (name, _) in enumerate(TARGETS) if name == "banned_phrases")
    if clean_preds[banned_idx]:
        print(f"\n🔴 ISSUE FOUND: Clean sample incorrectly flagged for banned_phrases!")
        print(f"   Probability: {clean_probs[banned_idx]:.4f}")
        print(f"   Threshold: {THRESHOLD}")
        print(f"   This is the root cause of the problem.")


if __name__ == "__main__":
    main()

