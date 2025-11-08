#!/usr/bin/env python3
"""
Deep analysis of DistilBERT model to understand overfitting:
1. Check train/dev split quality
2. Analyze prediction distributions
3. Check for label distribution issues
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths
REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "huggingface_space" / "model" / "distilbert"
DATA_PATH = REPO_ROOT / "synthetic_proposals" / "dataset" / "synthetic_proposals.json"

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


def analyze_label_distribution(records: List[Dict], split_name: str):
    """Analyze label distribution in a split."""
    print(f"\n{'='*70}")
    print(f"Label Distribution: {split_name}")
    print(f"{'='*70}")
    
    label_counts = {name: [0, 0] for name, _ in TARGETS}  # [negative, positive]
    
    for record in records:
        labels = record.get("labels", {})
        for j, (name, target_fn) in enumerate(TARGETS):
            label = target_fn(labels)
            label_counts[name][label] += 1
    
    total = len(records)
    for name, (neg, pos) in label_counts.items():
        pos_pct = (pos / total) * 100 if total > 0 else 0
        print(f"  {name:20s}: {pos:4d} positive ({pos_pct:5.1f}%) | {neg:4d} negative ({100-pos_pct:5.1f}%)")
    
    return label_counts


def check_text_similarity(train_records: List[Dict], dev_records: List[Dict]):
    """Check if train and dev sets have similar text patterns."""
    print(f"\n{'='*70}")
    print("Train/Dev Text Similarity Analysis")
    print(f"{'='*70}")
    
    # Extract unique words from each split
    train_words = set()
    dev_words = set()
    
    for record in train_records[:100]:  # Sample for speed
        text = chunks_text(record).lower()
        train_words.update(text.split())
    
    for record in dev_records[:100]:  # Sample for speed
        text = chunks_text(record).lower()
        dev_words.update(text.split())
    
    overlap = len(train_words & dev_words)
    union = len(train_words | dev_words)
    jaccard = overlap / union if union > 0 else 0
    
    print(f"Train unique words (sample): {len(train_words)}")
    print(f"Dev unique words (sample): {len(dev_words)}")
    print(f"Overlap: {overlap}")
    print(f"Jaccard similarity: {jaccard:.3f}")
    
    if jaccard > 0.8:
        print("⚠️  WARNING: Very high word overlap suggests train/dev sets are too similar!")
    
    return jaccard


def analyze_predictions(model, tokenizer, device, records: List[Dict], split_name: str):
    """Analyze prediction distributions."""
    print(f"\n{'='*70}")
    print(f"Prediction Analysis: {split_name}")
    print(f"{'='*70}")
    
    all_probs = []
    all_labels = []
    
    for record in records:
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
        
        labels = np.zeros(len(TARGETS), dtype=np.float32)
        for j, (name, target_fn) in enumerate(TARGETS):
            labels[j] = target_fn(record.get("labels", {}))
        all_labels.append(labels)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"\nProbability distributions:")
    for j, (name, _) in enumerate(TARGETS):
        probs_j = all_probs[:, j]
        labels_j = all_labels[:, j]
        
        print(f"\n  {name}:")
        print(f"    Mean prob: {probs_j.mean():.4f}")
        print(f"    Std prob: {probs_j.std():.4f}")
        print(f"    Min prob: {probs_j.min():.4f}")
        print(f"    Max prob: {probs_j.max():.4f}")
        print(f"    Median prob: {np.median(probs_j):.4f}")
        
        # Check if predictions are too confident
        very_low = (probs_j < 0.1).sum()
        very_high = (probs_j > 0.9).sum()
        print(f"    Very low (<0.1): {very_low} ({very_low/len(probs_j)*100:.1f}%)")
        print(f"    Very high (>0.9): {very_high} ({very_high/len(probs_j)*100:.1f}%)")
        
        # Check separation between positive and negative
        pos_probs = probs_j[labels_j == 1]
        neg_probs = probs_j[labels_j == 0]
        if len(pos_probs) > 0 and len(neg_probs) > 0:
            print(f"    Pos mean prob: {pos_probs.mean():.4f}")
            print(f"    Neg mean prob: {neg_probs.mean():.4f}")
            separation = pos_probs.mean() - neg_probs.mean()
            print(f"    Separation: {separation:.4f}")
            
            if separation > 0.8:
                print(f"    ⚠️  WARNING: Very high separation suggests overfitting!")
    
    return all_probs, all_labels


def main():
    print("="*70)
    print("Deep DistilBERT Model Analysis")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load dataset
    print(f"\nLoading dataset from {DATA_PATH}...")
    data = load_dataset(DATA_PATH)
    train_records = [r for r in data if r.get("split") == "train"]
    dev_records = [r for r in data if r.get("split") == "dev"]
    print(f"Loaded {len(train_records)} train and {len(dev_records)} dev records")
    
    # Analyze label distributions
    train_labels = analyze_label_distribution(train_records, "Train")
    dev_labels = analyze_label_distribution(dev_records, "Dev")
    
    # Check if distributions are similar (potential issue)
    print(f"\n{'='*70}")
    print("Label Distribution Comparison")
    print(f"{'='*70}")
    for name, _ in TARGETS:
        train_pos_pct = (train_labels[name][1] / len(train_records)) * 100
        dev_pos_pct = (dev_labels[name][1] / len(dev_records)) * 100
        diff = abs(train_pos_pct - dev_pos_pct)
        print(f"  {name:20s}: Train={train_pos_pct:5.1f}% | Dev={dev_pos_pct:5.1f}% | Diff={diff:5.1f}%")
        if diff < 2.0:
            print(f"    ⚠️  WARNING: Very similar distributions!")
    
    # Check text similarity
    jaccard = check_text_similarity(train_records, dev_records)
    
    # Analyze predictions on dev set
    dev_probs, dev_labels = analyze_predictions(model, tokenizer, device, dev_records, "Dev")
    
    # Summary
    print(f"\n{'='*70}")
    print("ROOT CAUSE ANALYSIS")
    print(f"{'='*70}")
    
    issues = []
    if jaccard > 0.8:
        issues.append("High train/dev text similarity (Jaccard > 0.8)")
    
    # Check if model predictions are too confident
    for j, (name, _) in enumerate(TARGETS):
        probs_j = dev_probs[:, j]
        very_confident = ((probs_j < 0.1) | (probs_j > 0.9)).sum()
        if very_confident / len(probs_j) > 0.9:
            issues.append(f"Model too confident on '{name}' (>90% predictions <0.1 or >0.9)")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No obvious issues found in analysis")
    
    print("\n💡 RECOMMENDATIONS:")
    print("  1. Check if train/dev split is truly random")
    print("  2. Consider using a test set that's completely separate")
    print("  3. Add regularization (dropout, weight decay)")
    print("  4. Reduce model capacity or use early stopping more aggressively")
    print("  5. Check if synthetic data generation creates patterns that are too easy to memorize")


if __name__ == "__main__":
    main()

