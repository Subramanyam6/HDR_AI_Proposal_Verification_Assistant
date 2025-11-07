"""
TF-IDF + Logistic Regression multi-label classifier using scikit-learn.

- Reads aggregated dataset: synthetic_proposals/dataset/synthetic_proposals.json
- Uses train split for fitting; evaluates on dev split
- Predicts three text-driven labels:
  * crosswalk_error -> bool(labels.crosswalk_errors)
  * banned_phrases -> bool(labels.banned_phrases_found)
  * name_inconsistency -> not labels.name_consistency_flag

Outputs per-label precision/recall/F1/accuracy, micro/macro F1, and sample predictions.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "synthetic_proposals.json"

Target = Tuple[str, Callable[[Dict], int]]

# Define targets matching the Naive Bayes baseline
TARGETS: Sequence[Target] = (
    ("crosswalk_error", lambda y: int(bool(y.get("crosswalk_errors")))),
    ("banned_phrases", lambda y: int(bool(y.get("banned_phrases_found")))),
    ("name_inconsistency", lambda y: int(not y.get("name_consistency_flag", True))),
)


def load_dataset(path: Path) -> List[Dict]:
    """Load the aggregated JSON dataset."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunks_text(record: Dict) -> str:
    """Extract and concatenate all chunk text from a proposal record."""
    return " ".join((c.get("text", "") for c in record.get("chunks", [])))


def compute_per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, F1, and accuracy for a single label."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {"precision": prec, "recall": rec, "f1": f1, "acc": acc}


def main() -> None:
    # Sanity check: dataset exists
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    
    print("Loading dataset...")
    data = load_dataset(DATA_PATH)
    
    train_records = [r for r in data if r.get("split") == "train"]
    dev_records = [r for r in data if r.get("split") == "dev"]
    
    # Sanity check: splits exist
    if not train_records:
        print("ERROR: No train records found in dataset", file=sys.stderr)
        sys.exit(1)
    if not dev_records:
        print("ERROR: No dev records found in dataset", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(train_records)} train and {len(dev_records)} dev records.")
    
    # Extract text features
    print("\nExtracting text from proposals...")
    X_train_text = [chunks_text(r) for r in train_records]
    X_dev_text = [chunks_text(r) for r in dev_records]
    
    # Build multi-label target matrix (n_samples x n_labels)
    y_train = np.zeros((len(train_records), len(TARGETS)), dtype=int)
    y_dev = np.zeros((len(dev_records), len(TARGETS)), dtype=int)
    
    for j, (name, target_fn) in enumerate(TARGETS):
        for i, r in enumerate(train_records):
            y_train[i, j] = target_fn(r.get("labels", {}))
        for i, r in enumerate(dev_records):
            y_dev[i, j] = target_fn(r.get("labels", {}))
    
    # Sanity check: warn if any label is constant in train
    for j, (name, _) in enumerate(TARGETS):
        if y_train[:, j].sum() == 0:
            print(f"WARNING: Label '{name}' has no positive examples in train split.", file=sys.stderr)
        elif y_train[:, j].sum() == len(train_records):
            print(f"WARNING: Label '{name}' has all positive examples in train split.", file=sys.stderr)
    
    # TF-IDF Vectorization
    print("\nVectorizing text with TF-IDF (word + bigram)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2,                # ignore very rare terms
        max_df=0.95,             # ignore very common terms
        sublinear_tf=True,       # use log scaling
        lowercase=True,
        max_features=10000       # cap vocabulary size
    )
    
    X_train = vectorizer.fit_transform(X_train_text)
    X_dev = vectorizer.transform(X_dev_text)
    
    print(f"TF-IDF feature matrix: {X_train.shape[0]} train samples x {X_train.shape[1]} features")
    
    # Multi-label Logistic Regression (OneVsRest)
    print("\nTraining OneVsRest Logistic Regression (max_iter=1000, balanced)...")
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced',  # handle class imbalance
            random_state=42
        )
    )
    clf.fit(X_train, y_train)
    
    # Predict on dev
    print("Predicting on dev set...")
    y_pred = clf.predict(X_dev)
    
    # Compute per-label metrics
    print("\n" + "="*70)
    print("Per-label metrics (dev):")
    print("="*70)
    per_label_results = {}
    for j, (name, _) in enumerate(TARGETS):
        metrics = compute_per_label_metrics(y_dev[:, j], y_pred[:, j])
        per_label_results[name] = metrics
        print(f"  {name:20s} -> precision={metrics['precision']:.3f} "
              f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} acc={metrics['acc']:.3f}")
    
    # Micro-F1 and Macro-F1
    # Micro: flatten all labels and compute metrics
    micro_metrics = compute_per_label_metrics(y_dev.flatten(), y_pred.flatten())
    
    # Macro: average F1 across labels
    macro_f1 = np.mean([per_label_results[name]["f1"] for name, _ in TARGETS])
    
    print("="*70)
    print(f"Micro-F1: {micro_metrics['f1']:.3f}  |  Macro-F1: {macro_f1:.3f}")
    print("="*70)
    
    # Show 3 sample predictions
    print("\nSample dev predictions (showing first 3):")
    print("-"*70)
    for idx in range(min(3, len(dev_records))):
        r = dev_records[idx]
        true_dict = {name: int(y_dev[idx, j]) for j, (name, _) in enumerate(TARGETS)}
        pred_dict = {name: int(y_pred[idx, j]) for j, (name, _) in enumerate(TARGETS)}
        print(f"  Proposal ID: {r.get('id')}")
        print(f"    TRUE: {true_dict}")
        print(f"    PRED: {pred_dict}")
        print()

    # Save model to huggingface_space/model/
    model_dir = Path(__file__).resolve().parents[2] / "huggingface_space" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Saving TF-IDF model...")
    print(f"Model directory: {model_dir}")
    print("="*70)

    # Save vectorizer
    vectorizer_path = model_dir / "vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Saved vectorizer to {vectorizer_path}")

    # Save classifier
    classifier_path = model_dir / "classifier.pkl"
    with open(classifier_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"✓ Saved classifier to {classifier_path}")

    # Save config
    config = {
        "model_type": "tfidf_logreg",
        "labels": [name for name, _ in TARGETS],
        "micro_f1": float(micro_metrics["f1"]),
        "macro_f1": float(macro_f1),
        "per_label_f1": {name: float(per_label_results[name]["f1"]) for name, _ in TARGETS}
    }
    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config to {config_path}")

    print("\n✅ TF-IDF model saved successfully!")
    print("Done!")


if __name__ == "__main__":
    main()
