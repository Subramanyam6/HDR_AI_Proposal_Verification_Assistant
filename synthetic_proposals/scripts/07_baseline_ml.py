"""
Baseline text-only multi-label classifier (pure Python, no external deps).

- Reads aggregated dataset: synthetic_proposals/dataset/synthetic_proposals.json
- Uses train split for fitting; evaluates on dev split
- Predicts three text-driven labels:
  * crosswalk_error -> bool(labels.crosswalk_errors)
  * banned_phrases -> bool(labels.banned_phrases_found)
  * name_inconsistency -> not labels.name_consistency_flag

Outputs per-label precision/recall/F1/accuracy, micro/macro F1, and a few sample predictions.
Saves trained Naive Bayes models + metadata under huggingface_space/model/nb_baseline.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Sequence, Tuple


DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "synthetic_proposals.json"
MODEL_BASE_DIR = Path(__file__).resolve().parents[2] / "huggingface_space" / "model" / "nb_baseline"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunks_text(record: Dict) -> str:
    return " ".join((c.get("text", "") for c in record.get("chunks", [])))


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


Target = Tuple[str, Callable[[Dict], int]]


TARGETS: Sequence[Target] = (
    ("crosswalk_error", lambda y: int(bool(y.get("crosswalk_errors")))),
    ("banned_phrases", lambda y: int(bool(y.get("banned_phrases_found")))),
    ("name_inconsistency", lambda y: int(not y.get("name_consistency_flag", True))),
)


class NB:
    """Simple Multinomial Naive Bayes binary classifier."""

    def __init__(self) -> None:
        self.c1 = 0
        self.c0 = 0
        self.t1: Dict[str, int] = {}
        self.t0: Dict[str, int] = {}
        self.total_t1 = 0
        self.total_t0 = 0
        self.V = 1
        self.prior1 = 0.0
        self.prior0 = 0.0

    def fit(self, docs: Sequence[List[str]], ys: Sequence[int]) -> None:
        vocab = {}
        for tokens, y in zip(docs, ys):
            if y == 1:
                self.c1 += 1
                for w in tokens:
                    self.t1[w] = self.t1.get(w, 0) + 1
                    self.total_t1 += 1
            else:
                self.c0 += 1
                for w in tokens:
                    self.t0[w] = self.t0.get(w, 0) + 1
                    self.total_t0 += 1
            for w in tokens:
                vocab[w] = 1
        self.V = max(1, len(vocab))
        n = self.c1 + self.c0
        self.prior1 = math.log((self.c1 + 1) / (n + 2))
        self.prior0 = math.log((self.c0 + 1) / (n + 2))

    def predict_one(self, tokens: Sequence[str]) -> int:
        s1 = self.prior1
        s0 = self.prior0
        for w in tokens:
            n1 = self.t1.get(w, 0)
            n0 = self.t0.get(w, 0)
            s1 += math.log((n1 + 1) / (self.total_t1 + self.V))
            s0 += math.log((n0 + 1) / (self.total_t0 + self.V))
        return 1 if s1 >= s0 else 0


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "acc": acc}


def main() -> None:
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"
    data = load_dataset(DATA_PATH)
    train = [r for r in data if r.get("split") == "train"]
    dev = [r for r in data if r.get("split") == "dev"]

    models: Dict[str, NB] = {}
    per_label_results: Dict[str, Dict[str, float]] = {}

    # Fit a small NB per label
    for name, target_fn in TARGETS:
        X = [tokenize(chunks_text(r)) for r in train]
        y = [target_fn(r.get("labels", {})) for r in train]
        clf = NB()
        clf.fit(X, y)
        models[name] = clf

        # Evaluate on dev
        Xd = [tokenize(chunks_text(r)) for r in dev]
        yd = [target_fn(r.get("labels", {})) for r in dev]
        yp = [clf.predict_one(x) for x in Xd]
        per_label_results[name] = compute_metrics(yd, yp)

    # Micro/Macro F1 over all labels
    micro_true: List[int] = []
    micro_pred: List[int] = []
    macro_f1s: List[float] = []
    for name, target_fn in TARGETS:
        yd = [target_fn(r.get("labels", {})) for r in dev]
        Xd = [tokenize(chunks_text(r)) for r in dev]
        yp = [models[name].predict_one(x) for x in Xd]
        micro_true.extend(yd)
        micro_pred.extend(yp)
        macro_f1s.append(per_label_results[name]["f1"])
    micro = compute_metrics(micro_true, micro_pred)
    macro_f1 = mean(macro_f1s) if macro_f1s else 0.0

    # Print results
    print("Per-label metrics (dev):")
    for name in per_label_results:
        m = per_label_results[name]
        print(f"  {name:18s} -> precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} acc={m['acc']:.3f}")
    print(f"Micro-F1: {micro['f1']:.3f}  Macro-F1: {macro_f1:.3f}")

    # Show a few sample predictions
    print("\nSample dev predictions:")
    for r in dev[:5]:
        pred = {name: models[name].predict_one(tokenize(chunks_text(r))) for name, _ in TARGETS}
        truth = {
            "crosswalk_error": int(bool(r.get("labels", {}).get("crosswalk_errors"))),
            "banned_phrases": int(bool(r.get("labels", {}).get("banned_phrases_found"))),
            "name_inconsistency": int(not r.get("labels", {}).get("name_consistency_flag", True)),
        }
        print(f"  ID={r.get('id')} | TRUE={truth} | PRED={pred}")

    # Persist models + metadata for downstream use
    MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    print("\nSaving Naive Bayes baseline models...")
    saved_paths = []
    for name, clf in models.items():
        path = MODEL_BASE_DIR / f"{name}_nb.pkl"
        with path.open("wb") as f:
            pickle.dump(clf, f)
        saved_paths.append(str(path))
        print(f"  ✓ {name} -> {path}")

    metadata = {
        "timestamp_utc": timestamp,
        "dataset_path": str(DATA_PATH),
        "train_size": len(train),
        "dev_size": len(dev),
        "targets": [name for name, _ in TARGETS],
        "per_label_metrics": per_label_results,
        "micro_f1": micro["f1"],
        "macro_f1": macro_f1,
        "model_files": saved_paths,
    }
    metadata_path = MODEL_BASE_DIR / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata -> {metadata_path}")
    print("Baseline training complete.")


if __name__ == "__main__":
    main()
