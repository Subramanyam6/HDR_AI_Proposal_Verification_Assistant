"""
Transformer-based multi-label classifier using DistilBERT with MPS acceleration.

- Reads aggregated dataset: synthetic_proposals/dataset/synthetic_proposals.json
- Uses train split for fine-tuning; evaluates on dev split
- Predicts four text-driven labels:
  * crosswalk_error -> bool(labels.crosswalk_errors)
  * banned_phrases -> bool(labels.banned_phrases_found)
  * name_inconsistency -> not labels.name_consistency_flag
  * date_inconsistency -> not labels.date_consistency_flag

Optimized for Apple Silicon (M4 Pro) with MPS backend acceleration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)


DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "synthetic_proposals.json"

Target = Tuple[str, Callable[[Dict], int]]

# Define targets matching existing baselines
TARGETS: Sequence[Target] = (
    ("crosswalk_error", lambda y: int(bool(y.get("crosswalk_errors")))),
    ("banned_phrases", lambda y: int(bool(y.get("banned_phrases_found")))),
    ("name_inconsistency", lambda y: int(not y.get("name_consistency_flag", True))),
    ("date_inconsistency", lambda y: int(not y.get("date_consistency_flag", True))),
)


def load_dataset(path: Path) -> List[Dict]:
    """Load the aggregated JSON dataset."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunks_text(record: Dict) -> str:
    """Extract and concatenate all chunk text from a proposal record."""
    return " ".join((c.get("text", "") for c in record.get("chunks", [])))


class ProposalDataset(Dataset):
    """PyTorch Dataset wrapper for proposal records."""
    
    def __init__(self, records: List[Dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract texts
        self.texts = [chunks_text(r) for r in records]
        
        # Extract labels (multi-label binary matrix)
        self.labels = np.zeros((len(records), len(TARGETS)), dtype=np.float32)
        for i, r in enumerate(records):
            for j, (name, target_fn) in enumerate(TARGETS):
                self.labels[i, j] = target_fn(r.get("labels", {}))
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32)
        }


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


def evaluate_model(model, dataloader, device: torch.device, use_autocast: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model and return predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            # Forward pass with optional autocast
            if use_autocast and device.type == "mps":
                with torch.autocast(device_type="mps"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Apply sigmoid for multi-label classification
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)


def train_epoch(model, dataloader, optimizer, scheduler, device: torch.device, 
                use_autocast: bool = True, gradient_accumulation_steps: int = 1,
                pos_weight: torch.Tensor = None) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    # Custom loss function with class weights
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass with optional autocast
        if use_autocast and device.type == "mps":
            with torch.autocast(device_type="mps"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
        
        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train transformer model with MPS acceleration")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased",
                        help="HuggingFace model name (default: distilbert-base-uncased)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training (default: 8)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Maximum number of epochs (default: 10)")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--use-compile", action="store_true",
                        help="Use torch.compile() for optimization (experimental on MPS)")
    parser.add_argument("--no-autocast", action="store_true",
                        help="Disable mixed precision with autocast")
    args = parser.parse_args()
    
    # Sanity check: dataset exists
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print(f"✓ Using MPS (Apple Silicon GPU acceleration)")
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    print(f"  Mixed precision (autocast): {not args.no_autocast}")
    print(f"  torch.compile(): {args.use_compile}")
    
    # Load dataset
    print(f"\nLoading dataset from {DATA_PATH}...")
    data = load_dataset(DATA_PATH)
    
    train_records = [r for r in data if r.get("split") == "train"]
    dev_records = [r for r in data if r.get("split") == "dev"]
    
    if not train_records:
        print("ERROR: No train records found in dataset", file=sys.stderr)
        sys.exit(1)
    if not dev_records:
        print("ERROR: No dev records found in dataset", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(train_records)} train and {len(dev_records)} dev records.")
    
    # Initialize tokenizer and model
    print(f"\nInitializing tokenizer and model ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Multi-label classification setup
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(TARGETS),
        problem_type="multi_label_classification"
    )
    
    # Optional torch.compile
    if args.use_compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    model = model.to(device)
    
    # Create datasets and dataloaders
    print("Creating dataloaders...")
    train_dataset = ProposalDataset(train_records, tokenizer, max_length=args.max_length)
    dev_dataset = ProposalDataset(dev_records, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Compute positive class weights for handling class imbalance
    pos_counts = train_dataset.labels.sum(axis=0)
    neg_counts = len(train_dataset) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).to(device)
    
    print(f"\nClass balance (positive samples per label):")
    for j, (name, _) in enumerate(TARGETS):
        print(f"  {name:20s}: {int(pos_counts[j]):3d}/{len(train_dataset)} (weight: {pos_weight[j].item():.2f})")
    
    # Setup optimizer and scheduler
    total_steps = len(train_loader) * args.max_epochs // args.gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    print(f"\nTotal training steps: {total_steps} (warmup: {warmup_steps})")
    
    # Training loop with early stopping
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    best_micro_f1 = 0.0
    patience_counter = 0
    use_autocast = not args.no_autocast
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            use_autocast=use_autocast,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            pos_weight=pos_weight
        )
        print(f"  Train loss: {train_loss:.4f}")
        
        # Evaluate on dev
        y_pred, y_true = evaluate_model(model, dev_loader, device, use_autocast=use_autocast)
        
        # Compute metrics
        per_label_results = {}
        for j, (name, _) in enumerate(TARGETS):
            metrics = compute_per_label_metrics(y_true[:, j], y_pred[:, j])
            per_label_results[name] = metrics
        
        # Micro F1
        micro_metrics = compute_per_label_metrics(y_true.flatten(), y_pred.flatten())
        micro_f1 = micro_metrics['f1']
        
        # Macro F1
        macro_f1 = np.mean([per_label_results[name]["f1"] for name, _ in TARGETS])
        
        print(f"  Dev Micro-F1: {micro_f1:.3f}  |  Macro-F1: {macro_f1:.3f}")
        
        # Early stopping check
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            patience_counter = 0
            print(f"  ✓ New best Micro-F1: {best_micro_f1:.3f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stopping_patience})")
            
            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Final evaluation
    print("\n" + "="*70)
    print("Final evaluation on dev set:")
    print("="*70)
    
    y_pred, y_true = evaluate_model(model, dev_loader, device, use_autocast=use_autocast)
    
    # Per-label metrics
    per_label_results = {}
    for j, (name, _) in enumerate(TARGETS):
        metrics = compute_per_label_metrics(y_true[:, j], y_pred[:, j])
        per_label_results[name] = metrics
        print(f"  {name:20s} -> precision={metrics['precision']:.3f} "
              f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} acc={metrics['acc']:.3f}")
    
    # Micro and Macro F1
    micro_metrics = compute_per_label_metrics(y_true.flatten(), y_pred.flatten())
    macro_f1 = np.mean([per_label_results[name]["f1"] for name, _ in TARGETS])
    
    print("="*70)
    print(f"Micro-F1: {micro_metrics['f1']:.3f}  |  Macro-F1: {macro_f1:.3f}")
    print("="*70)
    
    # Show 3 sample predictions
    print("\nSample dev predictions (showing first 3):")
    print("-"*70)
    for idx in range(min(3, len(dev_records))):
        r = dev_records[idx]
        true_dict = {
            "crosswalk_error": int(y_true[idx, 0]),
            "banned_phrases": int(y_true[idx, 1]),
            "name_inconsistency": int(y_true[idx, 2]),
            "date_inconsistency": int(y_true[idx, 3]),
        }
        pred_dict = {
            "crosswalk_error": int(y_pred[idx, 0]),
            "banned_phrases": int(y_pred[idx, 1]),
            "name_inconsistency": int(y_pred[idx, 2]),
            "date_inconsistency": int(y_pred[idx, 3]),
        }
        print(f"  Proposal ID: {r.get('id')}")
        print(f"    TRUE: {true_dict}")
        print(f"    PRED: {pred_dict}")
        print()
    
    print("Done!")


if __name__ == "__main__":
    main()

