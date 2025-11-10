"""
Transformer-based multi-label classifier using DistilBERT with MPS acceleration.
ENHANCED VERSION with GPU verification and monitoring.

- Reads aggregated dataset: synthetic_proposals/dataset/synthetic_proposals.json
- Uses train split for fine-tuning; evaluates on dev split
- Predicts three text-driven labels:
  * crosswalk_error -> bool(labels.crosswalk_errors)
  * banned_phrases -> bool(labels.banned_phrases_found)
  * name_inconsistency -> not labels.name_consistency_flag

Optimized for Apple Silicon (M4 Pro) with MPS backend acceleration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    ("crosswalk_error", lambda y: int(len(y.get("crosswalk_errors", [])) > 0)),
    ("banned_phrases", lambda y: int(len(y.get("banned_phrases_found", [])) > 0)),
    ("name_inconsistency", lambda y: int(not y.get("name_consistency_flag", True))),
)

THRESHOLDS: Sequence[float] = (0.3, 0.5, 0.7)


def verify_device_usage(model, batch, device):
    """Verify that model and data are on the correct device."""
    # Check model parameters
    model_device = next(model.parameters()).device
    
    # Check batch tensors
    batch_devices = {k: v.device if isinstance(v, torch.Tensor) else None 
                     for k, v in batch.items()}
    
    return model_device, batch_devices


def get_mps_memory_allocated():
    """Get MPS memory allocated (if available)."""
    try:
        # PyTorch 2.0+ has MPS memory tracking
        if hasattr(torch.mps, 'current_allocated_memory'):
            return torch.mps.current_allocated_memory() / 1024**3  # GB
        return None
    except:
        return None


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
        
        return {
            "text": text,
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
    """Evaluate model and return probabilities and true labels."""
    model.eval()
    all_probs = []
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
            
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    return np.vstack(all_probs), np.vstack(all_labels)


def train_epoch(model, dataloader, optimizer, scheduler, device: torch.device, 
                use_autocast: bool = True, gradient_accumulation_steps: int = 1,
                pos_weight: torch.Tensor = None, grad_clip: float = 1.0, verbose: bool = False) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    # Custom loss function with class weights
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Verify device usage on first batch
        if step == 0 and verbose:
            model_dev, batch_devs = verify_device_usage(model, 
                {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}, 
                device)
            print(f"\n  Device Verification:")
            print(f"    Model on: {model_dev}")
            print(f"    input_ids on: {batch_devs['input_ids']}")
            print(f"    attention_mask on: {batch_devs['attention_mask']}")
            print(f"    labels on: {batch_devs['labels']}")
            mem = get_mps_memory_allocated()
            if mem:
                print(f"    MPS Memory: {mem:.2f} GB")
        
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
    
    elapsed = time.time() - start_time
    
    return total_loss / len(dataloader), elapsed


def main():
    parser = argparse.ArgumentParser(description="Train transformer model with MPS acceleration")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased",
                        help="HuggingFace model name (default: distilbert-base-uncased)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training (default: 4, optimized for MPS)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                        help="Gradient accumulation steps (default: 8, effective batch size: 32)")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for DistilBERT layers (default: 5e-5; classifier head uses 5x)")
    parser.add_argument("--max-epochs", type=int, default=20,
                        help="Maximum number of epochs (default: 20)")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping value (default: 1.0)")

    parser.add_argument("--use-compile", action="store_true",
                        help="Use torch.compile() for optimization (experimental on MPS)")
    parser.add_argument("--no-autocast", action="store_true",
                        help="Disable mixed precision with autocast")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU usage for comparison")
    args = parser.parse_args()
    if args.early_stopping_patience > 3:
        print("⚠️  Capping early stopping patience to 3 to avoid prolonged plateaus.")
        args.early_stopping_patience = 3
    
    # Sanity check: dataset exists
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    
    print("="*70)
    print("GPU/MPS VERIFICATION")
    print("="*70)

    # Clear MPS cache and run garbage collection before starting
    import gc
    if torch.backends.mps.is_available():
        print("Clearing MPS cache from previous runs...")
        torch.mps.empty_cache()
        gc.collect()
        print("✓ MPS cache cleared")

    # Check MPS availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if args.force_cpu:
        device = torch.device("cpu")
        print(f"\n⚠️  FORCED CPU MODE (for comparison)")
    elif not torch.backends.mps.is_available():
        print("\n⚠️  WARNING: MPS not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
        print(f"\n✓ Using MPS (Apple Silicon GPU acceleration)")
        
        # Try to set MPS memory limit (if available)
        if hasattr(torch.mps, 'set_per_process_memory_fraction'):
            try:
                torch.mps.set_per_process_memory_fraction(0.8)
                print(f"  MPS memory fraction set to 0.8")
            except:
                pass
    
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
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
    
    # Move model to device and verify
    print(f"Moving model to {device}...")
    model = model.to(device)
    
    # Verify model is on correct device
    model_device = next(model.parameters()).device
    print(f"✓ Model parameters verified on: {model_device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create datasets and dataloaders
    print("\nCreating dataloaders...")
    train_dataset = ProposalDataset(train_records, tokenizer, max_length=args.max_length)
    dev_dataset = ProposalDataset(dev_records, tokenizer, max_length=args.max_length)
    
    def collate_fn(batch):
        texts = [example["text"] for example in batch]
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
        encodings["labels"] = torch.stack([example["labels"] for example in batch])
        return encodings
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Compute positive class weights for handling class imbalance
    pos_counts = train_dataset.labels.sum(axis=0)
    neg_counts = len(train_dataset) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).to(device)
    
    print(f"\nClass balance (positive samples per label):")
    for j, (name, _) in enumerate(TARGETS):
        print(f"  {name:20s}: {int(pos_counts[j]):3d}/{len(train_dataset)} (weight: {pos_weight[j].item():.2f})")
    
    # Setup optimizer and scheduler
    total_steps = len(train_loader) * args.max_epochs // args.gradient_accumulation_steps
    warmup_steps = int(0.15 * total_steps)
    
    classifier_lr = args.learning_rate * 5.0
    optimizer = torch.optim.AdamW([
        {'params': model.distilbert.parameters(), 'lr': args.learning_rate},
        {'params': model.classifier.parameters(), 'lr': classifier_lr},
        {'params': model.pre_classifier.parameters(), 'lr': classifier_lr}
    ], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # DEBUG: Check learning rate calculation
    print(f"DEBUG: args.learning_rate = {args.learning_rate}")
    print(f"DEBUG: classifier_lr calculation = {args.learning_rate} * 5.0 = {classifier_lr}")

    print(f"\nLearning rates:")
    print(f"  DistilBERT layers: {args.learning_rate}")
    print(f"  Classifier layer: {classifier_lr}")
    
    print(f"\nTotal training steps: {total_steps} (warmup: {warmup_steps})")
    
    # Training loop with early stopping
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    best_micro_f1 = 0.0
    patience_counter = 0
    perfect_f1_counter = 0  # Track consecutive epochs with F1=1.0
    use_autocast = not args.no_autocast

    # Model save path - save directly to backend/app/models
    model_save_path = Path(__file__).resolve().parents[2] / "backend" / "app" / "models" / "distilbert"
    model_save_path.mkdir(parents=True, exist_ok=True)
    print(f"\nModel will be saved to: {model_save_path}")
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        
        # Train (verbose on first epoch to verify device)
        train_loss, train_time = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            use_autocast=use_autocast,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            pos_weight=pos_weight,
            grad_clip=args.grad_clip,
            verbose=(epoch == 0)
        )
        
        samples_per_sec = len(train_dataset) / train_time
        print(f"  Train loss: {train_loss:.4f} | Time: {train_time:.1f}s | Speed: {samples_per_sec:.1f} samples/s")
        
        # Monitor classifier weights to ensure they're updating
        classifier_weight_norm = model.classifier.weight.norm().item()
        classifier_bias_norm = model.classifier.bias.norm().item()
        print(f"  Classifier weight norm: {classifier_weight_norm:.4f} | Bias norm: {classifier_bias_norm:.4f}")
        if epoch == 0 and classifier_bias_norm < 0.1:
            print(f"  ⚠️  WARNING: Bias norm very small - classifier may not be learning!")
        
        # Evaluate on dev
        eval_start = time.time()
        y_probs, y_true = evaluate_model(model, dev_loader, device, use_autocast=use_autocast)
        eval_time = time.time() - eval_start

        avg_probs = y_probs.mean(axis=0)
        print("  Avg dev probabilities:")
        for j, (name, _) in enumerate(TARGETS):
            print(f"    {name:20s}: {avg_probs[j]:.3f}")

        print("  Threshold sweep (micro metrics):")
        micro_metrics_by_threshold = {}
        for thr in THRESHOLDS:
            y_pred_thr = (y_probs >= thr).astype(int)
            metrics_thr = compute_per_label_metrics(y_true.flatten(), y_pred_thr.flatten())
            micro_metrics_by_threshold[thr] = metrics_thr
            print(f"    τ={thr:.1f}: precision={metrics_thr['precision']:.3f} "
                  f"recall={metrics_thr['recall']:.3f} f1={metrics_thr['f1']:.3f}")

        default_threshold = 0.5
        y_pred = (y_probs >= default_threshold).astype(int)
        
        # Compute metrics
        per_label_results = {}
        for j, (name, _) in enumerate(TARGETS):
            metrics = compute_per_label_metrics(y_true[:, j], y_pred[:, j])
            per_label_results[name] = metrics
        
        # Micro F1
        micro_metrics = micro_metrics_by_threshold[default_threshold]
        micro_f1 = micro_metrics["f1"]
        
        # Macro F1
        macro_f1 = np.mean([per_label_results[name]["f1"] for name, _ in TARGETS])
        
        print(f"  Dev Micro-F1: {micro_f1:.3f}  |  Macro-F1: {macro_f1:.3f}  |  Eval time: {eval_time:.1f}s")

        # Check for perfect F1
        if macro_f1 >= 1.0:
            perfect_f1_counter += 1
            print(f"  🎯 PERFECT F1 achieved! ({perfect_f1_counter}/2)")
        else:
            perfect_f1_counter = 0  # Reset if not perfect

        # Early stopping check
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            patience_counter = 0
            print(f"  ✓ New best Micro-F1: {best_micro_f1:.3f}")

            # Save best model
            print(f"  💾 Saving best model to {model_save_path}...")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"  ✓ Model saved successfully")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stopping_patience})")

            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Perfect F1 early stopping
        if perfect_f1_counter >= 2:
            print(f"\n🎉 PERFECT F1 achieved for 2 consecutive epochs!")
            print(f"  Stopping training early - model has converged to perfection")
            # Save final perfect model
            print(f"  💾 Saving final perfect model to {model_save_path}...")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"  ✓ Model saved successfully")
            break
    
    # Final evaluation
    print("\n" + "="*70)
    print("Final evaluation on dev set:")
    print("="*70)
    
    y_probs, y_true = evaluate_model(model, dev_loader, device, use_autocast=use_autocast)

    avg_probs = y_probs.mean(axis=0)
    print("  Avg dev probabilities:")
    for j, (name, _) in enumerate(TARGETS):
        print(f"    {name:20s}: {avg_probs[j]:.3f}")

    print("\n  Threshold sweep (micro metrics):")
    micro_metrics_by_threshold = {}
    for thr in THRESHOLDS:
        y_pred_thr = (y_probs >= thr).astype(int)
        metrics_thr = compute_per_label_metrics(y_true.flatten(), y_pred_thr.flatten())
        micro_metrics_by_threshold[thr] = metrics_thr
        print(f"    τ={thr:.1f}: precision={metrics_thr['precision']:.3f} "
              f"recall={metrics_thr['recall']:.3f} f1={metrics_thr['f1']:.3f}")

    default_threshold = 0.5
    y_pred = (y_probs >= default_threshold).astype(int)
    
    # Per-label metrics
    per_label_results = {}
    for j, (name, _) in enumerate(TARGETS):
        metrics = compute_per_label_metrics(y_true[:, j], y_pred[:, j])
        per_label_results[name] = metrics
        print(f"  {name:20s} -> precision={metrics['precision']:.3f} "
              f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} acc={metrics['acc']:.3f}")
    
    # Micro and Macro F1
    micro_metrics = micro_metrics_by_threshold[default_threshold]
    macro_f1 = np.mean([per_label_results[name]["f1"] for name, _ in TARGETS])
    
    print("="*70)
    print(f"Micro-F1: {micro_metrics['f1']:.3f}  |  Macro-F1: {macro_f1:.3f}")
    print("="*70)
    
    # Show 3 sample predictions
    print("\nSample dev predictions (showing first 3):")
    print("-"*70)
    for idx in range(min(3, len(dev_records))):
        r = dev_records[idx]
        true_dict = {name: int(y_true[idx, j]) for j, (name, _) in enumerate(TARGETS)}
        pred_dict = {name: int(y_pred[idx, j]) for j, (name, _) in enumerate(TARGETS)}
        print(f"  Proposal ID: {r.get('id')}")
        print(f"    TRUE: {true_dict}")
        print(f"    PRED: {pred_dict}")
        print()

    # Model already saved during training (best checkpoint)
    print(f"\n✓ Best model saved to: {model_save_path}")
    print(f"  Files: {list(model_save_path.glob('*'))}")

    # Final MPS memory check
    if device.type == "mps":
        mem = get_mps_memory_allocated()
        if mem:
            print(f"\nFinal MPS Memory Usage: {mem:.2f} GB")

    print("Done!")


if __name__ == "__main__":
    main()
