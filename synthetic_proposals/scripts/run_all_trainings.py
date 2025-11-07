#!/usr/bin/env python3
"""
Utility to execute all three training pipelines back-to-back:
  1. 07_baseline_ml.py          (Naive Bayes baseline)
  2. 08_tfidf_logreg.py         (TF-IDF + Logistic Regression)
  3. 09_transformer_mps_verified.py (DistilBERT fine-tuning on MPS)

Each step streams its stdout/stderr into a timestamped log file under
<repo>/training_logs/<timestamp> and surfaces failures immediately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_LOG_ROOT = REPO_ROOT / "training_logs"


def build_steps(transformer_args: List[str]) -> List[dict]:
    scripts_dir = REPO_ROOT / "synthetic_proposals" / "scripts"
    return [
        {
            "name": "naive_bayes",
            "script": scripts_dir / "07_baseline_ml.py",
            "args": [],
        },
        {
            "name": "tfidf_logreg",
            "script": scripts_dir / "08_tfidf_logreg.py",
            "args": [],
        },
        {
            "name": "distilbert_transformer",
            "script": scripts_dir / "09_transformer_mps_verified.py",
            "args": transformer_args,
        },
    ]


def run_step(index: int, name: str, script: Path, args: List[str], log_dir: Path) -> dict:
    log_path = log_dir / f"{index:02d}_{name}.log"
    cmd = [sys.executable, str(script), *args]
    start_ts = datetime.now(timezone.utc).isoformat()
    log_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        header = f"# Command: {' '.join(cmd)}\n# Started: {start_ts}\n\n"
        log_file.write(header)
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None  # for mypy/static
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        process.wait()

        end_ts = datetime.now(timezone.utc).isoformat()
        footer = f"\n# Finished: {end_ts}\n# Return code: {process.returncode}\n"
        log_file.write(footer)

    status = "success" if process.returncode == 0 else "failure"
    print(f"[{status.upper()}] {name} -> {log_path}")
    if process.returncode != 0:
        raise SystemExit(f"{name} failed. Inspect log: {log_path}")
    return {
        "name": name,
        "command": cmd,
        "log_path": str(log_path),
        "started": start_ts,
        "finished": end_ts,
        "return_code": process.returncode,
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all training pipelines sequentially.")
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional note saved alongside the run summary.",
    )
    parser.add_argument(
        "--learning-rate",
        type=str,
        default="5e-5",
        help="Learning rate for the transformer training step.",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="6",
        help="Batch size for the transformer training step.",
    )
    parser.add_argument(
        "--grad-accum",
        type=str,
        default="4",
        help="Gradient accumulation steps for the transformer training step.",
    )
    parser.add_argument(
        "--max-epochs",
        type=str,
        default="20",
        help="Maximum epochs for the transformer training step.",
    )
    parser.add_argument(
        "--early-stopping",
        type=str,
        default="5",
        help="Early stopping patience for the transformer training step.",
    )
    parser.add_argument(
        "--grad-clip",
        type=str,
        default="1.0",
        help="Gradient clipping value for the transformer training step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = TRAINING_LOG_ROOT / timestamp
    transformer_args = [
        "--learning-rate",
        args.learning_rate,
        "--batch-size",
        args.batch_size,
        "--gradient-accumulation-steps",
        args.grad_accum,
        "--max-epochs",
        args.max_epochs,
        "--early-stopping-patience",
        args.early_stopping,
        "--grad-clip",
        args.grad_clip,
    ]
    steps = build_steps(transformer_args)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
        "log_directory": str(run_dir),
        "steps": [],
    }

    print(f"Logs will be written to: {run_dir}")
    for idx, step in enumerate(steps, start=1):
        info = run_step(idx, step["name"], step["script"], step["args"], run_dir)
        summary["steps"].append(info)

    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll trainings completed. Summary -> {summary_path}")


if __name__ == "__main__":
    main()
