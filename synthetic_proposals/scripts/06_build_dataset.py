"""Orchestrate the full synthetic proposal dataset build pipeline."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import common, pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end dataset build pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="knobs.yaml path (defaults to config/knobs.yaml).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root directory (defaults to dataset/).",
    )
    return parser.parse_args()


def _collect_summary(dataset_root: Path) -> Dict[str, int]:
    mistake_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    for split, proposal_dir in common.iter_proposal_dirs(dataset_root):
        split_counter[split] += 1
        meta_path = proposal_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = common.load_json_file(meta_path)
        for key, flagged in meta.get("mistakes", {}).items():
            if flagged:
                mistake_counter[key] += 1
    lines = ["Build summary:"]
    lines.append("Splits:" + ", ".join(f" {split}={count}" for split, count in sorted(split_counter.items())))
    lines.append("Mistakes:" + ", ".join(f" {key}={count}" for key, count in mistake_counter.most_common()))
    for line in lines:
        print(line)
    return dict(mistake_counter)


def _print_tree(dataset_root: Path) -> None:
    total_yaml = 0
    total_json = 0
    for split, proposal_dir in common.iter_proposal_dirs(dataset_root):
        total_yaml += 1 if (proposal_dir / "proposal.yaml").exists() else 0
        total_json += 1 if (proposal_dir / "proposal.json").exists() else 0
    print(f"Dataset tree summary: {total_yaml} YAML manifests, {total_json} JSON manifests")
    for split_dir in sorted((dataset_root / split for split in ["train", "dev", "test"] if (dataset_root / split).exists()), key=lambda p: p.name):
        count = len([p for p in split_dir.iterdir() if p.is_dir()])
        print(f"  {split_dir.name}: {count} proposals")


def main() -> None:
    args = parse_args()
    pipeline.build_dataset(config_path=args.config, dataset_root=args.dataset_root)
    summary_root = args.dataset_root or common.DATASET_ROOT
    _print_tree(summary_root)
    _collect_summary(summary_root)


if __name__ == "__main__":
    main()
