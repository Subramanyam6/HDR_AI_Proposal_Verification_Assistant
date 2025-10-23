"""
Entry point for generating synthetic proposal YAML manifests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic proposal YAML manifests.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to knobs.yaml configuration (defaults to config/knobs.yaml).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Output dataset root directory (defaults to dataset/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline.generate_yaml_batch(config_path=args.config, dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
