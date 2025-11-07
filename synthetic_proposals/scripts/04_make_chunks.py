"""Generate text chunks for embedding pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create normalized text chunks from proposal YAML.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root directory (defaults to dataset/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline.make_chunks(dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
