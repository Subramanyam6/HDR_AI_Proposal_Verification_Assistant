"""
Shared utilities for the synthetic proposal dataset pipeline.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from faker import Faker


logger = logging.getLogger("synthetic_proposals")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
TEMPLATE_ROOT = REPO_ROOT / "templates"
CONFIG_ROOT = REPO_ROOT / "config"
DATASET_ROOT = REPO_ROOT / "dataset"


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_yaml_file(path: Path) -> Any:
    """Load YAML from the given path."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml_file(data: Any, path: Path) -> None:
    """Write YAML data to the given path."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def load_json_file(path: Path) -> Any:
    """Load JSON from a file path."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json_file(data: Any, path: Path, *, indent: int = 2) -> None:
    """Persist JSON data to disk."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_all_yaml_in_dir(directory: Path) -> List[Any]:
    """Load all YAML files in a directory (non-recursive)."""
    payload: List[Any] = []
    for path in sorted(directory.glob("*.yaml")):
        payload.append(load_yaml_file(path))
    return payload


def load_knobs(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load pipeline knobs configuration."""
    path = config_path or CONFIG_ROOT / "knobs.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return load_yaml_file(path)


def load_dictionaries() -> Dict[str, Any]:
    """Load structured dictionaries used during generation."""
    dictionaries: Dict[str, Any] = {}
    if not DATA_ROOT.exists():
        logger.warning("Data root %s does not exist; returning empty dictionaries.", DATA_ROOT)
        return dictionaries
    dict_dir = DATA_ROOT / "dictionaries"
    for dict_path in dict_dir.glob("*.yaml"):
        dictionaries[dict_path.stem] = load_yaml_file(dict_path)
    return dictionaries


def load_seeds() -> List[Dict[str, Any]]:
    """Load seed proposal YAML fixtures."""
    seeds_dir = DATA_ROOT / "seeds"
    if not seeds_dir.exists():
        return []
    seeds: List[Dict[str, Any]] = []
    for seed_file in sorted(seeds_dir.glob("*.yaml")):
        seeds.append(load_yaml_file(seed_file))
    return seeds


def slugify(value: str) -> str:
    """Create a filesystem-friendly slug."""
    trimmed = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in trimmed:
        trimmed = trimmed.replace("--", "-")
    return trimmed.strip("-") or "value"


def random_phone(rng: random.Random) -> str:
    """Generate a fake North American phone number string."""
    area = rng.randint(200, 989)
    exchange = rng.randint(200, 998)
    subscriber = rng.randint(1000, 9999)
    return f"{area}-{exchange}-{subscriber}"


@dataclass
class RandomSource:
    """Convenience wrapper around random.Random and Faker instances."""

    seed: int
    locale: str = "en_US"
    _random: random.Random = None  # type: ignore[assignment]
    _faker: Faker = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)
        self._faker = Faker(self.locale)
        self._faker.seed_instance(self.seed)

    @property
    def random(self) -> random.Random:
        return self._random

    @property
    def faker(self) -> Faker:
        return self._faker

    def reseed(self, salt: int) -> None:
        combined_seed = (self.seed * 37 + salt) % (2**32)
        self._random.seed(combined_seed)
        self._faker.seed_instance(combined_seed)

    def choice(self, seq: Sequence[Any]) -> Any:
        return self._random.choice(list(seq))

    def weighted_choice(self, choices: Sequence[Tuple[Any, float]]) -> Any:
        total = sum(weight for _, weight in choices)
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        threshold = self._random.uniform(0, total)
        cumulative = 0.0
        for item, weight in choices:
            cumulative += weight
            if threshold <= cumulative:
                return item
        return choices[-1][0]

    def sample(self, seq: Sequence[Any], k: int) -> List[Any]:
        seq_list = list(seq)
        if k >= len(seq_list):
            return seq_list
        return self._random.sample(seq_list, k)

    def boolean(self, probability: float) -> bool:
        return self._random.random() < probability

    def randint(self, a: int, b: int) -> int:
        return self._random.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._random.uniform(a, b)

    def shuffle(self, seq: List[Any]) -> None:
        self._random.shuffle(seq)


def iter_proposal_dirs(dataset_root: Optional[Path] = None) -> Iterable[Tuple[str, Path]]:
    """Yield (split, proposal_dir) pairs for each proposal currently on disk."""
    root = dataset_root or DATASET_ROOT
    if not root.exists():
        return []
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for proposal_dir in sorted(split_dir.iterdir()):
            if proposal_dir.is_dir():
                yield split, proposal_dir


def safe_copy_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a deepcopy for YAML-derived dictionaries."""
    return copy.deepcopy(payload)
