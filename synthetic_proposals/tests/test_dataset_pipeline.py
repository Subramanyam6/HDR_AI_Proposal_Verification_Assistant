import importlib.util
from pathlib import Path

import numpy as np
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "09_transformer_mps_verified.py"
spec = importlib.util.spec_from_file_location("transformer_trainer", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

MODEL_DIR = Path(__file__).resolve().parents[2] / "backend" / "app" / "models" / "distilbert"


def test_dataset_splits_present():
    data = module.load_dataset(module.DATA_PATH)
    splits = {rec.get("split") for rec in data}
    assert {"train", "dev", "test"}.issubset(splits)
    split_counts = {split: 0 for split in splits}
    for rec in data:
        split_counts[rec.get("split")] += 1
    assert split_counts["train"] > 0
    assert split_counts["dev"] > 0
    assert split_counts["test"] > 0


def test_tokenization_shapes():
    data = module.load_dataset(module.DATA_PATH)
    sample_records = [rec for rec in data if rec.get("split") == "train"][:4]
    tokenizer = module.AutoTokenizer.from_pretrained(MODEL_DIR)
    dataset = module.ProposalDataset(sample_records, tokenizer, max_length=128)
    item = dataset[0]
    assert set(item.keys()) == {"text", "labels"}
    assert isinstance(item["text"], str)
    assert item["labels"].shape == (len(module.TARGETS),)

    def collate_fn(batch):
        texts = [example["text"] for example in batch]
        encodings = tokenizer(texts, padding=True, return_tensors="pt")
        encodings["labels"] = torch.stack([example["labels"] for example in batch])
        return encodings

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["input_ids"].ndim == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == (2, len(module.TARGETS))
    labels_np = batch["labels"].numpy()
    assert np.all((labels_np == 0) | (labels_np == 1))
