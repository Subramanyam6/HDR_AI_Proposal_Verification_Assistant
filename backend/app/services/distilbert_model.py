"""DistilBERT transformer model service."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
from ..config import settings


class DistilBERTModelService:
    """DistilBERT model loading and inference."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.labels: List[str] = ["crosswalk_error", "banned_phrases", "name_inconsistency"]
        self.label_display: Dict[str, str] = {
            "crosswalk_error": "Crosswalk Error",
            "banned_phrases": "Banned Phrases",
            "name_inconsistency": "Name Inconsistency",
        }
        self._loaded = False

    def load(self):
        """Load DistilBERT model from disk."""
        try:
            model_dir = settings.distilbert_model_dir

            print("Loading DistilBERT model...")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

            # Set device (CPU for compatibility)
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            print("✓ DistilBERT model loaded successfully")
        except FileNotFoundError as e:
            print(f"⚠ DistilBERT model files not found: {e}")
            print("  Please download model files from HuggingFace Spaces")
            self._loaded = False
        except Exception as e:
            print(f"✗ Failed to load DistilBERT model: {e}")
            self._loaded = False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def predict(self, text: str) -> Dict[str, bool]:
        """
        Run DistilBERT model inference.

        Args:
            text: Input proposal text

        Returns:
            Dictionary mapping label names to boolean predictions
        """
        if not self._loaded:
            return {self.label_display.get(label, label): False for label in self.labels}

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Run inference
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Apply sigmoid and threshold
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            predictions = (probabilities > settings.transformer_threshold).astype(int)

            # Build results
            results = {}
            for idx, label in enumerate(self.labels):
                display_label = self.label_display.get(label, label)
                results[display_label] = bool(predictions[idx])

            return results

        except Exception as e:
            print(f"✗ DistilBERT prediction failed: {e}")
            return {self.label_display.get(label, label): False for label in self.labels}


# Global instance
distilbert_service = DistilBERTModelService()
