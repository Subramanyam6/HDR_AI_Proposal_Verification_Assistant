"""TF-IDF + Logistic Regression model service."""
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional
from ..config import settings


class TFIDFModelService:
    """TF-IDF model loading and inference."""

    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.labels: List[str] = []
        self.label_display: Dict[str, str] = {
            "crosswalk_error": "Crosswalk Error",
            "banned_phrases": "Banned Phrases",
            "name_inconsistency": "Name Inconsistency",
        }
        self._loaded = False

    def load(self):
        """Load TF-IDF model from disk."""
        try:
            model_dir = settings.tfidf_model_dir

            # Load vectorizer
            vectorizer_path = model_dir / "vectorizer.pkl"
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)

            # Load classifier
            classifier_path = model_dir / "classifier.pkl"
            with open(classifier_path, "rb") as f:
                self.classifier = pickle.load(f)

            # Load config
            config_path = model_dir / "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.labels = config.get("labels", [])

            self._loaded = True
            print("✓ TF-IDF model loaded successfully")
        except FileNotFoundError as e:
            print(f"⚠ TF-IDF model files not found: {e}")
            print("  Please download model files from HuggingFace Spaces")
            self._loaded = False
        except Exception as e:
            print(f"✗ Failed to load TF-IDF model: {e}")
            self._loaded = False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def predict(self, text: str) -> Dict[str, bool]:
        """
        Run TF-IDF model inference.

        Args:
            text: Input proposal text

        Returns:
            Dictionary mapping label names to boolean predictions
        """
        if not self._loaded:
            return {self.label_display.get(label, label): False for label in self.labels}

        try:
            # Vectorize input
            features = self.vectorizer.transform([text])

            # Get probabilities
            probabilities = self.classifier.predict_proba(features)

            # Apply threshold
            results = {}
            for idx, label in enumerate(self.labels):
                display_label = self.label_display.get(label, label)
                results[display_label] = bool(probabilities[0][idx] >= settings.tfidf_threshold)

            return results

        except Exception as e:
            print(f"✗ TF-IDF prediction failed: {e}")
            return {self.label_display.get(label, label): False for label in self.labels}


# Global instance
tfidf_service = TFIDFModelService()
