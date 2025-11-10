"""Naive Bayes baseline model service (optional)."""
import re
import pickle
from pathlib import Path
from typing import Dict, List, Any
from ..config import settings


NB_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class NaiveBayesModelService:
    """Naive Bayes model loading and inference."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.labels: List[str] = ["crosswalk_error", "banned_phrases", "name_inconsistency"]
        self.label_display: Dict[str, str] = {
            "crosswalk_error": "Crosswalk Error",
            "banned_phrases": "Banned Phrases",
            "name_inconsistency": "Name Inconsistency",
        }
        self._loaded = False

    def load(self):
        """Load Naive Bayes models from disk (if available)."""
        try:
            model_dir = settings.nb_model_dir

            if not model_dir.exists():
                print("⚠ Naive Bayes models directory not found (optional)")
                self._loaded = False
                return

            # Try loading each label's model
            for label in self.labels:
                model_path = model_dir / f"{label}.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self.models[label] = pickle.load(f)

            if self.models:
                self._loaded = True
                print(f"✓ Naive Bayes models loaded ({len(self.models)} labels)")
            else:
                print("⚠ No Naive Bayes model files found (optional)")
                self._loaded = False

        except Exception as e:
            print(f"⚠ Failed to load Naive Bayes models (optional): {e}")
            self._loaded = False

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for Naive Bayes model."""
        tokens = [token.lower() for token in NB_TOKEN_RE.findall(text)]
        normalized: List[str] = []

        pm_tokens = {"sarah", "martinez", "lee", "davis", "jennifer", "thompson", "kevin", "vazquez", "lisa", "dana"}

        for tok in tokens:
            if tok.startswith("requirement"):
                normalized.append("requirement_token")
            elif tok in pm_tokens:
                normalized.append("pm_name_token")
            else:
                normalized.append(tok)

        return normalized

    def predict(self, text: str) -> Dict[str, bool]:
        """
        Run Naive Bayes model inference.

        Args:
            text: Input proposal text

        Returns:
            Dictionary mapping label names to boolean predictions
        """
        if not self._loaded or not self.models:
            return {self.label_display.get(label, label): False for label in self.labels}

        try:
            tokens = self._tokenize(text)

            # Convert token list to bag-of-words dictionary for river-ml
            token_counts: Dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            results: Dict[str, bool] = {}

            for label in self.labels:
                display_label = self.label_display.get(label, label)
                model = self.models.get(label)

                if model is None:
                    results[display_label] = False
                else:
                    # River-ml predict_one expects dictionary of features
                    try:
                        results[display_label] = bool(model.predict_one(token_counts))
                    except Exception as ex:
                        # Fallback if predict_one fails
                        print(f"⚠ NB model prediction failed for {label}: {ex}")
                        results[display_label] = False

            return results

        except Exception as e:
            print(f"✗ Naive Bayes prediction failed: {e}")
            return {self.label_display.get(label, label): False for label in self.labels}


# Global instance
nb_service = NaiveBayesModelService()
