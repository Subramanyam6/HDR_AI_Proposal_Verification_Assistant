from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env.development if it exists, otherwise .env
env = os.getenv("ENVIRONMENT", "development")
project_root = Path(__file__).resolve().parent.parent.parent

# Try .env.development first, then .env
env_file = project_root / f".env.{env}"
if not env_file.exists():
    env_file = project_root / ".env"

if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded environment from {env_file}")


def parse_cors_origins(v):
    """Parse CORS origins from string or list"""
    if isinstance(v, str):
        if v.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]
    elif isinstance(v, list):
        return v
    return ["*"]


class Settings(BaseSettings):
    # Environment
    environment: str = "development"

    # API Keys
    openai_api_key: str = "sk-dummy-key-replace-me"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent
    models_dir: Path = base_dir / "models"
    samples_dir: Path = base_dir / "samples"
    tfidf_model_dir: Path = models_dir
    distilbert_model_dir: Path = models_dir / "distilbert"
    nb_model_dir: Path = models_dir / "nb_baseline"

    # Model thresholds
    transformer_threshold: float = 0.50
    tfidf_threshold: float = 0.50

    # CORS - store as string, parse in main.py
    cors_origins_str: str = "*"

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None,  # We load it manually above
    )

    @property
    def cors_origins(self) -> list[str]:
        """Parse CORS origins as needed"""
        return parse_cors_origins(self.cors_origins_str)


settings = Settings()
