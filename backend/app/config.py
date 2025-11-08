from pydantic_settings import BaseSettings
from pathlib import Path


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

    # CORS
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
