from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & External Services
    GEMINI_API_KEY: str = ""
    GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
    GEMINI_GEN_MODEL: str = "gemini-3.1-flash-lite"

    # Path Settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CHROMA_DB_PATH: str = "./data/chroma_db"
    DATA_SOURCE_PATH: str = "./data/raw"

    # Retrieval Settings
    # RETRIEVAL_CONFIDENCE_THRESHOLD: empirically derived for the FastAPI docs corpus
    # using RRF fusion with rrf_k=60.  Real-match scores observed: 0.031–0.033 (both
    # dense + sparse systems agreed at rank 1).  Noise scores observed: 0.016 (only one
    # system matched at rank 1).  0.025 sits cleanly between these two clusters.
    # ⚠️  This value MUST be re-derived if: (a) the corpus changes, (b) rrf_k changes,
    # or (c) the retrieval depth / top_k parameters change significantly.  See
    # scripts/validate_confidence_threshold.py for the evaluation harness.
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.025

    # Courtesy sleep between sequential _verify_one calls within a single
    # CitationVerifier.verify() pass.  gemini-2.5-flash free tier allows ~10 RPM;
    # top_k=5 could fire 5 rapid calls in <1s.  A 0.5 s gap keeps the burst rate
    # under 6 RPM with negligible latency impact (total added: 2 s for 5 citations).
    # Set to 0.0 in tests (injected via CitationVerifier constructor).
    VERIFICATION_INTER_CALL_SLEEP: float = 0.5

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    @property
    def absolute_chroma_path(self) -> Path:
        path = Path(self.CHROMA_DB_PATH)
        if not path.is_absolute():
            return self.BASE_DIR / path
        return path

    @property
    def absolute_data_source_path(self) -> Path:
        path = Path(self.DATA_SOURCE_PATH)
        if not path.is_absolute():
            return self.BASE_DIR / path
        return path

    @property
    def gemini_api_keys(self) -> list[str]:
        keys = [k.strip() for k in self.GEMINI_API_KEY.split(",") if k.strip()]
        if not keys:
            raise ValueError("GEMINI_API_KEY is not set in configuration.")
        return keys



settings = Settings()
