from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://lexintel:lexintel-local@127.0.0.1/lexintel"
    )
    storage: Path = Path(os.getenv("LEXINTEL_STORAGE", "/var/lib/lexintel/files"))
    max_upload_bytes: int = 50 * 1024 * 1024
    max_upload_files: int = 20
    max_pdf_pages: int = 250
    parser_timeout: int = 600
    lease_seconds: int = 90
    max_attempts: int = 3
    ocr_languages: str = os.getenv("LEXINTEL_OCR_LANGUAGES", "eng+hin")
    ranker_url: str = os.getenv("LEXINTEL_RANKER_URL", "")


def embedding_key() -> str:
    if os.getenv("LEXINTEL_PAID_INDEXING") != "1":
        return ""
    key_file = os.getenv("OPENROUTER_API_KEY_FILE")
    if key_file:
        return Path(key_file).read_text().strip()
    return os.getenv("OPENROUTER_API_KEY", "").strip()


settings = Settings()
