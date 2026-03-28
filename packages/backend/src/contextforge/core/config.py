from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Bu dosyanın bulunduğu yerden 5 seviye yukarısı = workspace root
# config.py → core → contextforge → src → backend → packages → ROOT
_ROOT = Path(__file__).resolve().parents[5]


class Settings(BaseSettings):

    # Pydantic bu class'ı görünce şunu yapar:
    # 1. .env dosyasını oku
    # 2. Ortam değişkenlerine bak
    # 3. Bulamazsa default değeri kullan
    # Sıralama bu - .env, ortam değişkenini ezmez, tam tersi

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env2de fazladan key varsa hata verme
    )

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "contextforge"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # App
    app_env: str = "development"
    log_level: str = "INFO"


# Modül yüklendiğinde tek instance oluşturulur.
# Her yerden `from contextforge.core.config import settings` ile erişilir.
# Bu "singleton via module" pattern — Python'da en temiz DI yöntemi.
settings = Settings()
