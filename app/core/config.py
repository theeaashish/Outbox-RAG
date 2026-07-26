from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Applications configuration loaded from environment variables"""

    # application

    app_name: str = Field(default="Basic RAG")
    app_env: str = Field(default="development")
    app_debug: bool = Field(default=True)

    app_host: str = Field(default="127.0.0.1")
    app_port: int = Field(default=8000)

    api_v1_prefix: str = Field(default="/api/v1")

    # database

    database_url: str

    # gemini

    google_api_key: str

    gemini_chat_model: str = Field(default="gemini-2.5-flash")

    gemini_embedding_model: str = Field(default="gemini-embedding-001")

    # upload

    upload_directory: str = Field(default="uploads")

    max_upload_size_mb: int = Field(default=20)

    # retrieval

    default_top_k: int = Field(default=5)

    similarity_threshold: float = Field(default=0.7)

    # logging

    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance"""
    return Settings()


settings = get_settings()
