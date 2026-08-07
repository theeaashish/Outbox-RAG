from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Applications configuration loaded from environment variables"""

    # application

    app_name: str = Field(default="Basic RAG")
    app_env: Environment = Environment.DEVELOPMENT
    app_debug: bool = Field(default=True)

    app_host: str = Field(default="127.0.0.1")
    app_port: int = Field(default=8000)

    api_v1_prefix: str = Field(default="/api/v1")

    # database

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/basic_rag"
    )

    # gemini

    google_api_key: str = Field(default="")

    gemini_chat_model: str = Field(default="gemini-2.5-flash")

    gemini_embedding_model: str = Field(default="gemini-embedding-001")

    gemini_chat_temperature: float = Field(default=0.7)

    gemini_chat_max_output_tokens: int = Field(default=4_096, ge=1)

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

    # Rag settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    top_k_chunks: int = 5

    # chat

    chat_history_message_limit: int = Field(default=20, ge=1)

    chat_max_message_characters: int = Field(default=10_000, ge=1)

    chat_stream_first_token_timeout_seconds: int = Field(default=20, ge=1)

    chat_stream_idle_timeout_seconds: int = Field(default=30, ge=1)

    chat_stream_total_timeout_seconds: int = Field(default=120, ge=1)

    chat_stream_max_buffered_characters: int = Field(default=65_536, ge=1)

    chat_stream_ping_interval_seconds: int = Field(default=15, ge=1)

    api_v2_prefix: str = Field(default="/api/v2")

    cursor_signing_key: str = Field(default="development-only-change-me")

    cursor_previous_signing_key: str | None = Field(default=None)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance"""
    return Settings()


settings = get_settings()
