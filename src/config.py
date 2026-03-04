"""
RAGFlow Developer Docs MCP Server - Configuration
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # DashScope / Qwen
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )
    embedding_model: str = Field(default="text-embedding-v4", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1024, alias="EMBEDDING_DIMENSIONS")
    chat_model: str = Field(default="qwen-plus", alias="CHAT_MODEL")

    # PostgreSQL
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="ragflow_docs", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")

    # Chunking
    chunk_size: int = Field(default=1024, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, alias="CHUNK_OVERLAP")

    # Retrieval
    top_k: int = Field(default=8, alias="TOP_K")
    vector_weight: float = Field(default=0.6, alias="VECTOR_WEIGHT")
    fts_weight: float = Field(default=0.4, alias="FTS_WEIGHT")
    similarity_threshold: float = Field(default=0.3, alias="SIMILARITY_THRESHOLD")

    # Docs source
    docs_dir: str = Field(default="docs", alias="DOCS_DIR")
    github_raw_base: str = Field(
        default="https://raw.githubusercontent.com/infiniflow/ragflow-docs/main/website/docs/references",
        alias="GITHUB_RAW_BASE",
    )
    doc_files: list[str] = Field(
        default=[
            "http_api_reference.md",
            "python_api_reference.md",
            "glossary.mdx",
        ],
        alias="DOC_FILES",
    )

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {"env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()
