"""
DevDocs RAG Framework - Configuration

A Python RAG framework for Developer Docs QA and vibecoding scenarios.
Supports any documentation library via project-level configuration.
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Project Identity ─────────────────────────────────────────────────
    project_name: str = Field(
        default="RAGFlow",
        alias="PROJECT_NAME",
        description="Name of the project whose docs are being indexed (e.g. 'RAGFlow', 'LangChain', 'FastAPI').",
    )
    project_description: str = Field(
        default="RAGFlow API documentation (HTTP API and Python SDK references)",
        alias="PROJECT_DESCRIPTION",
        description="Brief description used in prompts and tool descriptions.",
    )
    sdk_class_names: list[str] = Field(
        default=[
            "RAGFlow", "DataSet", "Dataset", "Document", "Chunk",
            "Chat", "Session", "Agent", "Memory", "Ragflow",
        ],
        alias="SDK_CLASS_NAMES",
        description="SDK class names to detect in documentation for signature extraction.",
    )

    # ── LLM Provider ─────────────────────────────────────────────────────
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )
    embedding_model: str = Field(default="text-embedding-v4", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1024, alias="EMBEDDING_DIMENSIONS")
    chat_model: str = Field(default="qwen-plus", alias="CHAT_MODEL")
    light_model: str = Field(
        default="qwen-flash",
        alias="LIGHT_MODEL",
        description="Fast small model for decompose / evaluate / metadata / pre-filter tasks",
    )
    enable_thinking: bool = Field(
        default=False,
        alias="ENABLE_THINKING",
        description="Whether to enable Qwen3.5 thinking mode (adds latency)",
    )

    # ── PostgreSQL ───────────────────────────────────────────────────────
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="ragflow_docs", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1024, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, alias="CHUNK_OVERLAP")

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k: int = Field(default=8, alias="TOP_K")
    vector_weight: float = Field(default=0.6, alias="VECTOR_WEIGHT")
    fts_weight: float = Field(default=0.4, alias="FTS_WEIGHT")
    similarity_threshold: float = Field(default=0.3, alias="SIMILARITY_THRESHOLD")

    # ── Agentic search ───────────────────────────────────────────────────
    agentic_max_rounds: int = Field(default=3, alias="AGENTIC_MAX_ROUNDS")

    # ── Docs source ──────────────────────────────────────────────────────
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

    # ── Derived helpers ──────────────────────────────────────────────────
    @property
    def project_slug(self) -> str:
        """Lowercase, hyphen-separated project name for identifiers."""
        return self.project_name.lower().replace(" ", "-")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True}


def get_settings() -> Settings:
    return Settings()
