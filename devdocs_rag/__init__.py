"""
DevDocs RAG — A Python RAG framework for Developer Docs QA and vibecoding.

Designed to make developer documentation intelligently searchable via
hybrid RAG (vector + full-text search), agentic multi-step retrieval,
and MCP tool serving.

Quick Start:
    # Configure via environment variables:
    #   PROJECT_NAME=MyProject
    #   GITHUB_RAW_BASE=https://raw.githubusercontent.com/org/repo/main/docs
    #   DOC_FILES=["api.md", "sdk.md"]
    #   DASHSCOPE_API_KEY=sk-xxx

    from devdocs_rag.config import get_settings
    from devdocs_rag.db import Database
    from devdocs_rag.embedder import Embedder
    from devdocs_rag.retriever import Retriever
    from devdocs_rag.generator import Generator
    from devdocs_rag.agentic_search import AgenticSearch
    from devdocs_rag.indexer import run_indexing_pipeline
    from devdocs_rag.mcp_server import main as serve_mcp

Supports any documentation library by setting PROJECT_NAME and related
environment variables.
"""

__version__ = "0.2.0"
