# DevDocs RAG Framework

**A Python RAG framework for Developer Docs QA and vibecoding**

[中文版](./README_zh_CN.md)

An MCP (Model Context Protocol) framework inspired by [Stripe MCP](https://docs.stripe.com/mcp), providing AI Agents with intelligent retrieval and Q&A capabilities over any developer documentation library. Configure for RAGFlow, LangChain, FastAPI, or any project by setting a few environment variables.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP Client (Cursor / VS Code / Claude)    │
└──────────────────────┬───────────────────────────────────────┘
                       │ Streamable HTTP (MCP Protocol)
┌──────────────────────▼───────────────────────────────────────┐
│                    MCP Server (mcp_server.py)                │
│  Tools (names derived from PROJECT_NAME):                    │
│    • search_{slug}_docs     — hybrid document retrieval      │
│    • ask_{slug}_docs        — RAG Q&A                        │
│    • list_api_endpoints     — list API endpoints             │
│    • lookup_api_endpoint    — look up a specific API endpoint│
│    • agentic_search_{slug}_docs — multi-step agentic search  │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Retriever       Generator      Embedder
   (hybrid)        (Qwen3.5+)    (embedding-v4)
        │              │              │
        ▼              │              │
   PostgreSQL ◄────────┘──────────────┘
   (pgvector + FTS)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| MCP Server | `mcp` Python SDK (Streamable HTTP transport) |
| Vector Database | PostgreSQL + pgvector (cosine similarity) |
| Full-Text Search | PostgreSQL tsvector (weighted A/B/C) |
| Embedding Model | Alibaba Cloud DashScope `text-embedding-v4` (1024-dim) |
| Generation Model | Alibaba Cloud DashScope `qwen3.5-plus` (Qwen3.5-Plus) |
| Document Chunking | Custom Markdown hierarchical chunking (H2→H3→H4) |
| Retrieval Strategy | Vector + full-text hybrid retrieval (weighted fusion) |

## Documentation Sources

By default, documentation is downloaded from the `website/docs/references/` directory of [infiniflow/ragflow-docs](https://github.com/infiniflow/ragflow-docs):

- `http_api_reference.md` — Complete HTTP API reference
- `python_api_reference.md` — Complete Python SDK reference
- `glossary.mdx` — Glossary

You can point the framework at any documentation library by setting `GITHUB_RAW_BASE` and `DOC_FILES` in your `.env` — see [Supporting Different Documentation Libraries](#supporting-different-documentation-libraries) below.

## Quick Start

### Docker (Recommended)

The easiest way to get started is with Docker Compose, which sets up both the application and a PostgreSQL database with the pgvector extension automatically.

```bash
# 1. Copy and configure the environment file
cp .env.example .env
# Edit .env and set your DASHSCOPE_API_KEY

# 2. Start services (PostgreSQL + MCP server)
docker compose up -d

# 3. Index the documentation
docker compose exec app python cli.py index

# 4. Verify everything is running
docker compose exec app python cli.py status
```

The MCP server is then available at `http://localhost:8000/mcp`.

To run other CLI commands inside the container:
```bash
docker compose exec app python cli.py search "how to create a dataset"
docker compose exec app python cli.py ask "How do I upload documents using the Python SDK?"
```

To stop the services:
```bash
docker compose down
```

---

### Manual Setup

### 1. Environment Setup

```bash
# Requires uv (https://docs.astral.sh/uv/) and Python 3.11+
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. PostgreSQL Setup

```bash
# Install pgvector extension (if not already installed)
# Ubuntu: sudo apt install postgresql-16-pgvector
# macOS: brew install pgvector

# Create database
createdb ragflow_docs

# Enable pgvector extension
psql ragflow_docs -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Or run the full setup SQL
psql ragflow_docs -f setup_db.sql
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env and fill in your DashScope API Key and database connection info
```

`.env` example:
```env
# Project identity (change these for a different documentation library)
PROJECT_NAME=RAGFlow
PROJECT_DESCRIPTION=RAGFlow API documentation (HTTP API and Python SDK references)

DASHSCOPE_API_KEY=sk-your-dashscope-api-key
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=ragflow_docs
```

### 4. Index Documents

```bash
# Download documents and build index
uv run python cli.py index

# Force re-download and re-index
uv run python cli.py index --force-download --force-reindex
```

### 5. Test

```bash
# Check index status
uv run python cli.py status

# Search documents
uv run python cli.py search "how to create a dataset"

# Interactive search
uv run python cli.py search

# RAG Q&A
uv run python cli.py ask "How do I upload documents to RAGFlow using Python SDK?"
```

### 6. Start MCP Server

```bash
uv run python cli.py serve --host 127.0.0.1 --port 8000 --path /mcp
```

## MCP Client Configuration

### Cursor

First, start the server:
```bash
uv run python cli.py serve --host 127.0.0.1 --port 8000 --path /mcp
```

Then add in Cursor Settings → MCP:
```json
{
  "mcpServers": {
    "ragflow-docs": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### VS Code (GitHub Copilot)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "ragflow-docs": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ragflow-docs": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

## MCP Tools

> **Note:** Tool names are generated dynamically based on `PROJECT_NAME`. The examples below use the default `PROJECT_NAME=RAGFlow`, which produces the slug `ragflow`. If you set `PROJECT_NAME=LangChain`, the tools become `search_langchain_docs`, `ask_langchain_docs`, etc.

### search_{slug}_docs

Search project documentation with hybrid retrieval (vector + full-text).
*(Default: `search_ragflow_docs`)*

**Parameters:**
- `query` (required): Search query
- `top_k`: Number of results to return (default 5)
- `search_mode`: `hybrid` | `vector` | `fts` (default `hybrid`)
- `doc_filter`: Restrict search to a specific document

**Example:**
```
Search for "create dataset API endpoint"
```

### ask_{slug}_docs

Document-grounded RAG Q&A — returns an AI-generated answer with cited sources.
*(Default: `ask_ragflow_docs`)*

**Parameters:**
- `question` (required): A question about the project
- `top_k`: Number of context chunks (default 6)

**Example:**
```
Ask "How do I configure a chat assistant with custom retrieval settings?"
```

### list_api_endpoints

List all API endpoints grouped by category.

**Parameters:**
- `category` (optional): Filter by category (dataset, document, chunk, chat, etc.)

### lookup_api_endpoint

Look up detailed documentation for a specific API endpoint.

**Parameters:**
- `url_pattern` (required): URL match pattern
- `method` (optional): HTTP method (GET/POST/PUT/DELETE)

### agentic_search_{slug}_docs

Perform an agentic search over the developer documentation. Unlike simple search, this tool automatically decomposes complex questions into sub-queries, performs multiple rounds of retrieval, evaluates whether enough context has been gathered, and synthesizes a comprehensive answer. Best for complex, multi-faceted questions that span multiple API endpoints, SDK methods, or concepts.
*(Default: `agentic_search_ragflow_docs`)*

**Parameters:**
- `question` (required): A complex question about the project
- `max_rounds`: Maximum number of search iterations (default 3)
- `top_k_per_query`: Number of results per sub-query per round (default 5)

**Example:**
```
Agentic search "What is the full workflow for building a RAG pipeline — from creating a dataset, uploading documents, to setting up a chat assistant with retrieval?"
```

## Supporting Different Documentation Libraries

DevDocs RAG is framework-agnostic. To index a different project's documentation, set the following environment variables in your `.env`:

### Example: LangChain

```env
PROJECT_NAME=LangChain
PROJECT_DESCRIPTION=LangChain Python SDK reference and guides
SDK_CLASS_NAMES=["ChatOpenAI","LLMChain","PromptTemplate","VectorStore","Runnable"]  # JSON array
GITHUB_RAW_BASE=https://raw.githubusercontent.com/langchain-ai/langchain/master/docs
DOC_FILES=["docs/api_reference.md"]
POSTGRES_DB=langchain_docs
```

### Example: FastAPI

```env
PROJECT_NAME=FastAPI
PROJECT_DESCRIPTION=FastAPI web framework reference documentation
SDK_CLASS_NAMES=["FastAPI","APIRouter","Depends","HTTPException","Request","Response"]  # JSON array
GITHUB_RAW_BASE=https://raw.githubusercontent.com/tiangolo/fastapi/master
DOC_FILES=["docs/en/docs/reference/apirouter.md","docs/en/docs/reference/fastapi.md"]
POSTGRES_DB=fastapi_docs
```

After changing these values, re-run the indexing step:
```bash
uv run python cli.py index --force-download --force-reindex
```

MCP tool names update automatically — for example, with `PROJECT_NAME=LangChain` the tools become `search_langchain_docs`, `ask_langchain_docs`, and `agentic_search_langchain_docs`.

## Project Structure

```
agentic-ragflow-dev-docs/
├── cli.py                  # CLI entry (index / serve / search / ask / agentic-search / status)
├── pyproject.toml          # Project metadata & dependencies (package: devdocs-rag)
├── requirements.txt        # Python dependencies (pip-compatible)
├── setup_db.sql            # Database initialization SQL
├── .env.example            # Environment variable template
├── docs/                   # Downloaded documents (auto-created)
│   ├── http_api_reference.md
│   ├── python_api_reference.md
│   └── glossary.mdx
└── devdocs_rag/
    ├── __init__.py
    ├── config.py            # Pydantic Settings configuration
    ├── downloader.py        # Download documents from GitHub
    ├── chunker.py           # Custom Markdown chunking strategy
    ├── embedder.py          # Qwen text-embedding-v4
    ├── db.py                # PostgreSQL + pgvector data layer
    ├── retriever.py         # Hybrid retrieval engine
    ├── generator.py         # Qwen3.5-Plus RAG generation
    ├── agentic_search.py    # Multi-step agentic search engine
    ├── metadata_enricher.py # Metadata enrichment for chunks
    ├── indexer.py           # Indexing pipeline
    └── mcp_server.py        # MCP protocol server
```

## Chunking Strategy

Chunking optimized for developer API documentation:

1. **Hierarchical parsing**: Build a document structure tree by Markdown headings (H2 → H3 → H4)
2. **API endpoint detection**: Automatically extract HTTP Method + URL and SDK method signatures
3. **Smart grouping**: Merge Request + Parameters together; separate Response and Examples
4. **Code block protection**: Code blocks are never truncated
5. **Metadata enrichment**: Each chunk includes structured metadata such as `section_path`, `chunk_type`, and `api_method`

## Hybrid Retrieval

```
Final Score = α × vector_score + β × fts_score

Default weights: α = 0.6, β = 0.4
```

- **Vector retrieval**: Semantic search based on cosine similarity
- **Full-text retrieval**: PostgreSQL tsvector weighted matching (API identifiers ranked highest)
- **Result fusion**: FULL OUTER JOIN merges results from both retrieval methods

## Memo

- See [docs/memo.md](./docs/memo.md) for issue notes and postmortems.

## License

MIT
