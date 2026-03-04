# RAGFlow Developer Docs MCP App

[中文版](./README_zh_CN.md)

An MCP (Model Context Protocol) app inspired by [Stripe MCP](https://docs.stripe.com/mcp), providing AI Agents with intelligent retrieval and Q&A capabilities over RAGFlow developer documentation.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP Client (Cursor / VS Code / Claude)    │
└──────────────────────┬───────────────────────────────────────┘
                       │ Streamable HTTP (MCP Protocol)
┌──────────────────────▼───────────────────────────────────────┐
│                    MCP Server (mcp_server.py)                │
│  Tools:                                                      │
│    • search_ragflow_docs    — hybrid document retrieval      │
│    • ask_ragflow_docs       — RAG Q&A                        │
│    • list_api_endpoints     — list API endpoints             │
│    • lookup_api_endpoint    — look up a specific API endpoint│
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

Downloaded from the `website/docs/references/` directory of [infiniflow/ragflow-docs](https://github.com/infiniflow/ragflow-docs):

- `http_api_reference.md` — Complete HTTP API reference
- `python_api_reference.md` — Complete Python SDK reference
- `glossary.mdx` — Glossary

## Quick Start

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

### search_ragflow_docs

Search RAGFlow documentation with hybrid retrieval (vector + full-text).

**Parameters:**
- `query` (required): Search query
- `top_k`: Number of results to return (default 5)
- `search_mode`: `hybrid` | `vector` | `fts` (default `hybrid`)
- `doc_filter`: Restrict search to a specific document

**Example:**
```
Search for "create dataset API endpoint"
```

### ask_ragflow_docs

Document-grounded RAG Q&A — returns an AI-generated answer with cited sources.

**Parameters:**
- `question` (required): A question about RAGFlow
- `top_k`: Number of context chunks (default 6)

**Example:**
```
Ask "How do I configure a chat assistant with custom retrieval settings?"
```

### list_api_endpoints

List all RAGFlow API endpoints grouped by category.

**Parameters:**
- `category` (optional): Filter by category (dataset, document, chunk, chat, etc.)

### lookup_api_endpoint

Look up detailed documentation for a specific API endpoint.

**Parameters:**
- `url_pattern` (required): URL match pattern
- `method` (optional): HTTP method (GET/POST/PUT/DELETE)

## Project Structure

```
agentic-ragflow-dev-docs/
├── cli.py                  # CLI entry (index / serve / search / ask / status)
├── pyproject.toml          # Project metadata & dependencies (uv)
├── requirements.txt        # Python dependencies (pip-compatible)
├── setup_db.sql            # Database initialization SQL
├── .env.example            # Environment variable template
├── docs/                   # Downloaded documents (auto-created)
│   ├── http_api_reference.md
│   ├── python_api_reference.md
│   └── glossary.mdx
└── src/
    ├── __init__.py
    ├── config.py            # Pydantic Settings configuration
    ├── downloader.py        # Download documents from GitHub
    ├── chunker.py           # Custom Markdown chunking strategy
    ├── embedder.py          # Qwen text-embedding-v4
    ├── db.py                # PostgreSQL + pgvector data layer
    ├── retriever.py         # Hybrid retrieval engine
    ├── generator.py         # Qwen3.5-Plus RAG generation
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

## License

MIT
