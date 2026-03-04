# RAGFlow Developer Docs MCP App

一个受 [Stripe MCP](https://docs.stripe.com/mcp) 启发的 MCP（Model Context Protocol）应用，为 AI Agent 提供 RAGFlow 开发者文档的智能检索和问答能力。

## 架构概述

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP Client (Cursor / VS Code / Claude)    │
└──────────────────────┬───────────────────────────────────────┘
                       │ Streamable HTTP (MCP Protocol)
┌──────────────────────▼───────────────────────────────────────┐
│                    MCP Server (mcp_server.py)                │
│  Tools:                                                      │
│    • search_ragflow_docs    — 混合检索文档                    │
│    • ask_ragflow_docs       — RAG 问答                       │
│    • list_api_endpoints     — 列出 API 端点                  │
│    • lookup_api_endpoint    — 查询特定 API 端点              │
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

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| MCP Server | `mcp` Python SDK (Streamable HTTP transport) |
| 向量数据库 | PostgreSQL + pgvector (cosine similarity) |
| 全文搜索 | PostgreSQL tsvector (加权 A/B/C) |
| Embedding 模型 | 阿里云 DashScope `text-embedding-v4` (1024 维) |
| 生成模型 | 阿里云 DashScope `qwen3.5-plus` (Qwen3.5-Plus) |
| 文档分块 | 自定义 Markdown 层级分块 (H2→H3→H4) |
| 检索策略 | 向量 + 全文混合检索 (加权融合) |

## 文档来源

从 [infiniflow/ragflow-docs](https://github.com/infiniflow/ragflow-docs) 的 `website/docs/references/` 目录下载：

- `http_api_reference.md` — HTTP API 完整参考
- `python_api_reference.md` — Python SDK 完整参考
- `glossary.mdx` — 术语表

## 快速开始

### 1. 环境准备

```bash
# 需要 uv (https://docs.astral.sh/uv/) 和 Python 3.11+
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. PostgreSQL 设置

```bash
# 安装 pgvector 扩展（如果尚未安装）
# Ubuntu: sudo apt install postgresql-16-pgvector
# macOS: brew install pgvector

# 创建数据库
createdb ragflow_docs

# 启用 pgvector 扩展
psql ragflow_docs -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 或者运行完整的 setup SQL
psql ragflow_docs -f setup_db.sql
```

### 3. 配置

```bash
cp .env.example .env
# 编辑 .env，填入你的 DashScope API Key 和数据库连接信息
```

`.env` 示例：
```env
DASHSCOPE_API_KEY=sk-your-dashscope-api-key
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=ragflow_docs
```

### 4. 索引文档

```bash
# 下载文档并建立索引
uv run python cli.py index

# 强制重新下载和索引
uv run python cli.py index --force-download --force-reindex
```

### 5. 测试

```bash
# 查看索引状态
uv run python cli.py status

# 搜索文档
uv run python cli.py search "how to create a dataset"

# 交互式搜索
uv run python cli.py search

# RAG 问答
uv run python cli.py ask "How do I upload documents to RAGFlow using Python SDK?"
```

### 6. 启动 MCP Server

```bash
uv run python cli.py serve --host 127.0.0.1 --port 8000 --path /mcp
```

## MCP 客户端配置

### Cursor

在 Cursor Settings → MCP 中添加：

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

在 `.vscode/mcp.json` 中添加：

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

在 `claude_desktop_config.json` 中添加：

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

搜索 RAGFlow 文档，支持混合检索（向量 + 全文）。

**参数：**
- `query` (必需): 搜索查询
- `top_k`: 返回结果数量 (默认 5)
- `search_mode`: `hybrid` | `vector` | `fts` (默认 `hybrid`)
- `doc_filter`: 限制搜索范围到特定文档

**示例：**
```
Search for "create dataset API endpoint"
```

### ask_ragflow_docs

基于文档的 RAG 问答，返回 AI 生成的答案和引用来源。

**参数：**
- `question` (必需): 关于 RAGFlow 的问题
- `top_k`: 上下文块数量 (默认 6)

**示例：**
```
Ask "How do I configure a chat assistant with custom retrieval settings?"
```

### list_api_endpoints

列出所有 RAGFlow API 端点，按类别分组。

**参数：**
- `category` (可选): 过滤类别 (dataset, document, chunk, chat 等)

### lookup_api_endpoint

查找特定 API 端点的详细文档。

**参数：**
- `url_pattern` (必需): URL 匹配模式
- `method` (可选): HTTP 方法 (GET/POST/PUT/DELETE)

## 项目结构

```
agentic-ragflow-dev-docs/
├── cli.py                  # CLI 入口 (index / serve / search / ask / status)
├── pyproject.toml          # 项目元数据 & 依赖 (uv)
├── requirements.txt        # Python 依赖 (兼容 pip)
├── setup_db.sql            # 数据库初始化 SQL
├── .env.example            # 环境变量模板
├── docs/                   # 下载的文档 (自动创建)
│   ├── http_api_reference.md
│   ├── python_api_reference.md
│   └── glossary.mdx
└── src/
    ├── __init__.py
    ├── config.py            # Pydantic Settings 配置
    ├── downloader.py        # 从 GitHub 下载文档
    ├── chunker.py           # 自定义 Markdown 分块策略
    ├── embedder.py          # Qwen text-embedding-v4
    ├── db.py                # PostgreSQL + pgvector 数据层
    ├── retriever.py         # 混合检索引擎
    ├── generator.py         # Qwen3.5-Plus RAG 生成
    ├── indexer.py           # 索引 Pipeline
    └── mcp_server.py        # MCP 协议服务器
```

## 分块策略

针对开发者 API 文档优化的分块方式：

1. **层级解析**: 按 Markdown 标题 (H2 → H3 → H4) 建立文档结构树
2. **API 端点识别**: 自动提取 HTTP Method + URL 和 SDK 方法签名
3. **智能分组**: 将 Request + Parameters 合并，Response 和 Examples 分开
4. **代码块保护**: 代码块不会被截断
5. **元数据增强**: 每个 chunk 包含 section_path、chunk_type、api_method 等结构化信息

## 混合检索

```
Final Score = α × vector_score + β × fts_score

默认权重: α = 0.6, β = 0.4
```

- **向量检索**: 基于 cosine similarity 的语义搜索
- **全文检索**: PostgreSQL tsvector 加权匹配 (API 标识符权重最高)
- **结果融合**: FULL OUTER JOIN 合并两种检索结果

## License

MIT
