"""
RAGFlow Developer Docs MCP Server

Provides tools for AI agents to search and query RAGFlow documentation,
inspired by Stripe's MCP approach (https://docs.stripe.com/mcp).

Tools:
- search_ragflow_docs: Search RAGFlow API documentation with hybrid retrieval
- ask_ragflow_docs: Ask questions about RAGFlow and get AI-generated answers
- list_api_endpoints: List all available API endpoints
- lookup_api_endpoint: Look up a specific API endpoint by URL pattern
- agentic_search_ragflow_docs: Multi-step agentic search with query decomposition
"""
import asyncio
import json
from contextlib import asynccontextmanager
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import (
    Tool,
    TextContent,
)
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

from src.config import get_settings
from src.db import Database
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator
from src.agentic_search import AgenticSearch

# ── MCP Server Setup ─────────────────────────────────────────────────────

app = Server("ragflow-docs")

# Lazy-initialized global components
_db: Database | None = None
_embedder: Embedder | None = None
_retriever: Retriever | None = None
_generator: Generator | None = None
_agentic_search: AgenticSearch | None = None


async def _get_components():
    """Lazy-initialize all components."""
    global _db, _embedder, _retriever, _generator, _agentic_search
    if _db is None:
        _db = Database()
        await _db.connect()
        _embedder = Embedder()
        _retriever = Retriever(_db, _embedder)
        _generator = Generator()
        _agentic_search = AgenticSearch(_retriever)
    return _db, _embedder, _retriever, _generator, _agentic_search


# ── Tool Definitions ─────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_ragflow_docs",
            description=(
                "Search the RAGFlow developer documentation (HTTP API and Python SDK references). "
                "Use this to find specific API endpoints, SDK methods, parameter details, "
                "request/response formats, and code examples. "
                "Supports hybrid search combining semantic understanding and keyword matching."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query. Can be natural language (e.g., 'how to create a dataset') "
                            "or specific terms (e.g., 'POST /api/v1/datasets', 'RAGFlow.create_dataset')."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return. Default: 5.",
                        "default": 5,
                    },
                    "search_mode": {
                        "type": "string",
                        "enum": ["hybrid", "vector", "fts"],
                        "description": (
                            "Search mode: 'hybrid' (default, best for most queries), "
                            "'vector' (semantic similarity only), "
                            "'fts' (keyword matching only, good for exact endpoint/method names)."
                        ),
                        "default": "hybrid",
                    },
                    "doc_filter": {
                        "type": "string",
                        "enum": [
                            "all",
                            "http_api_reference.md",
                            "python_api_reference.md",
                            "glossary.mdx",
                        ],
                        "description": "Filter by documentation source. Default: 'all'.",
                        "default": "all",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask_ragflow_docs",
            description=(
                "Ask a question about RAGFlow and get an AI-generated answer based on the official "
                "documentation. The answer includes relevant code examples and references. "
                "Use this for complex questions that need synthesized answers from multiple doc sections."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The question about RAGFlow. Examples: "
                            "'How do I upload documents to a dataset using the Python SDK?', "
                            "'What parameters does the retrieval API accept?', "
                            "'How to create a chat assistant with custom LLM settings?'"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documentation chunks to use as context. Default: 6.",
                        "default": 6,
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="list_api_endpoints",
            description=(
                "List all available RAGFlow API endpoints grouped by category. "
                "Returns a structured overview of the HTTP API including methods, URLs, and descriptions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional category filter. Examples: 'dataset', 'document', 'chunk', "
                            "'chat', 'agent', 'session', 'memory', 'file'. "
                            "Leave empty to list all."
                        ),
                    },
                },
            },
        ),
        Tool(
            name="lookup_api_endpoint",
            description=(
                "Look up detailed documentation for a specific RAGFlow API endpoint. "
                "Provide the HTTP method and URL pattern to get full details including "
                "parameters, request examples, and response formats."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                        "description": "HTTP method.",
                    },
                    "url_pattern": {
                        "type": "string",
                        "description": (
                            "URL pattern to search for. Examples: '/api/v1/datasets', "
                            "'/api/v1/retrieval', '/api/v1/chats'. "
                            "Partial matches are supported."
                        ),
                    },
                },
                "required": ["url_pattern"],
            },
        ),
        Tool(
            name="agentic_search_ragflow_docs",
            description=(
                "Perform an agentic search over RAGFlow developer documentation. "
                "Unlike simple search, this tool automatically decomposes complex questions "
                "into sub-queries, performs multiple rounds of retrieval, evaluates whether "
                "enough context has been gathered, and synthesizes a comprehensive answer. "
                "Best for complex, multi-faceted questions that span multiple API endpoints, "
                "SDK methods, or concepts. "
                "Example: 'How do I create a dataset, upload documents, and then set up "
                "a chat assistant that uses retrieval over those documents?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "A complex question about RAGFlow that may require searching "
                            "across multiple documentation sections. Examples: "
                            "'What is the full workflow for building a RAG pipeline using the Python SDK?', "
                            "'How do I manage document chunks and configure retrieval settings for a chat assistant?'"
                        ),
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "Maximum number of search rounds. Default: 3.",
                        "default": 3,
                    },
                    "top_k_per_query": {
                        "type": "integer",
                        "description": "Number of results per sub-query per round. Default: 5.",
                        "default": 5,
                    },
                },
                "required": ["question"],
            },
        ),
    ]


# ── Tool Handlers ────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    _, _, retriever, generator, agentic_search = await _get_components()

    if name == "search_ragflow_docs":
        return await _handle_search(retriever, arguments)
    elif name == "ask_ragflow_docs":
        return await _handle_ask(retriever, generator, arguments)
    elif name == "list_api_endpoints":
        return await _handle_list_endpoints(retriever, arguments)
    elif name == "lookup_api_endpoint":
        return await _handle_lookup_endpoint(retriever, arguments)
    elif name == "agentic_search_ragflow_docs":
        return await _handle_agentic_search(agentic_search, arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_search(retriever: Retriever, args: dict) -> list[TextContent]:
    """Handle search_ragflow_docs."""
    query = args["query"]
    top_k = args.get("top_k", 5)
    search_mode = args.get("search_mode", "hybrid")
    doc_filter = args.get("doc_filter", "all")

    results = await retriever.search(
        query=query,
        top_k=top_k,
        search_mode=search_mode,
    )

    # Apply doc filter
    if doc_filter != "all":
        results = [r for r in results if r.doc_name == doc_filter]

    if not results:
        return [TextContent(
            type="text",
            text=f"No results found for: '{query}'\n\nTry a different query or search mode.",
        )]

    # Format results
    output_parts = [f"## Search Results for: \"{query}\"\n"]
    output_parts.append(f"Mode: {search_mode} | Results: {len(results)}\n")

    for i, r in enumerate(results, 1):
        output_parts.append(f"### Result {i} (score: {r.hybrid_score:.4f})")
        if r.section_path:
            output_parts.append(f"**Section:** {r.section_path}")
        if r.api_method and r.endpoint_url:
            output_parts.append(f"**API:** `{r.api_method} {r.endpoint_url}`")
        if r.sdk_signature:
            output_parts.append(f"**SDK:** `{r.sdk_signature}`")
        output_parts.append(f"**Type:** {r.chunk_type} | **Source:** {r.doc_name}")
        output_parts.append(f"\n{r.content}\n")
        output_parts.append("---\n")

    return [TextContent(type="text", text="\n".join(output_parts))]


async def _handle_ask(
    retriever: Retriever, generator: Generator, args: dict
) -> list[TextContent]:
    """Handle ask_ragflow_docs — RAG-powered Q&A."""
    question = args["question"]
    top_k = args.get("top_k", 6)

    # Retrieve relevant docs
    results = await retriever.search(query=question, top_k=top_k, search_mode="hybrid")

    if not results:
        return [TextContent(
            type="text",
            text=(
                f"I couldn't find relevant documentation for: '{question}'\n\n"
                "Please try rephrasing your question or search for specific API endpoints."
            ),
        )]

    # Generate answer
    context_chunks = [r.to_dict() for r in results]
    answer = await generator.generate(question=question, context_chunks=context_chunks)

    # Append references
    refs = []
    for i, r in enumerate(results, 1):
        ref_parts = [f"{i}."]
        if r.section_path:
            ref_parts.append(r.section_path)
        if r.api_method and r.endpoint_url:
            ref_parts.append(f"({r.api_method} {r.endpoint_url})")
        ref_parts.append(f"[{r.doc_name}]")
        refs.append(" ".join(ref_parts))

    output = f"{answer}\n\n---\n**References:**\n" + "\n".join(refs)

    return [TextContent(type="text", text=output)]


async def _handle_list_endpoints(retriever: Retriever, args: dict) -> list[TextContent]:
    """Handle list_api_endpoints — search for API endpoint summaries."""
    category = args.get("category", "")

    query = f"API endpoints {category}" if category else "list all API endpoints HTTP methods"
    results = await retriever.search(query=query, top_k=20, search_mode="fts")

    # Filter to only api_endpoint chunks
    endpoints = [r for r in results if r.api_method and r.endpoint_url]

    if not endpoints:
        # Fallback: broader search
        results = await retriever.search(
            query=query, top_k=30, search_mode="hybrid"
        )
        endpoints = [r for r in results if r.api_method and r.endpoint_url]

    # Deduplicate by endpoint
    seen = set()
    unique_endpoints = []
    for ep in endpoints:
        key = f"{ep.api_method} {ep.endpoint_url}"
        if key not in seen:
            seen.add(key)
            unique_endpoints.append(ep)

    if not unique_endpoints:
        return [TextContent(
            type="text",
            text="Could not find API endpoint listings. Try searching with a specific query.",
        )]

    # Group by section
    groups: dict[str, list] = {}
    for ep in unique_endpoints:
        # Extract category from section_path
        parts = ep.section_path.split(" > ")
        group = parts[1] if len(parts) > 1 else "Other"
        groups.setdefault(group, []).append(ep)

    output_parts = ["## RAGFlow API Endpoints\n"]
    for group_name, eps in sorted(groups.items()):
        output_parts.append(f"### {group_name}\n")
        for ep in eps:
            output_parts.append(f"- **{ep.api_method}** `{ep.endpoint_url}`")
            # Extract brief description from section path
            parts = ep.section_path.split(" > ")
            if len(parts) > 2:
                output_parts[-1] += f" — {parts[-1]}"
        output_parts.append("")

    return [TextContent(type="text", text="\n".join(output_parts))]


async def _handle_lookup_endpoint(retriever: Retriever, args: dict) -> list[TextContent]:
    """Handle lookup_api_endpoint — find specific endpoint docs."""
    method = args.get("method", "")
    url_pattern = args["url_pattern"]

    query = f"{method} {url_pattern}" if method else url_pattern
    results = await retriever.search(query=query, top_k=10, search_mode="hybrid")

    # Filter by endpoint URL pattern
    matching = [
        r for r in results
        if url_pattern.lower() in (r.endpoint_url or "").lower()
        or url_pattern.lower() in r.content.lower()
    ]

    if method:
        matching = [r for r in matching if r.api_method == method.upper()] or matching

    if not matching:
        matching = results[:5]  # fallback to top results

    output_parts = [f"## API Documentation: {method} {url_pattern}\n"]

    for r in matching:
        if r.section_path:
            output_parts.append(f"**Section:** {r.section_path}")
        if r.api_method and r.endpoint_url:
            output_parts.append(f"**Endpoint:** `{r.api_method} {r.endpoint_url}`")
        output_parts.append(f"\n{r.content}\n")
        output_parts.append("---\n")

    return [TextContent(type="text", text="\n".join(output_parts))]


async def _handle_agentic_search(
    agentic_search: AgenticSearch, args: dict
) -> list[TextContent]:
    """Handle agentic_search_ragflow_docs — multi-step agentic search."""
    question = args["question"]
    max_rounds = args.get("max_rounds", 3)
    top_k_per_query = args.get("top_k_per_query", 5)

    result = await agentic_search.search(
        question=question,
        max_rounds=max_rounds,
        top_k_per_query=top_k_per_query,
    )

    # Build output with answer and search metadata
    output_parts = [result.answer]
    output_parts.append("\n\n---")
    output_parts.append(
        f"**Agentic Search Stats:** {result.rounds_executed} round(s), "
        f"{result.total_chunks_retrieved} unique chunks retrieved"
    )

    # Append references
    if result.all_results:
        refs = []
        for i, r in enumerate(result.all_results, 1):
            ref_parts = [f"{i}."]
            if r.section_path:
                ref_parts.append(r.section_path)
            if r.api_method and r.endpoint_url:
                ref_parts.append(f"({r.api_method} {r.endpoint_url})")
            ref_parts.append(f"[{r.doc_name}]")
            refs.append(" ".join(ref_parts))
        output_parts.append("\n**References:**\n" + "\n".join(refs))

    return [TextContent(type="text", text="\n".join(output_parts))]


# ── Server Entry Point ───────────────────────────────────────────────────

class StreamableHTTPASGIApp:
    """ASGI application wrapper for MCP Streamable HTTP session manager."""

    def __init__(self, session_manager: StreamableHTTPSessionManager):
        self.session_manager = session_manager

    async def __call__(self, scope, receive, send) -> None:
        await self.session_manager.handle_request(scope, receive, send)


def create_streamable_http_app(path: str = "/mcp") -> Starlette:
    """Create a Starlette app that serves MCP over Streamable HTTP."""
    normalized_path = path if path.startswith("/") else f"/{path}"
    session_manager = StreamableHTTPSessionManager(app=app)
    streamable_http_app = StreamableHTTPASGIApp(session_manager)

    @asynccontextmanager
    async def lifespan(_):
        async with session_manager.run():
            yield

    return Starlette(
        routes=[Route(normalized_path, endpoint=streamable_http_app)],
        lifespan=lifespan,
    )


async def run_server(host: str = "127.0.0.1", port: int = 8000, path: str = "/mcp"):
    """Run the MCP server via Streamable HTTP transport."""
    server = uvicorn.Server(
        uvicorn.Config(
            create_streamable_http_app(path=path),
            host=host,
            port=port,
            log_level="info",
        )
    )
    await server.serve()


def main(host: str = "127.0.0.1", port: int = 8000, path: str = "/mcp"):
    """Entry point for the MCP server."""
    asyncio.run(run_server(host=host, port=port, path=path))


if __name__ == "__main__":
    main()
