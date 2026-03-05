"""
CLI entry point for the DevDocs RAG Framework.

Commands:
    index           — Download and index documentation
    serve           — Start the MCP server
    search          — Test search (interactive or one-shot)
    ask             — Ask a question about the project docs
    agentic-search  — Multi-step agentic search
    status          — Show index status

Configured for the project set in PROJECT_NAME environment variable.
"""
import asyncio
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def _project_name() -> str:
    """Get the configured project name for CLI display."""
    from devdocs_rag.config import get_settings
    return get_settings().project_name


@click.group()
def cli():
    """DevDocs RAG — Developer documentation assistant powered by MCP."""
    pass


# ── index ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--force-download", is_flag=True, help="Re-download docs even if they exist.")
@click.option("--force-reindex", is_flag=True, help="Clear database and re-index everything.")
def index(force_download: bool, force_reindex: bool):
    """Download and index project documentation."""
    from devdocs_rag.indexer import run_indexing_pipeline

    asyncio.run(run_indexing_pipeline(
        force_download=force_download,
        force_reindex=force_reindex,
    ))


# ── serve ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host for MCP HTTP server.")
@click.option("--port", default=8000, show_default=True, help="Bind port for MCP HTTP server.")
@click.option("--path", default="/mcp", show_default=True, help="HTTP path for MCP endpoint.")
def serve(host: str, port: int, path: str):
    """Start the MCP server (Streamable HTTP transport)."""
    from devdocs_rag.mcp_server import main as mcp_main
    from devdocs_rag.config import get_settings

    settings = get_settings()
    project = settings.project_name
    slug = settings.project_slug

    console.print(Panel(
        f"[bold green]Starting {project} Docs MCP Server[/]\n"
        "Transport: streamable-http\n"
        f"Endpoint: http://{host}:{port}{path}\n"
        f"Tools: search_{slug}_docs, ask_{slug}_docs, list_api_endpoints,\n"
        f"       lookup_api_endpoint, agentic_search_{slug}_docs",
        title="MCP Server",
    ))
    mcp_main(host=host, port=port, path=path)


# ── search ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query", required=False)
@click.option("-k", "--top-k", default=5, help="Number of results.")
@click.option("-m", "--mode", default="hybrid", type=click.Choice(["hybrid", "vector", "fts"]))
def search(query: str | None, top_k: int, mode: str):
    """Search the documentation. Pass QUERY or enter interactive mode."""
    asyncio.run(_search(query, top_k, mode))


async def _search(query: str | None, top_k: int, mode: str):
    from devdocs_rag.db import Database
    from devdocs_rag.embedder import Embedder
    from devdocs_rag.retriever import Retriever

    db = Database()
    await db.connect()
    embedder = Embedder()
    retriever = Retriever(db, embedder)

    try:
        if query:
            await _run_search(retriever, query, top_k, mode)
        else:
            console.print("[bold]Interactive search mode[/] (type 'quit' to exit)\n")
            while True:
                q = console.input("[bold cyan]Search>[/] ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    await _run_search(retriever, q, top_k, mode)
    finally:
        await db.close()


async def _run_search(retriever, query: str, top_k: int, mode: str):
    results = await retriever.search(query=query, top_k=top_k, search_mode=mode)

    if not results:
        console.print("[yellow]No results found.[/]\n")
        return

    table = Table(title=f"Results for: {query}", show_lines=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Type", width=14)
    table.add_column("Section", width=40)
    table.add_column("Endpoint", width=30)

    for i, r in enumerate(results, 1):
        endpoint = ""
        if r.api_method and r.endpoint_url:
            endpoint = f"{r.api_method} {r.endpoint_url}"
        elif r.sdk_signature:
            endpoint = r.sdk_signature
        table.add_row(
            str(i),
            f"{r.hybrid_score:.4f}",
            r.chunk_type or "-",
            r.section_path or "-",
            endpoint or "-",
        )

    console.print(table)

    # Show first result content
    if results:
        console.print(Panel(
            Markdown(results[0].content[:1500]),
            title=f"Top Result: {results[0].section_path or 'N/A'}",
            border_style="green",
        ))
    console.print()


# ── ask ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question", required=False)
@click.option("-k", "--top-k", default=6, help="Number of context chunks.")
def ask(question: str | None, top_k: int):
    """Ask a question about the project documentation (RAG-powered)."""
    asyncio.run(_ask(question, top_k))


async def _ask(question: str | None, top_k: int):
    from devdocs_rag.db import Database
    from devdocs_rag.embedder import Embedder
    from devdocs_rag.retriever import Retriever
    from devdocs_rag.generator import Generator

    db = Database()
    await db.connect()
    embedder = Embedder()
    retriever = Retriever(db, embedder)
    generator = Generator()

    try:
        if question:
            await _run_ask(retriever, generator, question, top_k)
        else:
            console.print("[bold]Interactive Q&A mode[/] (type 'quit' to exit)\n")
            while True:
                q = console.input("[bold cyan]Ask>[/] ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    await _run_ask(retriever, generator, q, top_k)
    finally:
        await db.close()


async def _run_ask(retriever, generator, question: str, top_k: int):
    with console.status("[bold green]Searching documentation..."):
        results = await retriever.search(query=question, top_k=top_k, search_mode="hybrid")

    if not results:
        console.print("[yellow]No relevant documentation found.[/]\n")
        return

    context_chunks = [r.to_dict() for r in results]

    with console.status("[bold green]Generating answer..."):
        answer = await generator.generate(question=question, context_chunks=context_chunks)

    console.print(Panel(
        Markdown(answer),
        title="Answer",
        border_style="green",
    ))

    # Show references
    console.print("[dim]References:[/]")
    for i, r in enumerate(results, 1):
        ref = f"  {i}. {r.section_path or 'N/A'}"
        if r.api_method and r.endpoint_url:
            ref += f" ({r.api_method} {r.endpoint_url})"
        ref += f" [{r.doc_name}]"
        console.print(f"[dim]{ref}[/]")
    console.print()


# ── agentic-search ────────────────────────────────────────────────────────

@cli.command("agentic-search")
@click.argument("question", required=False)
@click.option("-r", "--max-rounds", default=3, help="Maximum search rounds.")
@click.option("-k", "--top-k", default=5, help="Results per sub-query.")
def agentic_search(question: str | None, max_rounds: int, top_k: int):
    """Agentic search — multi-step retrieval with query decomposition."""
    asyncio.run(_agentic_search(question, max_rounds, top_k))


async def _agentic_search(question: str | None, max_rounds: int, top_k: int):
    from devdocs_rag.db import Database
    from devdocs_rag.embedder import Embedder
    from devdocs_rag.retriever import Retriever
    from devdocs_rag.agentic_search import AgenticSearch

    db = Database()
    await db.connect()
    embedder = Embedder()
    retriever = Retriever(db, embedder)
    agent = AgenticSearch(retriever)

    try:
        if question:
            await _run_agentic_search(agent, question, max_rounds, top_k)
        else:
            console.print("[bold]Interactive agentic search mode[/] (type 'quit' to exit)\n")
            while True:
                q = console.input("[bold cyan]AgenticSearch>[/] ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    await _run_agentic_search(agent, q, max_rounds, top_k)
    finally:
        await db.close()


async def _run_agentic_search(agent, question: str, max_rounds: int, top_k: int):
    with console.status("[bold green]Running agentic search..."):
        result = await agent.search(
            question=question, max_rounds=max_rounds, top_k_per_query=top_k
        )

    # Show search stats
    console.print(Panel(
        f"Rounds: {result.rounds_executed} | "
        f"Unique chunks retrieved: {result.total_chunks_retrieved}",
        title="Agentic Search Stats",
        border_style="blue",
    ))

    # Show queries per round
    for i, rnd in enumerate(result.rounds, 1):
        queries_str = ", ".join(f'"{q}"' for q in rnd.queries)
        console.print(f"[dim]  Round {i}: {queries_str} → {len(rnd.results)} new chunks[/]")
    console.print()

    # Show answer
    console.print(Panel(
        Markdown(result.answer),
        title="Answer",
        border_style="green",
    ))

    # Show references
    if result.all_results:
        console.print("[dim]References:[/]")
        for i, r in enumerate(result.all_results, 1):
            ref = f"  {i}. {r.section_path or 'N/A'}"
            if r.api_method and r.endpoint_url:
                ref += f" ({r.api_method} {r.endpoint_url})"
            ref += f" [{r.doc_name}]"
            console.print(f"[dim]{ref}[/]")
    console.print()


# ── status ────────────────────────────────────────────────────────────────

@cli.command()
def status():
    """Show index status and statistics."""
    asyncio.run(_status())


async def _status():
    from devdocs_rag.db import Database
    from devdocs_rag.downloader import list_local_docs
    from devdocs_rag.config import get_settings

    settings = get_settings()

    # Check local docs
    local_docs = list_local_docs()
    console.print(Panel(
        "\n".join(f"  • {d}" for d in local_docs) if local_docs else "  [yellow]No documents downloaded[/]",
        title="Local Documents",
    ))

    # Check database
    try:
        db = Database()
        await db.connect()
        count = await db.count_chunks()
        await db.close()
        console.print(f"[green]Database connected[/] — {count} chunks indexed")
    except Exception as e:
        console.print(f"[red]Database not available:[/] {e}")

    console.print(f"\nEmbedding model: {settings.embedding_model}")
    console.print(f"Chat model: {settings.chat_model}")
    console.print(f"Docs directory: {settings.docs_dir}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
