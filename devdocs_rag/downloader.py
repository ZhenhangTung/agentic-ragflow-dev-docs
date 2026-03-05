"""
DevDocs RAG Framework - Download documentation from GitHub.
"""
import os
import httpx
from rich.console import Console
from devdocs_rag.config import get_settings

console = Console()


async def download_docs(force: bool = False) -> list[str]:
    """
    Download reference docs from the RAGFlow GitHub repo.
    Returns list of local file paths.
    """
    settings = get_settings()
    docs_dir = settings.docs_dir
    os.makedirs(docs_dir, exist_ok=True)

    downloaded = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for filename in settings.doc_files:
            local_path = os.path.join(docs_dir, filename)

            if os.path.exists(local_path) and not force:
                console.print(f"  [dim]Skipping {filename} (already exists)[/dim]")
                downloaded.append(local_path)
                continue

            url = f"{settings.github_raw_base}/{filename}"
            console.print(f"  [cyan]Downloading {filename}...[/cyan]")

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                downloaded.append(local_path)
                console.print(f"  [green]✓ {filename} ({len(resp.text)} chars)[/green]")
            except httpx.HTTPError as e:
                console.print(f"  [red]✗ Failed to download {filename}: {e}[/red]")

    return downloaded


def list_local_docs() -> list[str]:
    """List locally available doc files."""
    settings = get_settings()
    docs_dir = settings.docs_dir
    if not os.path.exists(docs_dir):
        return []
    return [
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if f.endswith((".md", ".mdx"))
    ]
