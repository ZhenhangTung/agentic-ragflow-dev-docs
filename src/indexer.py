"""
Indexing pipeline: Download docs → Chunk → Enrich metadata → Embed → Store in PostgreSQL.
"""
import asyncio
import json
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import get_settings
from src.downloader import download_docs
from src.chunker import chunk_all_docs, DocChunk
from src.embedder import Embedder
from src.db import Database
from src.metadata_enricher import (
    generate_file_metadata,
    enrich_chunks_with_file_metadata,
    FileMetadata,
)

console = Console()


async def run_indexing_pipeline(force_download: bool = False, force_reindex: bool = False):
    """
    Full indexing pipeline:
    1. Download docs from GitHub
    2. Chunk with developer-doc-aware strategy
    3. Generate embeddings with Qwen text-embedding-v4
    4. Store in PostgreSQL with pgvector + full-text search
    """
    settings = get_settings()

    console.print("\n[bold blue]🚀 RAGFlow Docs Indexing Pipeline[/bold blue]\n")

    # Step 1: Download docs
    console.print("[bold]Step 1:[/bold] Downloading documentation...")
    doc_paths = await download_docs(force=force_download)
    console.print(f"  [green]✓ {len(doc_paths)} documents ready[/green]\n")

    # Step 2: Chunk documents
    console.print("[bold]Step 2:[/bold] Chunking documents with dev-doc strategy...")
    chunks = chunk_all_docs(settings.docs_dir, max_chunk_tokens=settings.chunk_size)
    console.print(f"  [green]✓ {len(chunks)} chunks created[/green]")

    # Print chunk statistics
    chunk_types = {}
    for c in chunks:
        chunk_types[c.chunk_type] = chunk_types.get(c.chunk_type, 0) + 1
    for ct, count in sorted(chunk_types.items()):
        console.print(f"    {ct}: {count}")
    console.print()

    # Step 2.5: Generate file-level metadata (built-in + LLM)
    console.print("[bold]Step 2.5:[/bold] Generating file metadata (built-in + LLM)...")
    doc_files_set = set(c.doc_name for c in chunks)
    file_metadata: dict[str, FileMetadata] = {}
    for doc_name in sorted(doc_files_set):
        filepath = os.path.join(settings.docs_dir, doc_name)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                full_text = f.read()
        else:
            full_text = "\n".join(
                c.content for c in chunks if c.doc_name == doc_name
            )
        fm = await generate_file_metadata(doc_name, full_text)
        file_metadata[doc_name] = fm
        console.print(
            f"    {doc_name}: category={fm.file_category}, "
            f"topics={len(fm.topics)}, keywords={len(fm.keywords)}"
        )
    chunks = enrich_chunks_with_file_metadata(chunks, file_metadata)
    console.print(f"  [green]✓ Metadata generated for {len(file_metadata)} files[/green]\n")

    # Step 3: Initialize database
    console.print("[bold]Step 3:[/bold] Setting up PostgreSQL...")
    db = Database()
    await db.connect()
    await db.initialize()

    if force_reindex:
        console.print("  [yellow]Clearing existing data...[/yellow]")
        await db.clear_all()

    existing_count = await db.count_chunks()
    if existing_count > 0 and not force_reindex:
        console.print(f"  [yellow]Database already has {existing_count} chunks. Use --force-reindex to rebuild.[/yellow]")
        await db.close()
        return

    console.print(f"  [green]✓ Database initialized[/green]\n")

    # Step 4: Generate embeddings and insert
    console.print("[bold]Step 4:[/bold] Generating embeddings & indexing...")
    embedder = Embedder()

    # Prepare texts for batch embedding
    texts = [c.to_indexable_text() for c in chunks]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        # Embed in batches
        task = progress.add_task("Embedding chunks...", total=len(texts))
        embeddings = []
        batch_size = Embedder.MAX_BATCH_SIZE
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await embedder.embed_batch(batch, batch_size=batch_size)
            embeddings.extend(batch_embeddings)
            progress.advance(task, len(batch))

    console.print(f"  [green]✓ {len(embeddings)} embeddings generated[/green]\n")

    # Step 5: Insert into database
    console.print("[bold]Step 5:[/bold] Storing in PostgreSQL...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Inserting chunks...", total=len(chunks))
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            enriched_meta = getattr(chunk, "_enriched_metadata", None) or chunk.to_metadata()
            await db.insert_chunk(
                content=chunk.content,
                indexable_text=chunk.to_indexable_text(),
                embedding=embedding,
                doc_name=chunk.doc_name,
                section_path=chunk.section_path,
                chunk_type=chunk.chunk_type,
                api_method=chunk.api_method,
                endpoint_url=chunk.endpoint_url,
                sdk_signature=chunk.sdk_signature,
                language=chunk.language,
                chunk_index=chunk.chunk_index,
                metadata=enriched_meta,
            )
            progress.advance(task)

    # Step 6: Create search indexes
    console.print("\n[bold]Step 6:[/bold] Creating search indexes...")
    await db.create_indexes()
    console.print(f"  [green]✓ Indexes created[/green]\n")
    # Step 7: Store file metadata for pre-filter
    console.print("[bold]Step 7:[/bold] Storing file metadata for pre-filter...")
    await db.save_file_metadata(
        {name: fm.to_dict() for name, fm in file_metadata.items()}
    )
    console.print(f"  [green]✓ File metadata stored[/green]\n")
    final_count = await db.count_chunks()
    await db.close()

    console.print(f"[bold green]✅ Indexing complete! {final_count} chunks indexed.[/bold green]\n")
