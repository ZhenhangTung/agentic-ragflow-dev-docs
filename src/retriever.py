"""
Hybrid retrieval engine combining vector search and full-text search.

Supports optional LLM-based pre-filtering that uses a small model to
select the most relevant document files before running the expensive
hybrid search.
"""
import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from src.db import Database
from src.embedder import Embedder
from src.config import get_settings

logger = logging.getLogger(__name__)

# ── Pre-filter prompt ────────────────────────────────────────────────────

_PREFILTER_PROMPT = """\
You are a file-routing agent for RAGFlow developer documentation.

Given a developer's query and a list of documentation files with their metadata, \
select which files are most likely to contain the answer.

Query: {query}

Available files:
{file_descriptions}

Respond with ONLY a JSON array of file names that are relevant. \
Include ALL files that might contain relevant information. \
If unsure, include the file.

Example: ["http_api_reference.md", "python_api_reference.md"]"""


@dataclass
class RetrievalResult:
    """A single retrieval result with scoring details."""
    id: int
    content: str
    doc_name: str
    section_path: str
    chunk_type: str
    api_method: str
    endpoint_url: str
    sdk_signature: str
    metadata: dict
    vector_score: float
    fts_score: float
    hybrid_score: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "doc_name": self.doc_name,
            "section_path": self.section_path,
            "chunk_type": self.chunk_type,
            "api_method": self.api_method,
            "endpoint_url": self.endpoint_url,
            "sdk_signature": self.sdk_signature,
            "metadata": self.metadata,
            "vector_score": self.vector_score,
            "fts_score": self.fts_score,
            "hybrid_score": self.hybrid_score,
        }


class Retriever:
    """Hybrid retrieval engine for developer documentation."""

    def __init__(self, db: Database, embedder: Embedder):
        self.db = db
        self.embedder = embedder
        self.settings = get_settings()
        self._file_metadata_cache: list[dict] | None = None

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        vector_weight: float | None = None,
        fts_weight: float | None = None,
        search_mode: str = "hybrid",  # "hybrid", "vector", "fts"
        use_prefilter: bool = True,
    ) -> list[RetrievalResult]:
        """
        Search the documentation using hybrid retrieval.

        Args:
            query: The search query
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            fts_weight: Weight for full-text search score (0-1)
            search_mode: "hybrid", "vector", or "fts"
            use_prefilter: Use LLM pre-filter to select relevant files first
        """
        top_k = top_k or self.settings.top_k
        vector_weight = vector_weight if vector_weight is not None else self.settings.vector_weight
        fts_weight = fts_weight if fts_weight is not None else self.settings.fts_weight

        # Pre-filter: select relevant files using small model
        relevant_docs = None
        if use_prefilter and search_mode == "hybrid":
            relevant_docs = await self._prefilter_files(query)
            if relevant_docs:
                logger.info("Pre-filter selected files: %s", relevant_docs)

        if search_mode == "vector":
            return await self._vector_search(query, top_k)
        elif search_mode == "fts":
            return await self._fts_search(query, top_k)
        else:
            return await self._hybrid_search(
                query, top_k, vector_weight, fts_weight,
                doc_names=relevant_docs,
            )

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        vector_weight: float,
        fts_weight: float,
        doc_names: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid search: vector + full-text, optionally filtered by doc names."""
        query_embedding = await self.embedder.embed_text(query)

        if doc_names:
            rows = await self.db.filtered_hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
                doc_names=doc_names,
            )
        else:
            rows = await self.db.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k,
                vector_weight=vector_weight,
                fts_weight=fts_weight,
            )

        results = []
        for row in rows:
            score = row.get("hybrid_score", 0)
            if score < self.settings.similarity_threshold:
                continue
            results.append(RetrievalResult(
                id=row["id"],
                content=row["content"],
                doc_name=row["doc_name"],
                section_path=row["section_path"],
                chunk_type=row["chunk_type"],
                api_method=row.get("api_method", ""),
                endpoint_url=row.get("endpoint_url", ""),
                sdk_signature=row.get("sdk_signature", ""),
                metadata=row.get("metadata", {}),
                vector_score=row.get("vector_score", 0),
                fts_score=row.get("fts_score", 0),
                hybrid_score=score,
            ))
        return results

    async def _vector_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Pure vector search."""
        query_embedding = await self.embedder.embed_text(query)
        rows = await self.db.vector_search(query_embedding, top_k)

        return [
            RetrievalResult(
                id=row["id"],
                content=row["content"],
                doc_name=row["doc_name"],
                section_path=row["section_path"],
                chunk_type=row["chunk_type"],
                api_method=row.get("api_method", ""),
                endpoint_url=row.get("endpoint_url", ""),
                sdk_signature=row.get("sdk_signature", ""),
                metadata=row.get("metadata", {}),
                vector_score=row.get("vector_score", 0),
                fts_score=0,
                hybrid_score=row.get("vector_score", 0),
            )
            for row in rows
        ]

    async def _fts_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Pure full-text search."""
        rows = await self.db.fts_search(query, top_k)

        return [
            RetrievalResult(
                id=row["id"],
                content=row["content"],
                doc_name=row["doc_name"],
                section_path=row["section_path"],
                chunk_type=row["chunk_type"],
                api_method=row.get("api_method", ""),
                endpoint_url=row.get("endpoint_url", ""),
                sdk_signature=row.get("sdk_signature", ""),
                metadata=row.get("metadata", {}),
                vector_score=0,
                fts_score=row.get("fts_score", 0),
                hybrid_score=row.get("fts_score", 0),
            )
            for row in rows
        ]

    # ── Pre-filter ───────────────────────────────────────────────────

    async def _load_file_metadata(self) -> list[dict]:
        """Load and cache file metadata from the database."""
        if self._file_metadata_cache is None:
            self._file_metadata_cache = await self.db.get_all_file_metadata()
        return self._file_metadata_cache

    async def _prefilter_files(self, query: str) -> list[str] | None:
        """Use a small LLM to select the most relevant doc files for *query*.

        Returns a list of doc_name strings, or ``None`` to skip filtering
        (e.g. when no file metadata exists or the LLM call fails).
        """
        file_metas = await self._load_file_metadata()
        if not file_metas:
            return None

        descriptions = []
        for fm in file_metas:
            match_text = fm.get("match_text", "")
            desc = f"- {fm['doc_name']}: {match_text[:300]}"
            descriptions.append(desc)

        prompt = _PREFILTER_PROMPT.format(
            query=query,
            file_descriptions="\n".join(descriptions),
        )

        client = AsyncOpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url,
        )
        try:
            resp = await client.chat.completions.create(
                model=self.settings.light_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                extra_body={"enable_thinking": False},
            )
            text = resp.choices[0].message.content.strip()
            # Extract JSON array from response
            start = text.index("[")
            end = text.rindex("]") + 1
            doc_names: list[str] = json.loads(text[start:end])
            # Validate against known files
            known = {fm["doc_name"] for fm in file_metas}
            valid = [d for d in doc_names if d in known]
            return valid if valid else None
        except Exception:
            logger.warning("Pre-filter LLM call failed, skipping filter", exc_info=True)
            return None
