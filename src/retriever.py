"""
Hybrid retrieval engine combining vector search and full-text search.
"""
from dataclasses import dataclass
from src.db import Database
from src.embedder import Embedder
from src.config import get_settings


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

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        vector_weight: float | None = None,
        fts_weight: float | None = None,
        search_mode: str = "hybrid",  # "hybrid", "vector", "fts"
    ) -> list[RetrievalResult]:
        """
        Search the documentation using hybrid retrieval.
        
        Args:
            query: The search query
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            fts_weight: Weight for full-text search score (0-1)
            search_mode: "hybrid", "vector", or "fts"
        """
        top_k = top_k or self.settings.top_k
        vector_weight = vector_weight if vector_weight is not None else self.settings.vector_weight
        fts_weight = fts_weight if fts_weight is not None else self.settings.fts_weight

        if search_mode == "vector":
            return await self._vector_search(query, top_k)
        elif search_mode == "fts":
            return await self._fts_search(query, top_k)
        else:
            return await self._hybrid_search(query, top_k, vector_weight, fts_weight)

    async def _hybrid_search(
        self, query: str, top_k: int, vector_weight: float, fts_weight: float
    ) -> list[RetrievalResult]:
        """Hybrid search: vector + full-text."""
        query_embedding = await self.embedder.embed_text(query)

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
