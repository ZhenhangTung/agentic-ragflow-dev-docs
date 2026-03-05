"""
PostgreSQL database layer with pgvector for vector search
and tsvector for full-text search.
"""
import json
import asyncpg
from pgvector.asyncpg import register_vector
from src.config import get_settings


# ── SQL Statements ───────────────────────────────────────────────────────

CREATE_EXTENSION_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS doc_chunks (
    id              SERIAL PRIMARY KEY,
    content         TEXT NOT NULL,
    indexable_text  TEXT NOT NULL,
    embedding       vector({dimensions}),
    
    -- Rich metadata for developer docs
    doc_name        TEXT NOT NULL,
    section_path    TEXT NOT NULL DEFAULT '',
    chunk_type      TEXT NOT NULL DEFAULT 'concept',
    api_method      TEXT DEFAULT '',
    endpoint_url    TEXT DEFAULT '',
    sdk_signature   TEXT DEFAULT '',
    language        TEXT DEFAULT '',
    chunk_index     INTEGER DEFAULT 0,
    metadata        JSONB DEFAULT '{{}}',
    
    -- Full-text search column
    fts_vector      tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(endpoint_url, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(sdk_signature, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(section_path, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'C')
    ) STORED,
    
    created_at      TIMESTAMP DEFAULT NOW()
);
"""

CREATE_VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding 
ON doc_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 20);
"""

CREATE_FTS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_doc_chunks_fts 
ON doc_chunks USING gin (fts_vector);
"""

CREATE_METADATA_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_name ON doc_chunks (doc_name);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_chunk_type ON doc_chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_api_method ON doc_chunks (api_method);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_endpoint_url ON doc_chunks (endpoint_url);
"""

INSERT_CHUNK_SQL = """
INSERT INTO doc_chunks (
    content, indexable_text, embedding, 
    doc_name, section_path, chunk_type,
    api_method, endpoint_url, sdk_signature, 
    language, chunk_index, metadata
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
RETURNING id;
"""

# Hybrid search query combining vector similarity and full-text relevance
HYBRID_SEARCH_SQL = """
WITH vector_search AS (
    SELECT 
        id, content, doc_name, section_path, chunk_type,
        api_method, endpoint_url, sdk_signature, metadata,
        1 - (embedding <=> $1::vector) AS vector_score
    FROM doc_chunks
    ORDER BY embedding <=> $1::vector
    LIMIT $2
),
fts_search AS (
    SELECT 
        id, content, doc_name, section_path, chunk_type,
        api_method, endpoint_url, sdk_signature, metadata,
        ts_rank_cd(fts_vector, query) AS fts_score
    FROM doc_chunks, plainto_tsquery('english', $3) query
    WHERE fts_vector @@ query
    ORDER BY fts_score DESC
    LIMIT $2
)
SELECT 
    COALESCE(v.id, f.id) AS id,
    COALESCE(v.content, f.content) AS content,
    COALESCE(v.doc_name, f.doc_name) AS doc_name,
    COALESCE(v.section_path, f.section_path) AS section_path,
    COALESCE(v.chunk_type, f.chunk_type) AS chunk_type,
    COALESCE(v.api_method, f.api_method) AS api_method,
    COALESCE(v.endpoint_url, f.endpoint_url) AS endpoint_url,
    COALESCE(v.sdk_signature, f.sdk_signature) AS sdk_signature,
    COALESCE(v.metadata, f.metadata) AS metadata,
    COALESCE(v.vector_score, 0) AS vector_score,
    COALESCE(f.fts_score, 0) AS fts_score,
    ($4 * COALESCE(v.vector_score, 0) + $5 * COALESCE(f.fts_score, 0)) AS hybrid_score
FROM vector_search v
FULL OUTER JOIN fts_search f ON v.id = f.id
ORDER BY hybrid_score DESC
LIMIT $6;
"""

# Simple vector search
VECTOR_SEARCH_SQL = """
SELECT 
    id, content, doc_name, section_path, chunk_type,
    api_method, endpoint_url, sdk_signature, metadata,
    1 - (embedding <=> $1::vector) AS vector_score
FROM doc_chunks
ORDER BY embedding <=> $1::vector
LIMIT $2;
"""

# FTS-only search
FTS_SEARCH_SQL = """
SELECT 
    id, content, doc_name, section_path, chunk_type,
    api_method, endpoint_url, sdk_signature, metadata,
    ts_rank_cd(fts_vector, query) AS fts_score
FROM doc_chunks, plainto_tsquery('english', $1) query
WHERE fts_vector @@ query
ORDER BY fts_score DESC
LIMIT $2;
"""

COUNT_CHUNKS_SQL = "SELECT COUNT(*) FROM doc_chunks;"
CLEAR_CHUNKS_SQL = "TRUNCATE doc_chunks RESTART IDENTITY;"

# ── File metadata table for pre-filter ───────────────────────────────────

CREATE_FILE_METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS file_metadata (
    doc_name    TEXT PRIMARY KEY,
    metadata    JSONB NOT NULL DEFAULT '{}',
    match_text  TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMP DEFAULT NOW()
);
"""

UPSERT_FILE_METADATA_SQL = """
INSERT INTO file_metadata (doc_name, metadata, match_text)
VALUES ($1, $2, $3)
ON CONFLICT (doc_name) DO UPDATE SET
    metadata = EXCLUDED.metadata,
    match_text = EXCLUDED.match_text;
"""

GET_ALL_FILE_METADATA_SQL = """
SELECT doc_name, metadata, match_text FROM file_metadata;
"""

# ── Filtered hybrid search (restrict to specific doc_names) ──────────────

FILTERED_HYBRID_SEARCH_SQL = """
WITH vector_search AS (
    SELECT
        id, content, doc_name, section_path, chunk_type,
        api_method, endpoint_url, sdk_signature, metadata,
        1 - (embedding <=> $1::vector) AS vector_score
    FROM doc_chunks
    WHERE doc_name = ANY($6)
    ORDER BY embedding <=> $1::vector
    LIMIT $2
),
fts_search AS (
    SELECT
        id, content, doc_name, section_path, chunk_type,
        api_method, endpoint_url, sdk_signature, metadata,
        ts_rank_cd(fts_vector, query) AS fts_score
    FROM doc_chunks, plainto_tsquery('english', $3) query
    WHERE fts_vector @@ query AND doc_name = ANY($6)
    ORDER BY fts_score DESC
    LIMIT $2
)
SELECT
    COALESCE(v.id, f.id) AS id,
    COALESCE(v.content, f.content) AS content,
    COALESCE(v.doc_name, f.doc_name) AS doc_name,
    COALESCE(v.section_path, f.section_path) AS section_path,
    COALESCE(v.chunk_type, f.chunk_type) AS chunk_type,
    COALESCE(v.api_method, f.api_method) AS api_method,
    COALESCE(v.endpoint_url, f.endpoint_url) AS endpoint_url,
    COALESCE(v.sdk_signature, f.sdk_signature) AS sdk_signature,
    COALESCE(v.metadata, f.metadata) AS metadata,
    COALESCE(v.vector_score, 0) AS vector_score,
    COALESCE(f.fts_score, 0) AS fts_score,
    ($4 * COALESCE(v.vector_score, 0) + $5 * COALESCE(f.fts_score, 0)) AS hybrid_score
FROM vector_search v
FULL OUTER JOIN fts_search f ON v.id = f.id
ORDER BY hybrid_score DESC
LIMIT $7;
"""


class Database:
    """Async PostgreSQL database with vector + full-text search."""

    def __init__(self):
        self.settings = get_settings()
        self.pool: asyncpg.Pool | None = None

    async def connect(self):
        """Create connection pool and register vector type."""
        self.pool = await asyncpg.create_pool(
            self.settings.postgres_dsn,
            min_size=2,
            max_size=10,
        )
        # Register pgvector type with each connection
        async with self.pool.acquire() as conn:
            await register_vector(conn)

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def initialize(self):
        """Create tables and indexes."""
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(CREATE_EXTENSION_SQL)
            
            create_sql = CREATE_TABLE_SQL.format(
                dimensions=self.settings.embedding_dimensions
            )
            await conn.execute(create_sql)
            await conn.execute(CREATE_FILE_METADATA_TABLE_SQL)

    async def create_indexes(self):
        """Create search indexes (call after data is loaded)."""
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_VECTOR_INDEX_SQL)
            await conn.execute(CREATE_FTS_INDEX_SQL)
            await conn.execute(CREATE_METADATA_INDEXES_SQL)

    async def insert_chunk(
        self,
        content: str,
        indexable_text: str,
        embedding: list[float],
        doc_name: str,
        section_path: str,
        chunk_type: str,
        api_method: str = "",
        endpoint_url: str = "",
        sdk_signature: str = "",
        language: str = "",
        chunk_index: int = 0,
        metadata: dict | None = None,
    ) -> int:
        """Insert a chunk with its embedding and metadata."""
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            row_id = await conn.fetchval(
                INSERT_CHUNK_SQL,
                content,
                indexable_text,
                embedding,
                doc_name,
                section_path,
                chunk_type,
                api_method,
                endpoint_url,
                sdk_signature,
                language,
                chunk_index,
                json.dumps(metadata or {}),
            )
            return row_id

    async def insert_chunks_batch(self, chunks_data: list[tuple]) -> int:
        """Batch insert chunks for efficiency."""
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            # Use copy for better performance
            count = 0
            for chunk in chunks_data:
                await conn.fetchval(INSERT_CHUNK_SQL, *chunk)
                count += 1
            return count

    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 8,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
    ) -> list[dict]:
        """
        Perform hybrid search combining vector similarity and full-text search.
        """
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(
                HYBRID_SEARCH_SQL,
                query_embedding,
                top_k * 2,  # fetch more candidates for re-ranking
                query_text,
                vector_weight,
                fts_weight,
                top_k,
            )
            return [dict(row) for row in rows]

    async def vector_search(
        self, query_embedding: list[float], top_k: int = 8
    ) -> list[dict]:
        """Pure vector similarity search."""
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(
                VECTOR_SEARCH_SQL, query_embedding, top_k
            )
            return [dict(row) for row in rows]

    async def fts_search(self, query_text: str, top_k: int = 8) -> list[dict]:
        """Pure full-text search."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(FTS_SEARCH_SQL, query_text, top_k)
            return [dict(row) for row in rows]

    async def count_chunks(self) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(COUNT_CHUNKS_SQL)

    async def clear_all(self):
        async with self.pool.acquire() as conn:
            await conn.execute(CLEAR_CHUNKS_SQL)

    # ── File metadata ────────────────────────────────────────────────────

    async def save_file_metadata(self, metadata_dict: dict[str, dict]):
        """Store file-level metadata for pre-filter."""
        async with self.pool.acquire() as conn:
            for doc_name, meta in metadata_dict.items():
                match_text = self._build_match_text(meta)
                await conn.execute(
                    UPSERT_FILE_METADATA_SQL,
                    doc_name,
                    json.dumps(meta),
                    match_text,
                )

    async def get_all_file_metadata(self) -> list[dict]:
        """Retrieve all file metadata for pre-filter."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(GET_ALL_FILE_METADATA_SQL)
            return [
                {
                    "doc_name": row["doc_name"],
                    "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                    "match_text": row["match_text"],
                }
                for row in rows
            ]

    @staticmethod
    def _build_match_text(meta: dict) -> str:
        parts = [
            f"File: {meta.get('doc_name', '')}",
            f"Category: {meta.get('file_category', '')}",
            f"Summary: {meta.get('summary', '')}",
        ]
        if meta.get("keywords"):
            parts.append(f"Keywords: {', '.join(meta['keywords'])}")
        if meta.get("topics"):
            parts.append(f"Topics: {', '.join(meta['topics'])}")
        if meta.get("endpoints"):
            parts.append(f"Endpoints: {', '.join(meta['endpoints'][:20])}")
        if meta.get("sdk_methods"):
            parts.append(f"SDK methods: {', '.join(meta['sdk_methods'])}")
        if meta.get("covered_apis"):
            parts.append(f"Covered APIs: {'; '.join(meta['covered_apis'])}")
        if meta.get("target_queries"):
            parts.append(f"Example questions: {'; '.join(meta['target_queries'])}")
        return "\n".join(parts)

    # ── Filtered search ──────────────────────────────────────────────────

    async def filtered_hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        doc_names: list[str],
        top_k: int = 8,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
    ) -> list[dict]:
        """Hybrid search filtered to specific document files."""
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            rows = await conn.fetch(
                FILTERED_HYBRID_SEARCH_SQL,
                query_embedding,
                top_k * 2,
                query_text,
                vector_weight,
                fts_weight,
                doc_names,
                top_k,
            )
            return [dict(row) for row in rows]
