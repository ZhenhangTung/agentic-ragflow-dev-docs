-- RAGFlow Developer Docs MCP App — Database Setup
-- Run this script once to set up the PostgreSQL database.

-- 1. Create the database (run as superuser)
-- CREATE DATABASE ragflow_docs;

-- 2. Connect to the database and enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. The application creates its own tables on first run.
--    To create them manually, run the following:

CREATE TABLE IF NOT EXISTS doc_chunks (
    id              SERIAL PRIMARY KEY,
    content         TEXT NOT NULL,
    indexable_text  TEXT NOT NULL,
    embedding       vector(1024),
    doc_name        TEXT NOT NULL,
    section_path    TEXT,
    chunk_type      TEXT,
    api_method      TEXT,
    endpoint_url    TEXT,
    sdk_signature   TEXT,
    language        TEXT,
    chunk_index     INTEGER NOT NULL,
    metadata        JSONB DEFAULT '{}',
    fts_vector      tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(endpoint_url, '') || ' ' || coalesce(sdk_signature, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(section_path, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'C')
    ) STORED,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
    ON doc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_fts
    ON doc_chunks USING gin (fts_vector);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_name
    ON doc_chunks (doc_name);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_chunk_type
    ON doc_chunks (chunk_type);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_api_method
    ON doc_chunks (api_method);
