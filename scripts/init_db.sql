-- Enable pgvector and create a seed schema/table for RAG.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS rag;

-- We'll store source rows here (copied/synced from your business DB or inserted directly).
-- 1536 matches OpenAI embedding dims; we can make this configurable later.
CREATE TABLE IF NOT EXISTS rag.source_documents (
  id BIGSERIAL PRIMARY KEY,
  source_id TEXT,            -- e.g., business key from upstream DB
  title TEXT,
  body TEXT,
  metadata JSONB DEFAULT '{}'::jsonb,
  embedding vector(1536)
);

-- Vector index for similarity search (you'll tune ivfflat lists later).
CREATE INDEX IF NOT EXISTS idx_source_documents_embedding
ON rag.source_documents
USING ivfflat (embedding vector_cosine_ops);
