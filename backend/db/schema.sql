CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

CREATE TABLE matters (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    matter_id uuid NOT NULL REFERENCES matters(id) ON DELETE CASCADE,
    name text NOT NULL,
    file_type text NOT NULL CHECK (file_type IN ('pdf', 'docx', 'txt')),
    sha256 text NOT NULL,
    original_path text NOT NULL,
    preview_path text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (matter_id, sha256)
);

CREATE TABLE ingestion_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'queued' CHECK (
        status IN ('queued', 'parsing', 'indexing', 'retrying', 'awaiting_index', 'ready', 'ready_with_warnings', 'failed', 'cancelled')
    ),
    attempts integer NOT NULL DEFAULT 0,
    available_at timestamptz NOT NULL DEFAULT now(),
    lease_token uuid,
    lease_until timestamptz,
    error_code text,
    error_detail text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((lease_token IS NULL) = (lease_until IS NULL))
);

CREATE INDEX ingestion_claim ON ingestion_jobs (available_at, created_at)
    WHERE status IN ('queued', 'retrying', 'parsing', 'indexing');

CREATE TABLE parsed_documents (
    document_id uuid PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    artifact jsonb NOT NULL,
    parsed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE document_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal integer NOT NULL,
    kind text NOT NULL,
    section_name text,
    content text NOT NULL,
    search_text text NOT NULL,
    source_spans jsonb NOT NULL,
    embedding_model text NOT NULL,
    embedding vector(1024) NOT NULL,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX document_chunks_bm25 ON document_chunks USING bm25 (search_text)
    WITH (text_config = 'simple');

CREATE TABLE embedding_cache (
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    model text NOT NULL,
    input_sha256 text NOT NULL,
    embedding vector(1024) NOT NULL,
    PRIMARY KEY (document_id, model, input_sha256)
);

ALTER TABLE ingestion_jobs
    ADD COLUMN embedding_requests integer NOT NULL DEFAULT 0,
    ADD COLUMN embedding_tokens bigint NOT NULL DEFAULT 0;
