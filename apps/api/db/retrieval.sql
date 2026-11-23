CREATE TABLE IF NOT EXISTS lexical_passages (
    document_id uuid NOT NULL REFERENCES parsed_documents(document_id) ON DELETE CASCADE,
    matter_id uuid NOT NULL REFERENCES matters(id) ON DELETE CASCADE,
    ordinal integer NOT NULL,
    search_text text NOT NULL,
    PRIMARY KEY (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS lexical_passages_matter ON lexical_passages (matter_id);
CREATE INDEX IF NOT EXISTS lexical_passages_bm25 ON lexical_passages USING bm25 (search_text)
    WITH (text_config = 'simple');

CREATE OR REPLACE FUNCTION refresh_lexical_passages() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM lexical_passages WHERE document_id = NEW.document_id;
    INSERT INTO lexical_passages (document_id, matter_id, ordinal, search_text)
    SELECT NEW.document_id, d.matter_id, entry.ordinal - 1, entry.chunk ->> 'search_text'
    FROM documents d
    CROSS JOIN LATERAL jsonb_array_elements(NEW.artifact -> 'chunks') WITH ORDINALITY AS entry(chunk, ordinal)
    WHERE d.id = NEW.document_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS parsed_documents_lexical ON parsed_documents;
CREATE TRIGGER parsed_documents_lexical AFTER INSERT OR UPDATE OF artifact ON parsed_documents
    FOR EACH ROW EXECUTE FUNCTION refresh_lexical_passages();

INSERT INTO lexical_passages (document_id, matter_id, ordinal, search_text)
SELECT p.document_id, d.matter_id, entry.ordinal - 1, entry.chunk ->> 'search_text'
FROM parsed_documents p JOIN documents d ON d.id = p.document_id
CROSS JOIN LATERAL jsonb_array_elements(p.artifact -> 'chunks') WITH ORDINALITY AS entry(chunk, ordinal)
ON CONFLICT (document_id, ordinal) DO NOTHING;

CREATE TABLE IF NOT EXISTS conversations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    matter_id uuid NOT NULL REFERENCES matters(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (matter_id, id)
);

CREATE TABLE IF NOT EXISTS matter_queries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    matter_id uuid NOT NULL REFERENCES matters(id) ON DELETE CASCADE,
    conversation_id uuid,
    question text NOT NULL,
    answer text NOT NULL,
    citations jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (matter_id, conversation_id) REFERENCES conversations (matter_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS matter_queries_history ON matter_queries (matter_id, created_at DESC);
