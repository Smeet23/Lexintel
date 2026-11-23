CREATE TABLE IF NOT EXISTS local_passage_vectors (
    document_id uuid NOT NULL,
    ordinal integer NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (document_id, ordinal),
    FOREIGN KEY (document_id, ordinal) REFERENCES lexical_passages (document_id, ordinal) ON DELETE CASCADE
);
