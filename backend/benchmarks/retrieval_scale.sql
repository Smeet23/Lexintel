-- Replicate existing passages in isolation to exercise the database at a larger index size.
-- All changes are rolled back. Keep the six-document and 62-document benchmark matters loaded.
BEGIN;
CREATE TEMP TABLE scale_passages ON COMMIT DROP AS
SELECT CASE WHEN copies.n = 0 THEN p.matter_id
            ELSE md5(p.matter_id::text || ':' || copies.n)::uuid END AS matter_id,
       p.document_id, p.ordinal, p.search_text, v.embedding
FROM lexical_passages p
JOIN local_passage_vectors v USING (document_id, ordinal)
CROSS JOIN generate_series(0, 15) AS copies(n)
WHERE p.matter_id IN (:'oracle_matter'::uuid, :'heldout_matter'::uuid);

CREATE INDEX scale_by_matter ON scale_passages (matter_id, document_id, ordinal);
CREATE INDEX scale_bm25 ON scale_passages USING bm25 (search_text) WITH (text_config = 'simple');
ANALYZE scale_passages;
SELECT count(*) AS indexed_passages, count(DISTINCT matter_id) AS matters FROM scale_passages;

SELECT v.embedding::text AS sample_vector
FROM local_passage_vectors v
JOIN lexical_passages p USING (document_id, ordinal)
WHERE p.matter_id = :'oracle_matter'::uuid
ORDER BY p.document_id, p.ordinal LIMIT 1
\gset

EXPLAIN (ANALYZE, SUMMARY)
SELECT document_id, ordinal
FROM scale_passages
WHERE matter_id = :'oracle_matter'::uuid
ORDER BY search_text <@> to_bm25query('jury android declaring code verdict', 'scale_bm25'),
         document_id, ordinal
LIMIT 20;

EXPLAIN (ANALYZE, SUMMARY)
SELECT document_id, ordinal
FROM scale_passages
WHERE matter_id = :'oracle_matter'::uuid
ORDER BY embedding <=> :'sample_vector'::vector, document_id, ordinal
LIMIT 20;

ROLLBACK;
