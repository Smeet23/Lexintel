# LexIntel

Legal document analysis organized by matter. The ingestion pipeline converts matter files into versioned text, legal structure, source locations, metadata, and search records.

## Run ingestion locally

```bash
docker compose up --build -d
docker compose ps
```

Docker starts FastAPI, the ingestion worker, and PostgreSQL with pgvector and BM25. The API listens at `http://localhost:8000` and exposes interactive API documentation at `/docs`. The standalone frontend can run with `npm run dev` in `frontend/`. It uses the local API by default. The Compose stack is for local development and uses a loopback API binding.

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/matters \
  -F 'name=Patent fixture' \
  -F 'files=@test_cases/matter1_ip_dispute/complaint_patent_infringement.txt'
```

Create a matter with up to 20 files or upload a single file to an existing matter. PDF, DOCX, and UTF-8 text files up to 50 MiB are accepted. The worker supports native and scanned PDFs, OCR on mixed pages, LibreOffice PDF previews for Word, Docling layout and tables, type-specific metadata, and evidence-backed chunks. The worker may download public Docling model weights on its first PDF job. Docker volumes retain originals, parsed data, and models.

Judgment chunks follow numbered paragraphs. When a long paragraph splits at a sentence boundary, the next chunk can repeat one complete sentence, with up to 250 characters of overlap within the 1,750-character chunk limit. The repeated sentence keeps its original source offsets.

Visit the matter's Documents tab or use these API routes with the IDs returned from upload:

| Route | Output |
| --- | --- |
| `GET /matters/{matter}/documents` | Job status, warnings, and embedding request and token counts |
| `GET /matters/{matter}/documents/{document}/artifact` | Pages and coordinates, profile, sections, tables, references, chunks, and warnings |
| `GET /matters/{matter}/documents/{document}/preview` | PDF or rendered Word preview and plain-text originals |
| `GET /matters/{matter}/chunks?document_id={document}` | Parsed chunks, including when indexing is paused |
| `POST /matters/{matter}/documents/{document}/retry` | Requeue an eligible job using any existing parse and completed embedding batches |

## Embedding cost

Paid indexing is off by default. Parsing, OCR, metadata, and table extraction run locally and still produce a readable artifact. A successfully parsed document stays in `awaiting_index` until indexing is enabled. To enable it, set `LEXINTEL_PAID_INDEXING=1` and provide `OPENROUTER_API_KEY` to the worker in your local environment or ignored `.env`. Recreate the worker with `docker compose up -d worker`, then retry the documents you want indexed.

Indexing uses Voyage 4 Large at 1,024 dimensions. The worker batches distinct search passages, reuses successful batches after retries, and deduplicates identical uploads within a matter. It records successful provider request counts and input tokens. The worker rejects a document if the combined UTF-8 size of its distinct embedding inputs exceeds 2 MB.

## Tests

Run the fast checks from the repository root:

```bash
uv run --locked --project backend --directory backend python -m unittest discover -s tests -v
```

Run the integration suite against a separate Docker project. It exercises the real database, job worker, Docling, Tesseract, and LibreOffice with simulated embeddings.

```bash
docker compose -p lexintel-test run --rm --build --user root -T \
  -e LEXINTEL_INTEGRATION_TESTS=1 -e LEXINTEL_PAID_INDEXING=0 \
  -v "$PWD/backend/tests:/tests:ro" worker \
  sh -c 'uv sync --locked --group parser && python -m unittest discover -s /tests -v'
```

The local native PDF inspector is also available without starting services:

```bash
uv run --locked --project backend --directory backend -m lexintel_ingest /absolute/path/to/document.pdf
```

## Search uploaded evidence

The **Ask AI** tab searches passages from the selected matter with PostgreSQL BM25 and local E5 embeddings, then reranks the combined candidates with local MiniLM. The Docker ranking service downloads public model weights on first start and indexes parsed passages in the background. Search falls back to BM25 while the service is unavailable, busy, or indexing that matter. It shows ranked passages, source pages, and links to the originals or rendered previews. Searches use no provider tokens. Saved chats are scoped to the matter.

For an existing Compose database, apply the retrieval migrations before starting the updated API. The first command backfills BM25 passages. The ranking service fills the local vector table automatically.

```bash
docker compose exec -T db psql -v ON_ERROR_STOP=1 --single-transaction -U lexintel -f - < backend/db/retrieval.sql
docker compose exec -T db psql -v ON_ERROR_STOP=1 --single-transaction -U lexintel -f - < backend/db/local-ranking.sql
```

The [held-out retrieval benchmark](backend/benchmarks/HELDOUT.md) includes the scored comparison and reproduction commands. Set `LEXINTEL_RANKER_URL=` when starting Compose to use BM25 only.
