"""Run leased ingestion jobs, save parsed artifacts, and index validated chunks."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from uuid import UUID, uuid4
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import psycopg
from psycopg.types.json import Jsonb

from .config import embedding_key, settings
from .db import connection
from .embeddings import Embedder, EmbeddingFailure, MAX_BATCH_ITEMS, MAX_INPUT_BYTES, MODEL, OpenRouterEmbedder

logger = logging.getLogger(__name__)


class LeaseLost(Exception):
    pass


@dataclass(frozen=True)
class Job:
    id: UUID
    document_id: UUID
    token: UUID
    attempts: int
    name: str
    file_type: str
    original_path: str
    sha256: str


def claim_job() -> Job | None:
    token = uuid4()
    with connection() as conn:
        conn.execute(
            """
            UPDATE ingestion_jobs SET status = 'failed', lease_token = NULL, lease_until = NULL,
                error_code = 'attempts_exhausted', error_detail = 'Worker lease expired after the final attempt',
                updated_at = now()
            WHERE status IN ('parsing', 'indexing') AND lease_until < now() AND attempts >= %s
            """,
            (settings.max_attempts,),
        )
        row = conn.execute(
            """
            WITH candidate AS (
                SELECT j.id, p.document_id IS NOT NULL AS already_parsed
                FROM ingestion_jobs j LEFT JOIN parsed_documents p ON p.document_id = j.document_id
                WHERE j.attempts < %s AND (
                   (j.status IN ('queued', 'retrying') AND j.available_at <= now())
                   OR (j.status IN ('parsing', 'indexing') AND j.lease_until < now())
                )
                ORDER BY j.available_at, j.created_at
                LIMIT 1
                FOR UPDATE OF j SKIP LOCKED
            )
            UPDATE ingestion_jobs j
            SET status = CASE WHEN candidate.already_parsed THEN 'indexing' ELSE 'parsing' END,
                lease_token = %s, lease_until = now() + (%s * interval '1 second'),
                attempts = j.attempts + 1, error_code = NULL, error_detail = NULL, updated_at = now()
            FROM candidate, documents d
            WHERE j.id = candidate.id AND d.id = j.document_id
            RETURNING j.id, j.document_id, j.lease_token AS token, j.attempts,
                      d.name, d.file_type, d.original_path, d.sha256
            """,
            (settings.max_attempts, token, settings.lease_seconds),
        ).fetchone()
    return Job(**row) if row else None


def _lease_parameters(job):
    return job.id, job.token


def _require_lease(conn, job) -> None:
    # Use the same document-before-job lock order as deletion.
    document = conn.execute(
        "SELECT id FROM documents WHERE id = %s FOR KEY SHARE", (job.document_id,)
    ).fetchone()
    if document is None:
        raise LeaseLost
    row = conn.execute(
        """
        SELECT id FROM ingestion_jobs WHERE id = %s AND lease_token = %s
            AND lease_until > now() FOR UPDATE
        """,
        _lease_parameters(job),
    ).fetchone()
    if row is None:
        raise LeaseLost


def _heartbeat(job, stop, lost):
    while not stop.wait(15):
        try:
            with connection() as conn:
                cursor = conn.execute(
                    """
                    UPDATE ingestion_jobs SET lease_until = now() + (%s * interval '1 second')
                    WHERE id = %s AND lease_token = %s AND lease_until > now()
                    """,
                    (settings.lease_seconds, *_lease_parameters(job)),
                )
                if not cursor.rowcount:
                    lost.set()
                    return
        except psycopg.Error:
            logger.warning("Could not renew the ingestion lease for job %s", job.id)


def _run_parser(job, directory: Path, lost: Event):
    path = settings.storage / job.original_path
    with path.open("rb") as original:
        if sha256(original.read()).hexdigest() != job.sha256:
            raise EmbeddingFailure("source_changed", "Original file does not match its recorded hash")
    parser_environment = {
        key: value for key, value in os.environ.items()
        if key not in {"OPENROUTER_API_KEY", "OPENROUTER_API_KEY_FILE", "DATABASE_URL"}
    }
    task = subprocess.Popen(
        [sys.executable, "-m", "lexintel_ingest.pipeline", str(path), job.file_type, str(directory)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=parser_environment,
        start_new_session=True,
    )
    deadline = time.monotonic() + settings.parser_timeout
    try:
        while task.poll() is None:
            if lost.wait(.25):
                raise LeaseLost
            if time.monotonic() >= deadline:
                raise EmbeddingFailure("parser_timeout", "Document processing exceeded its time limit", retryable=True)
    finally:
        if task.poll() is None:
            try:
                os.killpg(task.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            task.wait()

    if task.returncode:
        detail = "The parser could not read this document"
        failure = directory / "failure.json"
        if failure.is_file():
            detail = json.loads(failure.read_text())["detail"]
        raise EmbeddingFailure("parse_failed", detail, retryable=task.returncode < 0)

    return json.loads((directory / "artifact.json").read_text())


def _get_artifact(job):
    with connection() as conn:
        row = conn.execute(
            "SELECT artifact FROM parsed_documents WHERE document_id = %s", (job.document_id,)
        ).fetchone()
    return row["artifact"] if row else None


def _save_artifact(job, artifact, directory):
    temporary_preview = artifact.pop("preview_file")
    artifact["sha256"] = job.sha256
    artifact["filename"] = job.name
    preview_path = None
    try:
        with connection() as conn:
            _require_lease(conn, job)
            if temporary_preview:
                preview_path = settings.storage / f"{job.document_id}.{job.token}.preview.pdf"
                os.replace(directory / temporary_preview, preview_path)
                conn.execute(
                    "UPDATE documents SET preview_path = %s WHERE id = %s", (preview_path.name, job.document_id)
                )
            conn.execute(
                "INSERT INTO parsed_documents (document_id, artifact) VALUES (%s, %s)",
                (job.document_id, Jsonb(artifact)),
            )
            conn.execute(
                "UPDATE ingestion_jobs SET status = 'indexing', updated_at = now() WHERE id = %s",
                (job.id,),
            )
    except Exception:
        if preview_path:
            preview_path.unlink(missing_ok=True)
        raise
    return artifact


def _await_key(job):
    with connection() as conn:
        _require_lease(conn, job)
        conn.execute(
            """
            UPDATE ingestion_jobs SET status = 'awaiting_index', lease_token = NULL, lease_until = NULL,
                error_code = 'embedding_key_missing',
                error_detail = 'Parsed document is available. Enable paid indexing and add a key to publish vectors.',
                updated_at = now()
            WHERE id = %s
            """,
            (job.id,),
        )


def _cached_vectors(job):
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT input_sha256, embedding::text AS vector
            FROM embedding_cache WHERE document_id = %s AND model = %s
            """,
            (job.document_id, MODEL),
        ).fetchall()
    return {row["input_sha256"]: row["vector"] for row in rows}


def _save_vectors(job, values, tokens):
    with connection() as conn:
        _require_lease(conn, job)
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO embedding_cache (document_id, model, input_sha256, embedding)
                VALUES (%s, %s, %s, %s::vector)
                ON CONFLICT DO NOTHING
                """,
                [(job.document_id, MODEL, key, vector) for key, vector in values.items()],
            )
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET embedding_requests = embedding_requests + 1, embedding_tokens = embedding_tokens + %s
            WHERE id = %s
            """,
            (tokens, job.id),
        )


def _index(job, artifact, embedder: Embedder, lost):
    chunks = artifact["chunks"]
    keys = [sha256(chunk["search_text"].encode("utf-8")).hexdigest() for chunk in chunks]
    unique = {key: chunk["search_text"] for key, chunk in zip(keys, chunks, strict=True)}
    if sum(len(text.encode("utf-8")) for text in unique.values()) > MAX_INPUT_BYTES:
        raise EmbeddingFailure("embedding_input_limit", "Document exceeds the embedding input budget")

    cached = _cached_vectors(job)
    remaining = [key for key in unique if key not in cached]
    for start in range(0, len(remaining), MAX_BATCH_ITEMS):
        if lost.is_set():
            raise LeaseLost
        batch = remaining[start:start + MAX_BATCH_ITEMS]
        result = embedder.embed([unique[key] for key in batch])
        values = {key: json.dumps(vector, separators=(",", ":"))
                  for key, vector in zip(batch, result.vectors, strict=True)}
        _save_vectors(job, values, result.input_tokens)
        cached.update(values)

    records = [
        (job.document_id, ordinal, chunk["kind"], chunk["section_name"], chunk["content"],
         chunk["search_text"], Jsonb(chunk["source_spans"]), MODEL, cached[key])
        for ordinal, (key, chunk) in enumerate(zip(keys, chunks, strict=True))
    ]
    with connection() as conn:
        _require_lease(conn, job)
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO document_chunks
                    (document_id, ordinal, kind, section_name, content, search_text, source_spans, embedding_model, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                records,
            )
        conn.execute(
            """
            UPDATE ingestion_jobs SET status = %s, lease_token = NULL, lease_until = NULL, updated_at = now()
            WHERE id = %s
            """,
            ("ready_with_warnings" if artifact["warnings"] else "ready", job.id),
        )
        # After publication the search rows own these vectors. Retain the cache
        # only while an unfinished indexing job can need its batches again.
        conn.execute("DELETE FROM embedding_cache WHERE document_id = %s", (job.document_id,))


def _fail(job, error):
    with connection() as conn:
        _require_lease(conn, job)
        retry = error.retryable and job.attempts < settings.max_attempts
        conn.execute(
            """
            UPDATE ingestion_jobs SET status = %s, error_code = %s, error_detail = %s, updated_at = now(),
                available_at = now() + (%s * interval '1 second'), lease_token = NULL, lease_until = NULL
            WHERE id = %s
            """,
            ("retrying" if retry else "failed", error.code, error.detail, 10 * 2 ** (job.attempts - 1), job.id),
        )


def run_once(embedder: Embedder | None = None) -> bool:
    job = claim_job()
    if job is None:
        return False
    stopped, lost = Event(), Event()
    heartbeat = Thread(target=_heartbeat, args=(job, stopped, lost), daemon=True)
    heartbeat.start()
    try:
        artifact = _get_artifact(job)
        if artifact is None:
            with TemporaryDirectory(prefix=".parse-", dir=settings.storage) as raw_directory:
                directory = Path(raw_directory)
                artifact = _save_artifact(job, _run_parser(job, directory, lost), directory)
        if embedder is not None:
            _index(job, artifact, embedder, lost)
        else:
            try:
                key = embedding_key()
            except OSError as error:
                raise EmbeddingFailure("unreadable_key_file", "Embedding key file could not be read") from error
            if key:
                with httpx.Client(
                    base_url="https://openrouter.ai", headers={"Authorization": f"Bearer {key}"},
                    timeout=httpx.Timeout(45, connect=10),
                ) as client:
                    _index(job, artifact, OpenRouterEmbedder(client), lost)
            else:
                _await_key(job)
    except LeaseLost:
        logger.info("Job %s was cancelled or its lease expired", job.id)
    except EmbeddingFailure as error:
        try:
            _fail(job, error)
        except LeaseLost:
            logger.info("Job %s ended after its lease expired", job.id)
        logger.info("Job %s stopped with %s", job.id, error.code)
    except Exception as error:
        try:
            _fail(job, EmbeddingFailure("worker_error", "Unexpected ingestion worker error", retryable=True))
        except (LeaseLost, psycopg.Error):
            pass
        logger.error("Job %s failed with %s", job.id, type(error).__name__)
    finally:
        stopped.set()
        heartbeat.join()
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Ingestion worker started on %s", socket.gethostname())
    while True:
        try:
            if run_once():
                continue
        except psycopg.Error as error:
            logger.warning("Database unavailable to worker: %s", type(error).__name__)
        time.sleep(1)


if __name__ == "__main__":
    main()
