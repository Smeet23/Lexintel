"""Local matter and ingestion endpoints used by the existing frontend."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
import asyncio
import codecs
import json
import os
import re
import time
from uuid import UUID, uuid4
from zipfile import BadZipFile, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .chat import router as chat_router
from .config import settings
from .db import connection

app = FastAPI(title="LexIntel")
app.include_router(chat_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

DOCUMENT_SELECT = """
    SELECT d.*, j.id AS task_id, j.status AS ingestion_status, j.attempts,
           j.error_code, j.error_detail, j.updated_at, j.available_at, j.embedding_requests, j.embedding_tokens,
           p.artifact #>> '{profile,type}' AS document_type,
           p.artifact #>> '{profile,metadata,jurisdiction,value}' AS jurisdiction,
           COALESCE(p.artifact -> 'warnings', '[]'::jsonb) AS warnings,
           COALESCE(jsonb_array_length(p.artifact -> 'chunks'), 0) AS chunk_count
    FROM documents d
    JOIN ingestion_jobs j ON j.document_id = d.id
    LEFT JOIN parsed_documents p ON p.document_id = d.id
"""

ACTIVE = {"queued", "parsing", "indexing", "retrying"}
STATUSES = {
    "queued": ("uploaded", 1, 12, "Document queued"),
    "parsing": ("chunking", 2, 35, "Extracting document content"),
    "retrying": ("chunking", 2, 35, "Waiting to retry processing"),
    "indexing": ("embedding", 3, 75, "Indexing validated passages"),
    "ready": ("ready", 4, 100, "Ingestion complete"),
    "ready_with_warnings": ("ready", 4, 100, "Ingestion complete with warnings"),
    "awaiting_index": ("ready", 4, 100, "Parsing complete. Indexing is paused"),
    "failed": ("error", 4, 100, "Ingestion failed"),
    "cancelled": ("error", 4, 100, "Ingestion cancelled"),
}


@dataclass(frozen=True)
class Upload:
    filename: str
    kind: str
    digest: str
    path: Path


def _validate_format(path: Path, kind: str) -> None:
    with path.open("rb") as stream:
        prefix = stream.read(1024)
    if kind == "pdf":
        if not prefix.startswith(b"%PDF-"):
            raise HTTPException(415, "PDF header is missing")
    elif kind == "docx":
        try:
            with ZipFile(path) as archive:
                entries = archive.infolist()
                names = {item.filename for item in entries}
                if (
                    len(entries) > 2000
                    or sum(item.file_size for item in entries) > 200 * 1024 * 1024
                    or len(names) != len(entries)
                ):
                    raise HTTPException(413, "DOCX archive exceeds supported limits")
                if not {"[Content_Types].xml", "word/document.xml"} <= names:
                    raise HTTPException(415, "Invalid DOCX structure")
        except BadZipFile as error:
            raise HTTPException(415, "Invalid DOCX archive") from error
    else:
        decoder = codecs.getincrementaldecoder("utf-8-sig")()
        try:
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    if b"\0" in block:
                        raise ValueError("NUL in text")
                    decoder.decode(block)
                decoder.decode(b"", final=True)
        except (ValueError, UnicodeDecodeError) as error:
            raise HTTPException(415, "Plain text uploads must be UTF-8") from error


def _stage_file(file: UploadFile, cleanup: ExitStack) -> Upload:
    filename = Path((file.filename or "").replace("\\", "/")).name
    kind = Path(filename).suffix.lower().lstrip(".")
    if kind not in {"pdf", "docx", "txt"}:
        raise HTTPException(415, "Supported formats are PDF, DOCX, and UTF-8 text")
    if len(filename) > 255:
        raise HTTPException(422, "Filename exceeds 255 characters")
    if file.size and file.size > settings.max_upload_bytes:
        raise HTTPException(413, "File exceeds the 50 MiB upload limit")

    settings.storage.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=settings.storage, prefix=".upload-", delete=False) as target:
        path = Path(target.name)
        cleanup.callback(path.unlink, missing_ok=True)
        digest = sha256()
        size = 0
        while block := file.file.read(1024 * 1024):
            size += len(block)
            if size > settings.max_upload_bytes:
                raise HTTPException(413, "File exceeds the 50 MiB upload limit")
            digest.update(block)
            target.write(block)
        if not size:
            raise HTTPException(422, "File is empty")
    _validate_format(path, kind)
    return Upload(filename, kind, digest.hexdigest(), path)


def _require_matter(conn, matter_id: UUID) -> dict:
    row = conn.execute("SELECT m.*, (SELECT count(*) FROM matter_queries q WHERE q.matter_id = m.id) AS queries_count FROM matters m WHERE id = %s", (matter_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Matter not found")
    return row


def _documents(conn, matter_id: UUID):
    return conn.execute(
        DOCUMENT_SELECT + " WHERE d.matter_id = %s ORDER BY d.created_at, d.id", (matter_id,)
    ).fetchall()


def _document(conn, matter_id: UUID, document_id: UUID):
    row = conn.execute(
        DOCUMENT_SELECT + " WHERE d.matter_id = %s AND d.id = %s", (matter_id, document_id)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Document not found")
    return row


def _ui_status(statuses: list[str]) -> str:
    if any(status in ACTIVE for status in statuses):
        return "processing"
    if "failed" in statuses:
        return "error"
    if statuses and all(status == "cancelled" for status in statuses):
        return "cancelled"
    return "ready"


def _matter_response(matter, documents):
    types = {doc["file_type"] for doc in documents}
    return {
        **matter,
        "status": _ui_status([doc["ingestion_status"] for doc in documents]),
        "file_type": "mixed" if len(types) > 1 else next(iter(types), ""),
        "documents_count": len(documents),
        "queries_count": matter.get("queries_count", 0),
        "updated_at": max((doc["updated_at"] for doc in documents), default=None),
    }


def _document_response(doc):
    return {
        key: value for key, value in doc.items() if key not in {"original_path", "preview_path"}
    } | {"status": _ui_status([doc["ingestion_status"]]), "summary": None}


def _record_upload(conn, matter_id: UUID, upload: Upload, created_paths: list[Path]):
    identifier = uuid4()
    basename = f"{identifier}.{upload.kind}"
    row = conn.execute(
        """
        INSERT INTO documents (id, matter_id, name, file_type, sha256, original_path)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (matter_id, sha256) DO NOTHING RETURNING id
        """,
        (identifier, matter_id, upload.filename, upload.kind, upload.digest, basename),
    ).fetchone()
    if not row:
        existing = conn.execute(
            "SELECT id FROM documents WHERE matter_id = %s AND sha256 = %s",
            (matter_id, upload.digest),
        ).fetchone()
        return _document(conn, matter_id, existing["id"])

    path = settings.storage / basename
    os.replace(upload.path, path)
    created_paths.append(path)
    conn.execute("INSERT INTO ingestion_jobs (document_id) VALUES (%s)", (identifier,))
    return _document(conn, matter_id, identifier)


def _save_files(files: list[UploadFile], *, matter_id: UUID | None = None, name: str | None = None):
    if not 1 <= len(files) <= settings.max_upload_files:
        raise HTTPException(422, "Upload between 1 and 20 files")
    created_paths = []
    with ExitStack() as cleanup:
        uploads = [_stage_file(file, cleanup) for file in files]
        try:
            with connection() as conn:
                if matter_id is None:
                    matter = conn.execute(
                        "INSERT INTO matters (name) VALUES (%s) RETURNING *", (name,)
                    ).fetchone()
                    matter_id = matter["id"]
                else:
                    matter = conn.execute(
                        "SELECT * FROM matters WHERE id = %s FOR KEY SHARE", (matter_id,)
                    ).fetchone()
                    if matter is None:
                        raise HTTPException(404, "Matter not found")
                documents = [_record_upload(conn, matter_id, upload, created_paths) for upload in uploads]
        except Exception:
            for path in created_paths:
                path.unlink(missing_ok=True)
            raise
    return matter, list({doc["id"]: doc for doc in documents}.values())


@app.get("/health")
def health():
    with connection() as conn:
        conn.execute("SELECT 1 FROM ingestion_jobs LIMIT 0")
    return {"status": "ok"}


@app.post("/matters", status_code=201)
def create_matter(name: str = Form(..., min_length=1, max_length=200), files: list[UploadFile] = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(422, "Matter name is required")
    matter, documents = _save_files(files, name=name)
    return _matter_response(matter, documents) | {"task_ids": [doc["task_id"] for doc in documents]}


@app.get("/matters")
def list_matters():
    with connection() as conn:
        matters = conn.execute("SELECT m.*, (SELECT count(*) FROM matter_queries q WHERE q.matter_id = m.id) AS queries_count FROM matters m ORDER BY created_at DESC").fetchall()
        return [_matter_response(matter, _documents(conn, matter["id"])) for matter in matters]


@app.get("/matters/{matter_id}")
@app.get("/matters/{matter_id}/status")
def get_matter(matter_id: UUID):
    with connection() as conn:
        return _matter_response(_require_matter(conn, matter_id), _documents(conn, matter_id))


@app.get("/matters/{matter_id}/documents")
def list_documents(matter_id: UUID):
    with connection() as conn:
        _require_matter(conn, matter_id)
        return [_document_response(doc) for doc in _documents(conn, matter_id)]


@app.post("/matters/{matter_id}/documents", status_code=202)
def upload_document(matter_id: UUID, file: UploadFile = File(...)):
    _, documents = _save_files([file], matter_id=matter_id)
    return _document_response(documents[0])


@app.get("/matters/{matter_id}/documents/{document_id}")
def get_document(matter_id: UUID, document_id: UUID):
    with connection() as conn:
        return _document_response(_document(conn, matter_id, document_id))


@app.get("/matters/{matter_id}/documents/{document_id}/artifact")
def get_artifact(matter_id: UUID, document_id: UUID):
    with connection() as conn:
        _document(conn, matter_id, document_id)
        row = conn.execute(
            "SELECT artifact FROM parsed_documents WHERE document_id = %s", (document_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(409, "Document has not been parsed yet")
        artifact = row["artifact"]
        if any(ref["relation"] == "amends_document" for ref in artifact["references"]):
            _link_known_agreements(conn, matter_id, artifact)
    return artifact


def _link_known_agreements(conn, matter_id, artifact):
    candidates = conn.execute(
        """
        SELECT d.id, p.artifact #> '{profile,metadata}' AS metadata, p.artifact -> 'sections' AS sections
        FROM documents d JOIN parsed_documents p ON p.document_id = d.id
        WHERE d.matter_id = %s AND p.artifact #>> '{profile,type}' = 'agreement'
        """,
        (matter_id,),
    ).fetchall()

    def normalize(title):
        return " ".join(re.findall(r"\w+", title.casefold()))

    references = artifact["references"]
    targets = {}
    for reference in references:
        if reference["relation"] != "amends_document" or not reference["target_date"]:
            continue
        matches = []
        for item in candidates:
            metadata = item["metadata"]
            title = metadata["title"]["value"]
            dates = {metadata.get(field, {}).get("value") for field in ("agreement_date", "effective_date")}
            if title and normalize(title) == normalize(reference["target_title"]) and reference["target_date"] in dates:
                matches.append(item)
        if len(matches) == 1:
            target = matches[0]
            reference["target_document_id"] = str(target["id"])
            targets[target["id"]] = target
    if len(targets) == 1:
        target = next(iter(targets.values()))
        for reference in references:
            if reference["relation"] not in {"amends", "deletes", "replaces", "restates"}:
                continue
            matches = [node for node in target["sections"] if node["label"] == reference["target_label"]]
            if len(matches) == 1:
                reference["target_document_id"] = str(target["id"])
                reference["target_node_id"] = matches[0]["id"]


@app.get("/matters/{matter_id}/documents/{document_id}/download")
@app.get("/matters/{matter_id}/documents/{document_id}/preview")
def download_document(matter_id: UUID, document_id: UUID, request: Request):
    with connection() as conn:
        doc = _document(conn, matter_id, document_id)

    if request.url.path.endswith("/preview") and doc["file_type"] == "docx":
        if not doc["preview_path"]:
            raise HTTPException(409, "Document preview is pending")
        return FileResponse(settings.storage / doc["preview_path"], media_type="application/pdf")
    mime = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
    }[doc["file_type"]]
    return FileResponse(
        settings.storage / doc["original_path"],
        media_type=mime,
        filename=doc["name"],
        content_disposition_type="inline",
    )


@app.get("/matters/{matter_id}/chunks")
def list_chunks(matter_id: UUID, document_id: UUID | None = None):
    with connection() as conn:
        _require_matter(conn, matter_id)
        if document_id:
            _document(conn, matter_id, document_id)
        condition = " AND d.id = %s" if document_id else ""
        params = (matter_id, document_id) if document_id else (matter_id,)
        rows = conn.execute(
            """
            SELECT COALESCE(c.id::text, d.id::text || ':' || (entry.ordinal - 1)::text) AS id,
                   entry.chunk ->> 'content' AS content, entry.chunk -> 'source_spans' AS source_spans,
                   entry.chunk ->> 'kind' AS section_type,
                   COALESCE(entry.chunk ->> 'section_name', '') AS section_name,
                   (entry.ordinal - 1) AS chunk_sequence
            FROM documents d
            JOIN parsed_documents p ON p.document_id = d.id
            CROSS JOIN LATERAL jsonb_array_elements(p.artifact -> 'chunks') WITH ORDINALITY AS entry(chunk, ordinal)
            LEFT JOIN document_chunks c ON c.document_id = d.id AND c.ordinal = entry.ordinal - 1
            WHERE d.matter_id = %s
            """ + condition + " ORDER BY d.created_at, d.id, entry.ordinal",
            params,
        ).fetchall()
    return [
        row | {
            "page_num": str(row["source_spans"][0]["page_number"] or ""),
            "concepts": [],
        }
        for row in rows
    ]


@app.post("/matters/{matter_id}/documents/{document_id}/retry", status_code=202)
def retry_document(matter_id: UUID, document_id: UUID):
    with connection() as conn:
        _document(conn, matter_id, document_id)
        row = conn.execute(
            """
            UPDATE ingestion_jobs SET status = 'queued', attempts = 0, available_at = now(),
                error_code = NULL, error_detail = NULL, updated_at = now()
            WHERE document_id = %s AND status IN ('failed', 'awaiting_index', 'cancelled')
            RETURNING id
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(409, "Document is not eligible for retry")
        return _document_response(_document(conn, matter_id, document_id))


@app.post("/matters/{matter_id}/cancel")
def cancel_matter(matter_id: UUID):
    with connection() as conn:
        _require_matter(conn, matter_id)
        cursor = conn.execute(
            """
            UPDATE ingestion_jobs j SET status = 'cancelled', lease_token = NULL, lease_until = NULL,
                updated_at = now()
            FROM documents d
            WHERE j.document_id = d.id AND d.matter_id = %s AND j.status = ANY(%s)
            """,
            (matter_id, list(ACTIVE | {"awaiting_index"})),
        )
        return {"id": matter_id, "cancelled": cursor.rowcount > 0,
                "status": _matter_response({"id": matter_id}, _documents(conn, matter_id))["status"]}


def _remove_files(rows) -> None:
    for row in rows:
        for key in ("original_path", "preview_path"):
            if filename := row[key]:
                (settings.storage / filename).unlink(missing_ok=True)


@app.delete("/matters/{matter_id}/documents/{document_id}")
def delete_document(matter_id: UUID, document_id: UUID):
    with connection() as conn:
        row = conn.execute(
            """
            DELETE FROM documents WHERE id = %s AND matter_id = %s
            RETURNING original_path, preview_path
            """,
            (document_id, matter_id),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Document not found")
    _remove_files([row])
    return {"id": document_id, "deleted": True}


@app.delete("/matters/{matter_id}")
def delete_matter(matter_id: UUID):
    with connection() as conn:
        matter = conn.execute("SELECT id FROM matters WHERE id = %s FOR UPDATE", (matter_id,)).fetchone()
        if matter is None:
            raise HTTPException(404, "Matter not found")
        files = conn.execute(
            "DELETE FROM documents WHERE matter_id = %s RETURNING original_path, preview_path", (matter_id,)
        ).fetchall()
        conn.execute("DELETE FROM matters WHERE id = %s", (matter_id,))
    _remove_files(files)
    return {"id": matter_id, "deleted": True}


def _progress_snapshot(matter_id: UUID):
    with connection() as conn:
        docs = _documents(conn, matter_id)
    counts = Counter(doc["ingestion_status"] for doc in docs)
    if not counts:
        return None

    active = [status for status in counts if status in ACTIVE]
    chosen = min(active, key=lambda status: STATUSES[status][2]) if active else (
        next((status for status in ("failed", "awaiting_index", "ready_with_warnings", "ready", "cancelled")
              if counts[status]), "ready")
    )
    stage, step, _, message = STATUSES[chosen]
    progress = round(sum(STATUSES[status][2] * count for status, count in counts.items()) / len(docs))
    return {
        "stage": stage, "step": step, "total_steps": 4, "progress": progress,
        "overall_progress": progress, "message": message, "current": len(docs) - sum(counts[s] for s in active),
        "total": len(docs), "error": message if stage == "error" else None,
    }


@app.get("/matters/{matter_id}/progress")
async def stream_progress(matter_id: UUID, request: Request):
    await run_in_threadpool(get_matter, matter_id)

    async def events():
        yield 'event: connected\ndata: {}\n\n'
        last = None
        last_send = time.monotonic()
        while not await request.is_disconnected():
            progress = await run_in_threadpool(_progress_snapshot, matter_id)
            if progress is None:
                break
            if progress != last:
                yield f"event: progress\ndata: {json.dumps(progress)}\n\n"
                last, last_send = progress, time.monotonic()
            elif time.monotonic() - last_send >= 15:
                yield ": keepalive\n\n"
                last_send = time.monotonic()
            if progress["stage"] in {"ready", "error"}:
                break
            await asyncio.sleep(1)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
