"""Internal Docker service for indexing and ranking local passage embeddings."""

from contextlib import asynccontextmanager
from hashlib import sha256
from threading import BoundedSemaphore, Event, Thread
from typing import Annotated
from uuid import UUID
import json
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from .db import connection
from .local_embeddings import LocalEmbeddings
from .local_reranker import LocalReranker
from .retrieval import (CANDIDATE_LIMIT, FUSION_DEPTH, RESULT_LIMIT, lexical_query,
                        rank_passages, reciprocal_rank_fusion)


logger = logging.getLogger(__name__)
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "vocab.txt", "special_tokens_map.json"]
EMBEDDING_MODEL = ("intfloat/e5-small-v2", "ffb93f3bd4047442299a41ebb6fa998a38507c52")
RERANKER_MODEL = ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a")


def _snapshot(model):
    from huggingface_hub import snapshot_download

    return snapshot_download(model[0], revision=model[1], allow_patterns=MODEL_FILES)


def _index_batch(embedder):
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT p.document_id, p.ordinal, p.search_text
            FROM lexical_passages p
            LEFT JOIN local_passage_vectors v USING (document_id, ordinal)
            WHERE v.document_id IS NULL
            ORDER BY p.document_id, p.ordinal LIMIT 16
            """
        ).fetchall()
    if not rows:
        return False
    vectors = embedder.encode([row["search_text"] for row in rows], "passage").tolist()
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO local_passage_vectors (document_id, ordinal, embedding)
                SELECT document_id, ordinal, %s::vector
                FROM lexical_passages
                WHERE document_id = %s AND ordinal = %s AND search_text = %s
                FOR KEY SHARE
                ON CONFLICT DO NOTHING
                """,
                [(json.dumps(vector), row["document_id"], row["ordinal"], row["search_text"])
                 for row, vector in zip(rows, vectors, strict=True)],
            )
    return True


def _run(app, stop):
    while not stop.is_set():
        try:
            if app.state.models is None:
                app.state.models = (LocalEmbeddings(_snapshot(EMBEDDING_MODEL)),
                                    LocalReranker(_snapshot(RERANKER_MODEL)))
            if not _index_batch(app.state.models[0]):
                stop.wait(2)
        except Exception as error:
            logger.warning("Local ranking setup or indexing failed (%s)", type(error).__name__)
            stop.wait(15)


@asynccontextmanager
async def lifespan(app):
    app.state.models = None
    app.state.rank_slot = BoundedSemaphore()
    stop = Event()
    task = Thread(target=_run, args=(app, stop), daemon=True)
    task.start()
    yield
    stop.set()
    task.join(timeout=2)


app = FastAPI(title="LexIntel local ranking", lifespan=lifespan)


class RankingRequest(BaseModel):
    matter_id: UUID
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


@app.get("/health")
def health():
    return {"models_ready": app.state.models is not None}


@app.post("/rank")
def rank(request: RankingRequest):
    if app.state.models is None:
        raise HTTPException(503, "Local ranking models are loading")
    if not lexical_query(request.question):
        return {"passages": []}
    if not app.state.rank_slot.acquire(blocking=False):
        raise HTTPException(503, "Local ranking is busy")
    try:
        return _rank(request)
    finally:
        app.state.rank_slot.release()


def _rank(request):
    embedder, reranker = app.state.models
    query_vector = json.dumps(embedder.encode([request.question], "query")[0].tolist())
    with connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        unindexed = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM lexical_passages p
                LEFT JOIN local_passage_vectors v USING (document_id, ordinal)
                WHERE p.matter_id = %s AND v.document_id IS NULL
            ) AS pending
            """, (request.matter_id,),
        ).fetchone()["pending"]
        if unindexed:
            raise HTTPException(503, "Passages are being indexed locally")
        candidates = hybrid_candidates(conn, request.matter_id, request.question, query_vector)
    ranked = reranker.rank(request.question, candidates[:CANDIDATE_LIMIT])
    return {"passages": [
        {
            "document_id": row["document_id"],
            "ordinal": row["ordinal"],
            "fingerprint": sha256(row["passage"]["search_text"].encode("utf-8")).hexdigest(),
        } for row in ranked[:RESULT_LIMIT]
    ]}


def hybrid_candidates(conn, matter_id, question, query_vector, *, document_ids=None):
    lexical = rank_passages(conn, matter_id, question, limit=FUSION_DEPTH, document_ids=document_ids)
    scope = "" if document_ids is None else "AND v.document_id = ANY(%s::uuid[])"
    params = (query_vector, matter_id) if document_ids is None else (query_vector, matter_id, document_ids)
    dense = conn.execute(
        f"""
        WITH nearest AS MATERIALIZED (
            SELECT v.document_id, v.ordinal, v.embedding <=> %s::vector AS distance
            FROM local_passage_vectors v
            JOIN lexical_passages p USING (document_id, ordinal)
            WHERE p.matter_id = %s {scope}
            ORDER BY distance, v.document_id, v.ordinal
            LIMIT %s
        )
        SELECT n.document_id, n.ordinal, a.artifact -> 'chunks' -> n.ordinal AS passage
        FROM nearest n
        JOIN parsed_documents a ON a.document_id = n.document_id
        ORDER BY n.distance, n.document_id, n.ordinal
        """, (*params, FUSION_DEPTH),
    ).fetchall()
    return reciprocal_rank_fusion(lexical, dense, lambda row: (row["document_id"], row["ordinal"]))
