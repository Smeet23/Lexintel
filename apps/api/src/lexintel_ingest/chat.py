"""Persist matter-scoped evidence search conversations."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from psycopg.types.json import Jsonb
from pydantic import BaseModel, StringConstraints

from .db import connection
from .retrieval import search_passages, serialize_source


router = APIRouter(prefix="/matters/{matter_id}")


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    conversation_id: UUID | None = None
    include_legal_research: bool = False


def _matter(conn, matter_id):
    if not conn.execute("SELECT 1 FROM matters WHERE id = %s FOR KEY SHARE", (matter_id,)).fetchone():
        raise HTTPException(404, "Matter not found")


def _conversation(conn, matter_id, conversation_id):
    row = conn.execute(
        "SELECT * FROM conversations WHERE matter_id = %s AND id = %s FOR KEY SHARE",
        (matter_id, conversation_id),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Conversation not found")
    return row


def _new_conversation(conn, matter_id):
    return conn.execute(
        "INSERT INTO conversations (matter_id) VALUES (%s) RETURNING id, created_at",
        (matter_id,),
    ).fetchone()


@router.post("/ask")
def ask(matter_id: UUID, request: AskRequest):
    if request.include_legal_research:
        raise HTTPException(400, "Search is available for documents in this matter.")
    with connection() as conn:
        _matter(conn, matter_id)
        if request.conversation_id:
            conversation_id = _conversation(conn, matter_id, request.conversation_id)["id"]
        else:
            conversation_id = _new_conversation(conn, matter_id)["id"]
        ranked, method = search_passages(conn, matter_id, request.question)
        sources = [serialize_source(row, method) for row in ranked]
        answer = (
            f"{len(sources)} matching {'passage' if len(sources) == 1 else 'passages'} from this matter."
            if sources else "No matching passages in this matter. Try a case name, clause number, or phrase from the document."
        )
        query_id = conn.execute(
            """
            INSERT INTO matter_queries (matter_id, conversation_id, question, answer, citations)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """,
            (matter_id, conversation_id, request.question, answer, Jsonb(sources)),
        ).fetchone()["id"]
    return {
        "answer": answer, "sources": sources, "citations": [], "matter_id": matter_id,
        "query": request.question, "query_id": query_id, "conversation_id": conversation_id,
        "model": method, "tokens_used": 0,
    }


@router.get("/conversations")
def list_conversations(matter_id: UUID):
    with connection() as conn:
        _matter(conn, matter_id)
        return conn.execute(
            """
            SELECT c.id, c.created_at, COALESCE(MAX(q.created_at), c.created_at) AS updated_at,
                   COUNT(q.id) AS message_count,
                   (array_agg(left(q.question, 80) ORDER BY q.created_at, q.id))[1] AS title,
                   (array_agg(left(q.question, 120) ORDER BY q.created_at DESC, q.id DESC))[1] AS last_message_preview
            FROM conversations c LEFT JOIN matter_queries q ON q.conversation_id = c.id
            WHERE c.matter_id = %s
            GROUP BY c.id
            ORDER BY updated_at DESC, c.id
            """,
            (matter_id,),
        ).fetchall()


@router.post("/conversations", status_code=201)
def create_conversation(matter_id: UUID):
    with connection() as conn:
        _matter(conn, matter_id)
        return _new_conversation(conn, matter_id) | {"title": None}


@router.get("/conversations/{conversation_id}")
def get_conversation(matter_id: UUID, conversation_id: UUID):
    with connection() as conn:
        row = _conversation(conn, matter_id, conversation_id)
        queries = conn.execute(
            "SELECT id, question, answer, citations, created_at FROM matter_queries "
            "WHERE matter_id = %s AND conversation_id = %s ORDER BY created_at, id",
            (matter_id, conversation_id),
        ).fetchall()
    return row | {
        "title": queries[0]["question"][:80] if queries else None,
        "updated_at": queries[-1]["created_at"] if queries else row["created_at"],
        "queries": queries,
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(matter_id: UUID, conversation_id: UUID):
    with connection() as conn:
        result = conn.execute("DELETE FROM conversations WHERE matter_id = %s AND id = %s",
                              (matter_id, conversation_id))
        if not result.rowcount:
            raise HTTPException(404, "Conversation not found")
    return {"id": conversation_id, "deleted": True}


@router.get("/queries")
def list_queries(matter_id: UUID, limit: Annotated[int, Query(ge=1, le=200)] = 50):
    with connection() as conn:
        _matter(conn, matter_id)
        return conn.execute(
            "SELECT id, question, answer, citations, created_at FROM matter_queries "
            "WHERE matter_id = %s ORDER BY created_at DESC, id DESC LIMIT %s", (matter_id, limit)
        ).fetchall()


@router.delete("/queries/{query_id}")
def delete_query(matter_id: UUID, query_id: UUID):
    with connection() as conn:
        result = conn.execute("DELETE FROM matter_queries WHERE matter_id = %s AND id = %s",
                              (matter_id, query_id))
        if not result.rowcount:
            raise HTTPException(404, "Query not found")
    return {"id": query_id, "deleted": True}


@router.delete("/queries")
def delete_queries(matter_id: UUID):
    with connection() as conn:
        _matter(conn, matter_id)
        result = conn.execute("DELETE FROM matter_queries WHERE matter_id = %s", (matter_id,))
    return {"matter_id": matter_id, "deleted_count": result.rowcount}
