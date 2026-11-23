"""Rank matter passages and return the exact evidence available to the reader."""

from collections import Counter
from unicodedata import name
from uuid import UUID
import logging
import re

import httpx

from .config import settings

logger = logging.getLogger(__name__)
RESULT_LIMIT = 5
CANDIDATE_LIMIT = 20
FUSION_DEPTH = 15

STOP_WORDS = frozenset("""
a an and are as at be been before by could did do does for from had has have how in into is it
its may of on or should than that the their them there these they this to was were what when
where which who why will with would you
""".split())


def lexical_query(question):
    words = re.findall(r"[^\W_]+(?:[.’'-][^\W_]+)*", question.casefold())
    return " ".join(dict.fromkeys(word for word in words if word not in STOP_WORDS))


def rank_passages(conn, matter_id, question, limit=5, *, document_ids=None):
    query = lexical_query(question)
    if not query:
        return []
    scope = "" if document_ids is None else "AND document_id = ANY(%s::uuid[])"
    params = (query, matter_id) if document_ids is None else (query, matter_id, document_ids)
    return conn.execute(
        f"""
        WITH ranked AS (
            SELECT document_id, ordinal,
                   search_text <@> to_bm25query(%s, 'lexical_passages_bm25') AS distance
            FROM lexical_passages
            WHERE matter_id = %s {scope}
            ORDER BY distance, document_id, ordinal
            LIMIT %s
        )
        SELECT ranked.*, d.name AS document_name, d.file_type,
               p.artifact -> 'chunks' -> ranked.ordinal AS passage
        FROM ranked
        JOIN documents d ON d.id = ranked.document_id
        JOIN parsed_documents p ON p.document_id = ranked.document_id
        WHERE ranked.distance < 0
        ORDER BY ranked.distance, ranked.document_id, ranked.ordinal
        """,
        (*params, limit),
    ).fetchall()


def reciprocal_rank_fusion(lexical, dense, key):
    scores, rows = Counter(), {}
    for ranking in (lexical, dense):
        for rank, row in enumerate(ranking[:FUSION_DEPTH], 1):
            identity = key(row)
            scores[identity] += 1 / (60 + rank)
            rows[identity] = row
    return [rows[identity] for identity, _ in scores.most_common()]


def _ranked_references(matter_id, question):
    with httpx.Client(timeout=httpx.Timeout(3.0, connect=0.4), trust_env=False) as client:
        response = client.post(f"{settings.ranker_url.rstrip('/')}/rank",
                               json={"matter_id": str(matter_id), "question": question})
        response.raise_for_status()
    passages = response.json()["passages"]
    if not isinstance(passages, list) or len(passages) > RESULT_LIMIT:
        raise ValueError("Unexpected ranking response")
    references = [(UUID(row["document_id"]), row["ordinal"], row["fingerprint"]) for row in passages]
    if any(
        type(ordinal) is not int or not 0 <= ordinal < 2**31 or not re.fullmatch("[a-f0-9]{64}", fingerprint)
        for _, ordinal, fingerprint in references
    ):
        raise ValueError("Invalid passage reference")
    if len({(identifier, ordinal) for identifier, ordinal, _ in references}) != len(references):
        raise ValueError("Duplicate passage")
    return references


def search_passages(conn, matter_id, question):
    compatible_script = all(not char.isalpha() or name(char, "").startswith("LATIN") for char in question)
    if settings.ranker_url and compatible_script:
        try:
            references = _ranked_references(matter_id, question)
        except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError) as error:
            if not isinstance(error, httpx.HTTPStatusError) or error.response.status_code != 503:
                logger.warning("Local ranking unavailable (%s)", type(error).__name__)
        else:
            if not references:
                return [], "hybrid_reranked"
            ids, ordinals, fingerprints = zip(*references)
            rows = conn.execute(
                """
                SELECT d.id AS document_id, d.name AS document_name, d.file_type, rank.ordinal,
                       a.artifact -> 'chunks' -> rank.ordinal AS passage
                FROM unnest(%s::uuid[], %s::integer[], %s::text[]) WITH ORDINALITY
                    AS rank(document_id, ordinal, fingerprint, position)
                JOIN lexical_passages p ON p.document_id = rank.document_id AND p.ordinal = rank.ordinal
                JOIN documents d ON d.id = rank.document_id
                JOIN parsed_documents a ON a.document_id = d.id
                WHERE d.matter_id = %s AND encode(sha256(convert_to(p.search_text, 'UTF8')), 'hex') = rank.fingerprint
                ORDER BY rank.position
                """,
                (list(ids), list(ordinals), list(fingerprints), matter_id),
            ).fetchall()
            if len(rows) == len(references):
                return rows, "hybrid_reranked"
    return rank_passages(conn, matter_id, question, limit=RESULT_LIMIT), "bm25"


def source_pages(passage):
    return sorted({span["page_number"] for span in passage["source_spans"]})


def serialize_source(row, method):
    passage = row["passage"]
    pages = source_pages(passage)
    identifier = row["document_id"]
    page_label = str(pages[0]) if len(pages) == 1 else f"{pages[0]}–{pages[-1]}"
    path = f"documents/{identifier}/preview"
    fragment = "" if row["file_type"] == "txt" else f"#page={pages[0]}"
    return {
        "chunk_id": f"{identifier}:{row['ordinal']}",
        "document_id": str(identifier),
        "document_name": row["document_name"],
        "page_num": str(pages[0]),
        "page_label": "" if row["file_type"] == "txt" else page_label,
        "pages": pages,
        "section_name": passage["section_name"] or "",
        "source_type": "document",
        "source_path": path + fragment,
        "content": passage["content"],
        "retrieval_method": method,
        "source_spans": passage["source_spans"],
    }
