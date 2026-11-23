"""Compare overlap policies against fixed answer locations using PostgreSQL BM25."""

import argparse
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.request import urlopen
from uuid import NAMESPACE_URL, uuid4, uuid5
import json
import re

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from lexintel_ingest.native_pdf import Page
from lexintel_ingest.retrieval import CANDIDATE_LIMIT, rank_passages, reciprocal_rank_fusion, source_pages
from lexintel_ingest.structure import build_structure


HERE = Path(__file__).resolve().parent
TOP_K = 5
TOKEN = re.compile(r"[^\W_]+")
CONTEXT = {
    "K3": "exclusion of Section 438",
    "B1": "31,41,000",
    "B3": "additional 9% interest",
    "P1": "SmartBot Decision Engine",
}


def tokens(text):
    return [match.group().casefold() for match in TOKEN.finditer(text)]


def load_corpus(directory, documents):
    corpus = {}
    for key, document in documents.items():
        artifact = json.loads((directory / f"{key}.json").read_text())
        if artifact["sha256"] != document["sha256"]:
            raise ValueError(f"Original file hash mismatch for {key}")
        corpus[key] = artifact
    return corpus


def corpus_from_api(base_url, documents):
    base_url = base_url.rstrip("/")

    def fetch(path):
        with urlopen(base_url + path, timeout=15) as response:
            return json.load(response)

    missing = {document["sha256"]: key for key, document in documents.items()}
    corpus = {}
    for matter in fetch("/matters"):
        path = f"/matters/{matter['id']}/documents"
        for document in fetch(path):
            digest = document["sha256"]
            if digest in missing and document["chunk_count"] > 0:
                corpus[missing.pop(digest)] = fetch(f"{path}/{document['id']}/artifact")
        if not missing:
            return {key: corpus[key] for key in documents}
    raise ValueError(f"Upload and parse the missing benchmark documents: {', '.join(missing.values())}")


def label_sources(corpus, questions):
    labeled = []
    for item in questions:
        page = corpus[item["document"]]["pages"][item["page"] - 1]
        words = list(TOKEN.finditer(page["text"]))
        anchor = tokens(item["anchor"])
        starts = [
            pos for pos in range(len(words) - len(anchor) + 1)
            if [word.group().casefold() for word in words[pos:pos + len(anchor)]] == anchor
        ]
        if len(starts) != 1:
            raise ValueError(f"{item['id']}: expected one anchor on page {item['page']}, found {len(starts)}")
        start = starts[0]
        offsets = [(word.start(), word.end()) for word in words[start:start + len(anchor)]]
        labeled.append(item | {"offsets": offsets})
    return labeled


def variant(corpus, overlap):
    result = {}
    for key, artifact in corpus.items():
        if artifact["profile"]["type"] != "judgment":
            result[key] = artifact
            continue
        pages = [
            Page(p["number"], p["label"], p["bounds"], p["rotation"], p["status"],
                 p["text"], p["character_boxes"])
            for p in artifact["pages"]
        ]
        _, text = build_structure(pages, artifact["profile"], artifact["layout_blocks"], judgment_overlap=overlap)
        tables = [chunk for chunk in artifact["chunks"] if chunk["kind"] == "table_row"]
        result[key] = artifact | {"chunks": text + tables}
    return result


def stage_variant(conn, corpus, documents):
    schema = f"bench_{uuid4().hex}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET LOCAL search_path TO {schema}, public")
    for filename in ("schema.sql", "retrieval.sql"):
        conn.execute((HERE.parent / "db" / filename).read_text())
    matter = conn.execute("INSERT INTO matters (name) VALUES ('Retrieval benchmark') RETURNING id").fetchone()["id"]
    ids = {}
    for key, artifact in corpus.items():
        doc = documents[key]
        identifier = conn.execute(
            """INSERT INTO documents (id, matter_id, name, file_type, sha256, original_path)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (uuid5(NAMESPACE_URL, doc["sha256"]), matter, doc["name"],
             artifact["profile"]["file_type"], doc["sha256"], doc["name"]),
        ).fetchone()["id"]
        conn.execute("INSERT INTO parsed_documents (document_id, artifact) VALUES (%s, %s)",
                     (identifier, Jsonb(artifact)))
        ids[identifier] = key
    return matter, ids


def is_gold(item, row):
    passage = row["passage"]
    if row["document_key"] != item["document"] or item["page"] not in source_pages(passage):
        return False
    if "table" in item and (passage.get("table_id"), passage.get("row")) != (item["table"], item["row"]):
        return False
    spans = [span for span in passage["source_spans"] if span["page_number"] == item["page"]]
    if not all(any(s["start"] <= start < end <= s["end"] for s in spans) for start, end in item["offsets"]):
        return False
    context = item.get("context") or CONTEXT.get(item["id"])
    return not context or f" {' '.join(tokens(context))} " in f" {' '.join(tokens(passage['content']))} "


def score(questions, results, corpus):
    hits = []
    repetition = total_characters = top_one = citations = valid_citations = 0
    for item, rows in zip(questions, results, strict=True):
        top = rows[:TOP_K]
        candidate_rank = next(
            (rank for rank, row in enumerate(rows[:CANDIDATE_LIMIT], 1) if is_gold(item, row)),
            None,
        )
        gold_rank = candidate_rank if candidate_rank is not None and candidate_rank <= TOP_K else None
        for row in top:
            pages = corpus[row["document_key"]]["pages"]
            spans = row["passage"]["source_spans"]
            citations += 1
            valid_citations += bool(spans) and all(
                1 <= s["page_number"] <= len(pages)
                and 0 <= s["start"] < s["end"] <= len(pages[s["page_number"] - 1]["text"])
                and pages[s["page_number"] - 1]["text"][s["start"]:s["end"]] == s["text"]
                for s in spans
            )
        positions = Counter(
            (row["document_key"], span["page_number"], offset)
            for row in top
            for span in row["passage"]["source_spans"]
            for offset in range(span["start"], span["end"])
        )
        duplicates = sum(count - 1 for count in positions.values())
        repetition += duplicates
        total_characters += positions.total()
        top_one += gold_rank == 1
        hits.append({
            "id": item["id"], "gold_rank": gold_rank, "candidate_rank": candidate_rank,
            "duplicate_source_characters": duplicates,
            "top": [{
                "document": row["document_key"], "pages": source_pages(row["passage"]),
                "section": row["passage"]["section_name"], "ordinal": row["ordinal"],
            } for row in top],
        })
    found = sum(item["gold_rank"] is not None for item in hits)
    return {
        "questions": len(questions), "evidence_hit_at_5": found, "evidence_hit_at_1": top_one,
        "evidence_hit_at_20": sum(item["candidate_rank"] is not None for item in hits),
        "valid_source_references": valid_citations, "source_references": citations,
        "duplicate_source_characters": repetition,
        "retrieved_source_characters": total_characters,
        "duplicate_source_percent": round(100 * repetition / max(1, total_characters), 2),
        "results": hits,
    }


def semantic_results(model_path, corpus, questions):
    from lexintel_ingest.local_embeddings import LocalEmbeddings

    embed = LocalEmbeddings(model_path)
    rows = [{"document_key": key, "ordinal": ordinal, "passage": passage}
            for key, artifact in corpus.items() for ordinal, passage in enumerate(artifact["chunks"])]
    passages = embed.encode([row["passage"]["search_text"] for row in rows], "passage")
    rankings, seconds = [], []
    for item in questions:
        started = perf_counter()
        query = embed.encode([item["question"]], "query")
        order = (query @ passages.T).argsort(dim=1, descending=True, stable=True).tolist()[0]
        rankings.append([rows[index] for index in order])
        seconds.append(perf_counter() - started)
    return rankings, seconds


def fuse(lexical, dense):
    rankings, seconds = [], []
    for left, right in zip(lexical, dense, strict=True):
        started = perf_counter()
        rankings.append(reciprocal_rank_fusion(left, right, lambda row: (row["document_key"], row["ordinal"])))
        seconds.append(perf_counter() - started)
    return rankings, seconds


def rerank(model_path, questions, ranking_sets):
    from lexintel_ingest.local_reranker import LocalReranker

    model = LocalReranker(model_path)
    reranked = {}
    for name, rankings in ranking_sets.items():
        results, seconds = [], []
        for item, rows in zip(questions, rankings, strict=True):
            started = perf_counter()
            results.append(model.rank(item["question"], rows[:CANDIDATE_LIMIT]))
            seconds.append(perf_counter() - started)
        reranked[name] = (results, seconds)
    return reranked

def latency(seconds):
    return {
        "latency_p50_ms": round(1000 * median(seconds), 2),
        "latency_p95_ms": round(1000 * sorted(seconds)[ceil(len(seconds) * 0.95) - 1], 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifacts", type=Path, help="Directory with the four named JSON artifacts")
    source.add_argument("--api", help="Local LexIntel API URL with the suite documents parsed")
    parser.add_argument("--suite", type=Path, default=HERE / "questions.json", help="Question and document manifest")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--e5", type=Path, help="Optional pre-downloaded E5 model directory")
    parser.add_argument("--reranker", type=Path, help="Optional pre-downloaded MiniLM cross-encoder directory")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    documents = suite["documents"]
    corpus = load_corpus(args.artifacts, documents) if args.artifacts else corpus_from_api(args.api, documents)
    questions = label_sources(corpus, suite["questions"])
    output, lexical_results, query_seconds = {}, {}, {}
    with psycopg.connect(args.dsn, row_factory=dict_row, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")
        for overlap in (False, True):
            label = "sentence_overlap" if overlap else "zero_overlap"
            current = variant(corpus, overlap)
            with conn.transaction(force_rollback=True):
                matter, ids = stage_variant(conn, current, documents)
                rankings, seconds = [], []
                for item in questions:
                    started = perf_counter()
                    rows = rank_passages(conn, matter, item["question"], limit=CANDIDATE_LIMIT)
                    rankings.append([row | {"document_key": ids[row["document_id"]]} for row in rows])
                    seconds.append(perf_counter() - started)
            lexical_results[label] = rankings
            query_seconds[label] = seconds
            output[label] = score(questions, rankings, corpus) | latency(seconds) | {"chunk_count": sum(
                len(artifact["chunks"]) for artifact in current.values()
            )}
    reranker_inputs = {"bm25_reranked": lexical_results["sentence_overlap"]}
    query_seconds["bm25_reranked"] = query_seconds["sentence_overlap"]
    if args.e5:
        dense, dense_seconds = semantic_results(args.e5, variant(corpus, True), questions)
        hybrid, fusion_seconds = fuse(lexical_results["sentence_overlap"], dense)
        hybrid_seconds = [sum(stages) for stages in zip(
            query_seconds["sentence_overlap"], dense_seconds, fusion_seconds, strict=True
        )]
        output["local_e5"] = score(questions, dense, corpus) | latency(dense_seconds)
        output["hybrid"] = score(questions, hybrid, corpus) | latency(hybrid_seconds)
        reranker_inputs["hybrid_reranked"] = hybrid
        query_seconds["hybrid_reranked"] = hybrid_seconds
    if args.reranker:
        for name, (rankings, seconds) in rerank(args.reranker, questions, reranker_inputs).items():
            total = [retrieval + ranking for retrieval, ranking in zip(query_seconds[name], seconds, strict=True)]
            output[name] = score(questions, rankings, corpus) | latency(total)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
