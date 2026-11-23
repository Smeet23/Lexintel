"""Compare filing and passage labels using the live local indexes in read-only mode."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from uuid import UUID

from huggingface_hub import snapshot_download

from lexintel_ingest.db import connection
from lexintel_ingest.local_embeddings import LocalEmbeddings
from lexintel_ingest.local_reranker import LocalReranker
from lexintel_ingest.rank_service import EMBEDDING_MODEL, MODEL_FILES, RERANKER_MODEL, hybrid_candidates
from lexintel_ingest.retrieval import CANDIDATE_LIMIT, source_pages
from retrieval import is_gold, label_sources, latency, score


SCOPED_LIMIT = 8


@dataclass(frozen=True)
class Filing:
    kind: str = ""
    court: str = ""
    year: str = ""
    party: str = ""
    docket_entry: str = ""


@dataclass(frozen=True)
class Intent:
    kinds: tuple = ()
    court: str = ""
    year: str = ""
    party: str = ""
    role: str = ""


def court_in(text):
    for term in ("supreme court", "federal circuit", "district court"):
        if term in text:
            return term
    return ""


def identify_filing(document):
    name = document["name"].casefold()
    title = (document["title"] or "").casefold()
    opening = document["pages"][0]["text"][:850].casefold() if document["pages"] else ""
    if re.search(r"\b(?:appendix|appendices|exhibits|docket excerpts)\b", name):
        # Multi-filing packets require a page range for each component.
        return Filing()
    for kind, pattern in (
        ("instructions", r"jury (?:instructions|charge)|final charge to the jury"),
        ("verdict", r"jury verdict|special verdict form"),
        ("judgment", r"final judgment"),
        ("transcript", r"\btranscript\b"),
        ("reply", r"\breply to (?:amended )?counterclaims\b"),
        ("answer", r"\banswer\b"),
        ("amended_complaint", r"\bamended complaint\b"),
        ("complaint", r"\bcomplaint\b"),
        ("merits_brief", r"\bmerits brief\b"),
        ("brief", r"\bbrief\b"),
        ("petition", r"\bpetition\b"),
        ("order", r"\border\b"),
        ("opinion", r"\bopinion\b"),
    ):
        if re.search(pattern, name):
            break
    else:
        kind = "opinion" if document["type"] == "judgment" else ""
        if title.startswith(("answer ", "complaint ")):
            kind = title.split()[0]
    date = re.search(r"^(?:ja\d - )?((?:19|20)\d\d)(?!\d)", name)
    entry = re.search(r"\becf (\d+)\b", name)
    party = re.search(r"\becf \d+ ([\w]+)|^\d{4}-\d{2}-\d{2} ([\w]+)", name)
    court = court_in(name) or court_in(opening)
    if "merits brief" in name or "certiorari" in name:
        court = "supreme court"
    if entry:
        court = "district court"
    filer = next((group for group in party.groups() if group), "") if party else ""
    return Filing(kind, court, date[1] if date else "", filer, entry[1] if entry else "")


def route(question):
    text = question.casefold()
    year = re.search(r"\b(?:19|20)\d\d\b", text)
    date = year[0] if year else ""
    court = court_in(text)
    if re.search(r"\bblank verdict form\b", text):
        return Intent(("instructions",), "district court", date, role="blank_form")
    if re.search(r"jury (?:instructions|charge)|instruct.{0,25}jury", text):
        return Intent(("instructions",), "district court", date, role="instruction")
    if re.search(r"\bjury\b", text) and not re.search(r"final judgment|new trial", text):
        return Intent(("verdict",), "district court", date, role="verdict")
    if "final judgment" in text:
        return Intent(("judgment",), court, date, role="disposition")
    if "new trial" in text and "motion" in text:
        return Intent(("order",), court, date, role="disposition")
    if "merits brief" in text:
        party = re.search(r"\bdid (\w+) (?:ask|say|argue|request)\b|\bin (\w+)[’']s", text)
        filer = next((group for group in party.groups() if group), "") if party else ""
        role = "requested_disposition" if re.search(r"\b(?:ask|request|disposition)\b", text) else "submission"
        return Intent(("merits_brief",), "supreme court", date, filer, role)
    if re.search(r"\b(?:in (?:its|his|her|their) own answer|assert in .* answer)\b", text):
        return Intent(("answer",), court, date, role="submission")
    if "amended complaint" in text:
        return Intent(("amended_complaint",), court, date, role="submission")
    if re.search(r"\b(?:complaint|allege[ds]?)\b", text):
        return Intent(("complaint",), court, date, role="submission")
    if re.search(r"\bdissent\b", text):
        return Intent(("opinion",), court, date, role="dissent")
    if court and re.search(r"\b(?:hold|held|rule|decide|ruling|send back)\b", text):
        return Intent(("opinion", "order"), court, date, role="holding")
    return Intent()


def filing_bonus(filing, intent):
    if not intent.kinds:
        return 0
    kind = 5 if filing.kind in intent.kinds else 0
    court = 2 if intent.court and filing.court == intent.court else 0
    year = 3 if intent.year and filing.year == intent.year else 0
    party = 4 if intent.party and filing.party == intent.party else 0
    return kind + court + year + party


def scoped_ids(filings, intent):
    return [
        identifier for identifier, filing in filings.items()
        if filing.kind in intent.kinds
        and (not intent.court or filing.court == intent.court)
        and (not intent.year or filing.year == intent.year)
        and (not intent.party or filing.party == intent.party)
    ]


def page_role(page, filing):
    text = " ".join(page.split())
    if filing.kind == "instructions":
        if "SPECIAL VERDICT FORM" in text[:360] and re.search(r"\byes\s+_{3,}", text, re.I):
            return "blank_form"
        return "instruction"
    if filing.kind == "verdict":
        return "verdict"
    if filing.kind == "opinion":
        head = text[:220].casefold()
        if "syllabus" in head:
            return "syllabus"
        if re.search(r"\b(?:j\.,|justice \w+,) dissenting\b", head):
            return "dissent"
        if "opinion of the court" in head:
            return "majority"
    return ""


def passage_role(passage, filing, page_roles):
    section = (passage["section_name"] or "").casefold()
    body = " ".join(passage["content"].split()).casefold()
    roles = {page_roles[number] for number in source_pages(passage)}
    if "blank_form" in roles and re.search(r"\byes\s+_{3,}", body):
        return "blank_form"
    if len(roles) == 1 and (role := roles.pop()) not in {"", "majority"}:
        return role
    if re.search(r"\.{5,}|table of (?:contents|authorities)", body[:420]):
        return "front_matter"
    if filing.kind in {"brief", "merits_brief", "complaint", "amended_complaint", "answer", "petition", "reply"}:
        if re.search(r"\b(?:this court should (?:affirm|reverse)|respectfully requests that the judgment)\b", body):
            return "requested_disposition"
        return "submission"
    if filing.kind in {"opinion", "judgment", "order"}:
        if (
            re.search(r"\b(?:conclusion|court disposition)\b", section)
            or re.search(
                r"\b(?:judgment is hereby entered|it is (?:so ordered|ordered accordingly)|"
                r"motions? (?:is|are) denied|we (?:hold|conclude|therefore reverse)|"
                r"are hereby quashed|judgment is reversed)\b", body
            )
        ):
            return "holding" if filing.kind == "opinion" else "disposition"
        if re.search(r"\b(?:submissions|arguments)\b", section):
            return "submission"
    return ""


def passage_bonus(role, intent):
    if not intent.role:
        return 0
    if role == intent.role or intent.role == "holding" and role == "disposition":
        return 7
    if role in {"front_matter", "syllabus", "dissent", "blank_form"}:
        return -8
    return 0


def document_rows(conn, matter):
    return {
        row["id"]: row for row in conn.execute(
            """
            SELECT d.id, d.name, d.sha256,
                   a.artifact #>> '{profile,type}' AS type,
                   a.artifact #>> '{profile,metadata,title,value}' AS title,
                   COALESCE((SELECT jsonb_agg(jsonb_build_object('text', p->'text') ORDER BY n)
                    FROM jsonb_array_elements(a.artifact->'pages') WITH ORDINALITY AS pages(p,n)), '[]') AS pages
            FROM documents d JOIN parsed_documents a ON a.document_id = d.id
            WHERE d.matter_id = %s
            """, (matter,),
        )
    }


def gold_sources(corpus, questions):
    result = {}
    for question in questions:
        inherited = {key: value for key, value in question.items() if key != "literal"}
        acceptable = label_sources(corpus, [question, *[
            inherited | alternate for alternate in question.get("alternates", [])
        ]])
        for item in acceptable:
            if literal := item.get("literal"):
                text = corpus[item["document"]]["pages"][item["page"] - 1]["text"]
                if text.count(literal) != 1:
                    raise ValueError(f"{item['id']}: expected one verdict marker")
                start = text.index(literal)
                item["offsets"].append((start, start + len(literal)))
        result[question["id"]] = acceptable
    return result


def identify_indexed_evidence(conn, keys, labels):
    indexed = {key: [] for key in keys.values()}
    for row in conn.execute(
        """
        SELECT p.document_id, p.ordinal, a.artifact -> 'chunks' -> p.ordinal AS passage
        FROM lexical_passages p JOIN parsed_documents a USING (document_id)
        WHERE p.document_id = ANY(%s::uuid[])
        """, (list(keys),),
    ):
        key = keys[row["document_id"]]
        indexed[key].append(row | {"document_key": key})
    return {
        qid: any(is_gold(item, row) for item in sources for row in indexed[item["document"]])
        for qid, sources in labels.items()
    }


def attach_ranking_metadata(ranked, keys, filings, page_roles, intent):
    evaluated = []
    for position, row in enumerate(ranked):
        identifier = row["document_id"]
        filing = filings[identifier]
        role = passage_role(row["passage"], filing, page_roles[identifier])
        weight = filing_bonus(filing, intent) - position
        evaluated.append(row | {
            "document_key": keys.get(identifier, str(identifier)), "filing": filing.kind, "role": role,
            "metadata_score": weight, "role_score": weight + passage_bonus(role, intent),
        })
    return evaluated


def candidate_identity(row):
    return row["document_id"], row["ordinal"]


def snapshot(model):
    return snapshot_download(model[0], revision=model[1], allow_patterns=MODEL_FILES, local_files_only=True)


def evaluate(conn, matter, suite, embedder, reranker):
    documents = document_rows(conn, matter)
    identities = {document["sha256"]: identifier for identifier, document in documents.items()}
    keys = {identities[doc["sha256"]]: key for key, doc in suite["documents"].items()}
    corpus = {keys.get(identifier, str(identifier)): document for identifier, document in documents.items()}
    labels = gold_sources(corpus, suite["questions"])
    filings = {identifier: identify_filing(doc) for identifier, doc in documents.items()}
    page_roles = {
        identifier: {n: page_role(page["text"], filings[identifier]) for n, page in enumerate(doc["pages"], 1)}
        for identifier, doc in documents.items()
    }
    coverage = conn.execute(
        """
        SELECT count(*) AS passages, count(v.embedding) AS indexed
        FROM lexical_passages p LEFT JOIN local_passage_vectors v USING (document_id, ordinal)
        WHERE p.matter_id = %s
        """, (matter,),
    ).fetchone()
    if not coverage["passages"] or coverage["passages"] != coverage["indexed"]:
        raise ValueError("The selected matter needs its complete local passage index.")

    indexed = identify_indexed_evidence(conn, keys, labels)
    phase_results = {"baseline": [], "filing_metadata": [], "filing_and_passage": [], "scoped_candidates": []}
    durations = {key: [] for key in phase_results}
    for question in suite["questions"]:
        started = perf_counter()
        text = question["question"]
        query_vector = json.dumps(embedder.encode([text], "query")[0].tolist())
        candidates = hybrid_candidates(conn, matter, text, query_vector)[:CANDIDATE_LIMIT]
        query_seconds = perf_counter() - started
        baseline = reranker.rank(text, candidates)
        baseline_seconds = perf_counter() - started

        started = perf_counter()
        intent = route(text)
        evaluated = attach_ranking_metadata(baseline, keys, filings, page_roles, intent)
        phase_results["baseline"].append(evaluated)
        phase_results["filing_metadata"].append(sorted(evaluated, key=lambda row: -row["metadata_score"]))
        phase_results["filing_and_passage"].append(sorted(evaluated, key=lambda row: -row["role_score"]))
        metadata_seconds = perf_counter() - started
        durations["baseline"].append(baseline_seconds)
        durations["filing_metadata"].append(baseline_seconds + metadata_seconds)
        durations["filing_and_passage"].append(baseline_seconds + metadata_seconds)

        started = perf_counter()
        preferred = scoped_ids(filings, intent)
        already_seen = {candidate_identity(row) for row in candidates}
        scoped = hybrid_candidates(conn, matter, text, query_vector, document_ids=preferred) if preferred else []
        additions = [row for row in scoped if candidate_identity(row) not in already_seen][:SCOPED_LIMIT]
        if additions:
            expanded = reranker.rank(text, candidates + additions)
            evaluated = attach_ranking_metadata(expanded, keys, filings, page_roles, intent)
        phase_results["scoped_candidates"].append(sorted(evaluated, key=lambda row: -row["role_score"]))
        unchanged_seconds = 0 if additions else baseline_seconds - query_seconds + metadata_seconds
        durations["scoped_candidates"].append(query_seconds + unchanged_seconds + perf_counter() - started)

    def measure(rankings):
        for question, rows in zip(suite["questions"], rankings, strict=True):
            gold = labels[question["id"]]
            yield rows, next((item for row in rows for item in gold if is_gold(item, row)), gold[0])

    results = {}
    for key, rankings in phase_results.items():
        rows, selected_gold = zip(*measure(rankings), strict=True)
        measured = score(selected_gold, rows, corpus) | latency(durations[key])
        with_evidence = 0
        for item, ranked, gold in zip(measured["results"], rows, selected_gold, strict=True):
            rank = next((rank for rank, row in enumerate(ranked, 1) if is_gold(gold, row)), None)
            with_evidence += rank is not None
            item.update(pool_size=len(ranked), pool_evidence_rank=rank, indexed_evidence=indexed[item["id"]])
            for reference, row in zip(item["top"], ranked[:5], strict=True):
                reference.update(name=documents[row["document_id"]]["name"], filing=row["filing"], role=row["role"])
        measured["evidence_in_candidate_pool"] = with_evidence
        results[key] = measured
    return {
        "documents": len(documents), "passages": coverage["passages"],
        "questions_with_indexed_evidence": sum(indexed.values()), "variants": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matter", type=UUID, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    embedder, reranker = LocalEmbeddings(snapshot(EMBEDDING_MODEL)), LocalReranker(snapshot(RERANKER_MODEL))
    with connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        result = evaluate(conn, args.matter, suite, embedder, reranker)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
