"""Evaluate an offline evidence search cascade against source-span labels."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from filing_retrieval import (Filing, Intent, filing_bonus, gold_sources, identify_filing, page_role,
                             passage_role, route, scoped_ids, snapshot)
from lexintel_ingest.local_embeddings import LocalEmbeddings
from lexintel_ingest.local_reranker import LocalReranker
from lexintel_ingest.rank_service import EMBEDDING_MODEL, RERANKER_MODEL
from lexintel_ingest.retrieval import STOP_WORDS, source_pages
from retrieval import is_gold

GLOBAL_LIMIT = 22
SCOPED_LIMIT = 32
MODEL_POOL = 18
WORDS = re.compile(r"[^\W_]+")
NOISE = STOP_WORDS | set("the s t asked say said told did happen effect result about within use".split())
LEGAL_ACTION = re.compile(
    r"\b(?:we (?:affirm|reverse|vacate|remand|conclude|hold)|"
    r"(?:judgment|order|conviction)(?: of (?:the )?[\w’'-]+(?: [\w’'-]+){0,6})?"
    r" (?:is|are) (?:affirmed|reversed|vacated|quashed)|"
    r"it is (?:so ordered|ordered accordingly)|we are inclined to remit)\b", re.I
)
REMEDY = re.compile(
    r"\b(?:respectfully (?:asks?|requests?|submits?)|"
    r"(?:court|judgment|decision)(?: of (?:the )?[\w’'-]+(?: [\w’'-]+){0,6})?"
    r" should (?:affirm|reverse|vacate|remand|dismiss|deny|grant|be reversed|be affirmed)|"
    r"(?:affirm|reverse|vacate|remand|dismiss|deny|grant) the (?:judgment|decision|petition|appeal|order))\b",
    re.I,
)


def normalized(text):
    return " ".join(text.replace("\x02", "").replace("\u00ad", "").split())


def terms(text):
    return [word for word in WORDS.findall(normalized(text).casefold()) if word not in NOISE]


def load_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


class LexicalIndex:
    def __init__(self, texts):
        self.count = len(texts)
        counts = defaultdict(list)
        lengths = []
        for index, text in enumerate(texts):
            words = terms(text)
            lengths.append(len(words))
            for word, count in Counter(words).items():
                counts[word].append((index, count))
        avg_length = max(1, np.mean(lengths))
        normalization = 1.2 * (0.27 + 0.73 * np.asarray(lengths) / avg_length)
        self.postings = {}
        for word, posting in counts.items():
            positions, frequencies = np.array(posting).T
            idf = np.log1p((self.count - len(positions) + 0.5) / (len(positions) + 0.5))
            self.postings[word] = positions, idf * 2.2 * frequencies / (normalization[positions] + frequencies)

    def rank(self, query, indices, limit):
        values = np.zeros(self.count)
        for word in dict.fromkeys(terms(query)):
            if posting := self.postings.get(word):
                values[posting[0]] += posting[1]
        ranks = best(values, indices, limit)
        return [index for index in ranks if values[index] > 0]


def best(scores, subset, limit):
    if not len(subset):
        return []
    return subset[np.argsort(-scores[subset], kind="stable")[:limit]].tolist()


def classify(document):
    if docket := document.get("docket"):
        return Filing(docket["kind"], docket["court"], docket["date"][:4], docket["party"])
    return identify_filing(document)


def question_intent(question):
    text = question.casefold()
    party_match = re.search(r"\bdid (\w+) (?:ask|say|argue|request)|\b(\w+)[’']s (?:reply|opposition|certiorari|merits brief)", text)
    party = next((part for part in party_match.groups() if part), "") if party_match else ""
    request = re.search(r"\b(?:disposition|request|ask)\b", text)
    if "reply brief" in text:
        return Intent(("reply_brief",), party=party, role="requested_disposition" if request else "submission")
    if "opposition to certiorari" in text:
        return Intent(("cert_opposition",), party=party, role="requested_disposition" if request else "submission")
    if "certiorari petition" in text:
        return Intent(("petition",), party=party, role="submission")
    intent = route(question)
    if not intent.kinds and re.search(r"supreme court (?:majority|opinion)", text):
        return Intent(("opinion",), "supreme court", role="majority")
    return intent


def classify_role(passage, filing, roles):
    role = passage_role(passage, filing, roles)
    body = normalized(passage["content"])
    if role in {"syllabus", "dissent", "blank_form"}:
        return role
    if filing.kind in {"opinion", "judgment", "order"} and LEGAL_ACTION.search(body):
        return "holding" if filing.kind == "opinion" else "disposition"
    if filing.kind in {"brief", "merits_brief", "reply_brief", "cert_opposition", "petition"} and REMEDY.search(body):
        return "requested_disposition"
    if role == "front_matter" and re.search(
        r"^\s*(?:jurisdiction|opinions below|statement of the case|summary of argument|introduction)\s*\n"
        r"\s*\w[^\n.]{46}", passage["content"], re.I | re.M,
    ):
        return "submission" if filing.kind else ""
    if filing.kind in {"reply_brief", "cert_opposition"} and not role:
        return "submission"
    if len(terms(body)) < 16 and filing.kind not in {"verdict", "instructions"}:
        return "front_matter"
    return role


def passage_strength(role, intent):
    if role == intent.role and role:
        return 1.75
    if role == "disposition" and intent.role == "holding":
        return 1.75
    if role == "holding" and intent.role == "majority":
        return 0.75
    if role in {"front_matter", "syllabus", "dissent", "blank_form"}:
        return -3.5
    return 0


def within_filing_query(question, intent):
    text = question
    if intent.party:
        text = re.sub(rf"\b{re.escape(intent.party)}[’']?s?\b", " ", text, flags=re.I)
    if intent.court:
        text = re.sub(re.escape(intent.court), " ", text, flags=re.I)
    text = re.sub(r"\b(?:merits brief|district court|supreme court|jury instructions|"
                  r"federal circuit|(?:19|20)\d{2})\b", " ", text, flags=re.I)
    if intent.role == "requested_disposition":
        text += " affirm reverse vacate remand dismiss judgment conclusion"
    return normalized(text)


def rank_fusion(rankings, limit, penalty=None):
    weight = Counter()
    for ranking in rankings:
        for position, index in enumerate(ranking, 1):
            weight[index] += 1 / (20 + position)
    if penalty is not None:
        for index in weight:
            weight[index] *= penalty[index]
    return [index for index, _ in weight.most_common(limit)]


def matches(row, gold):
    return any(is_gold(label, row) for label in gold)


class Cascade:
    def __init__(self, documents, rows, embedder, reranker):
        self.documents = documents
        self.rows = rows
        self.embedder = embedder
        self.reranker = reranker
        self.indices = np.arange(len(rows))
        self.embeddings = np.asarray([
            json.loads(row["embedding"]) if isinstance(row["embedding"], str) else row["embedding"]
            for row in rows
        ], dtype=np.float32)
        self.lexical = LexicalIndex([row["passage"]["content"] for row in rows])
        self.lexical_headings = LexicalIndex([row["passage"]["search_text"] for row in rows])
        self.filings = {key: classify(doc) for key, doc in documents.items()}
        roles_by_page = {
            key: {n: page_role(page["text"], self.filings[key]) for n, page in enumerate(doc["pages"], 1)}
            for key, doc in documents.items()
        }
        self.roles = [
            classify_role(row["passage"], self.filings[row["document_id"]], roles_by_page[row["document_id"]])
            for row in rows
        ]
        self.body_weight = np.asarray([0.22 if role in {"front_matter", "syllabus"} else 1.0 for role in self.roles])
        self.by_document = defaultdict(list)
        for index, row in enumerate(rows):
            self.by_document[row["document_id"]].append(index)

    def collect(self, question, rewrites):
        intent = question_intent(question)
        preferred = scoped_ids(self.filings, intent)
        queries = list(dict.fromkeys([question, within_filing_query(question, intent), *rewrites]))
        query_vectors = self.embedder.encode(queries, "query").numpy()
        similarities = query_vectors @ self.embeddings.T

        def search(subset, depth):
            rankings = []
            for query, dense in zip(queries, similarities, strict=True):
                rankings.extend([
                    self.lexical.rank(query, subset, depth),
                    self.lexical_headings.rank(query, subset, depth),
                    best(dense, subset, depth),
                ])
            return rankings

        broad = rank_fusion(search(self.indices, 32), GLOBAL_LIMIT, self.body_weight)
        scoped_subset = np.asarray([i for doc in preferred for i in self.by_document[doc]], dtype=int)
        scoped_rankings = search(scoped_subset, 42) if len(scoped_subset) else []
        scoped = rank_fusion(scoped_rankings, SCOPED_LIMIT, self.body_weight)
        # Reserve a small lane for decisive evidence whose text may share few query words.
        operative = np.asarray([
            i for i in scoped_subset if passage_strength(self.roles[i], intent) > 0 and self.roles[i] in
            {"blank_form", "verdict", "holding", "disposition", "requested_disposition", "dissent"}
        ], dtype=int)
        direct = rank_fusion(search(operative, 8), 8, self.body_weight) if len(operative) else []
        pool = list(dict.fromkeys([*scoped, *direct, *broad]))
        return pool, intent, set(preferred), scoped, direct

    def rank(self, question, rewrites):
        pool, intent, preferred, scoped, direct = self.collect(question, rewrites)
        rankable = [
            {"passage": {"search_text": normalized(self.rows[i]["passage"]["content"])}}
            for i in pool
        ]
        logits = self.reranker.score(question, rankable)
        def key(index, logit):
            row = self.rows[index]
            filing = self.filings[row["document_id"]]
            filing_weight = 3.25 * (row["document_id"] in preferred) + 0.1 * filing_bonus(filing, intent)
            role_weight = passage_strength(self.roles[index], intent)
            if role_weight > 0 and intent.role in {"blank_form", "requested_disposition", "disposition"}:
                role_weight *= 2
            return logit + filing_weight + role_weight
        ranked = [index for index, logit in sorted(zip(pool, logits, strict=True), key=lambda pair: -key(*pair))]
        selector = list(dict.fromkeys([*ranked[:9], *scoped[:5], *direct[:4], *ranked[9:]]))[:MODEL_POOL]
        return ranked, selector

    def reference(self, question):
        query = self.embedder.encode([question], "query").numpy()[0]
        lexical = self.lexical_headings.rank(question, self.indices, 15)
        dense = best(self.embeddings @ query, self.indices, 15)
        pool = rank_fusion([lexical, dense], 20)
        ranked = self.reranker.rank(question, [self.rows[i] for i in pool])
        indices = {(self.rows[i]["document_id"], self.rows[i]["ordinal"]): i for i in pool}
        return [indices[row["document_id"], row["ordinal"]] for row in ranked]

    def evidence(self, indices):
        result = []
        for index in indices:
            row = self.rows[index]
            identifier = row["document_id"]
            filing = self.filings[identifier]
            result.append({
                "id": index, "source": self.documents[identifier]["name"],
                "role": self.roles[index], "pages": source_pages(row["passage"]),
                "filing": ", ".join(part for part in (filing.kind, filing.court, filing.year, filing.party) if part),
                "extract": normalized(row["passage"]["content"]).replace("\uf0fc", "[checked]"),
            })
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, nargs="+", required=True)
    parser.add_argument("--passages", type=Path, nargs="+", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--matter", help="Limit candidates to one source matter")
    parser.add_argument("--rewrites", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    def includes(item):
        return args.matter is None or item["matter_id"] == args.matter
    documents = {d["id"]: d for path in args.documents for d in load_jsonl(path) if includes(d)}
    rows = [r for path in args.passages for r in load_jsonl(path) if includes(r)]
    suite = json.loads(args.suite.read_text())
    keys = {d["sha256"]: k for k, d in suite["documents"].items()}
    corpus = {keys.get(doc["sha256"], identifier): doc for identifier, doc in documents.items()}
    for row in rows:
        doc = documents[row["document_id"]]
        row["document_key"] = keys.get(doc["sha256"], row["document_id"])
    labels = gold_sources(corpus, suite["questions"])
    rewrites = {item["id"]: item["values"] for item in load_jsonl(args.rewrites)} if args.rewrites else {}
    embedder = LocalEmbeddings(snapshot(EMBEDDING_MODEL))
    reranker = LocalReranker(snapshot(RERANKER_MODEL), device=args.device)
    cascade = Cascade(documents, rows, embedder, reranker)
    with args.output.open("w") as output:
        for item in suite["questions"]:
            started = perf_counter()
            ranked, shortlist = cascade.rank(item["question"], rewrites.get(item["id"], []))
            hits = [idx for idx in ranked if matches(rows[idx], labels[item["id"]])]
            result = {
                "id": item["id"], "question": item["question"], "ranked": ranked, "hits": hits,
                "candidates": cascade.evidence(shortlist), "seconds": perf_counter() - started,
                "documents": len(documents), "passages": len(rows),
            }
            if args.reference:
                baseline = cascade.reference(item["question"])
                result["reference_rank"] = next(
                    (n for n, idx in enumerate(baseline, 1) if matches(rows[idx], labels[item["id"]])), None
                )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
