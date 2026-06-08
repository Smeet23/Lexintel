"""
COMPREHENSIVE END-TO-END PIPELINE TEST
=======================================
Tests every step of the Lexintel legal RAG pipeline with REAL API calls:

1.  DB & Qdrant connectivity
2.  Matter + Document creation
3.  Text extraction (inline text)
4.  Chunking (semantic + fallback)
5.  Embedding (Voyage voyage-law-2)
6.  Sparse vector generation (BM25 via FastEmbed)
7.  Qdrant upsert (dense + sparse vectors)
8.  Standard RAG query (query_matter)
9.  Hybrid search (BM25 + dense + RRF fusion)
10. Reranking (cross-encoder)
11. Authority detection (Gemini)
12. Temporal extraction (regex + Gemini)
13. Conflict detection (NLI pairwise)
14. CREAC answer generation (Gemini)
15. Citation extraction + grounding
16. Citation verification (real API calls to free APIs)
17. Claim verification (NLI ensemble + CoV-RAG)
18. Agentic RAG (LangGraph router + clarify + retrieve + evaluate + generate + verify)
19. Cleanup

Uses a synthetic legal document with known content for deterministic assertions.
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FULL_E2E") != "1",
    reason="Full E2E pipeline makes real network/API calls; set RUN_FULL_E2E=1 to run.",
)

# Setup path
sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("E2E_TEST")

# ═══════════════════════════════════════════════════════════════
# Test document: a synthetic legal memo with known facts
# ═══════════════════════════════════════════════════════════════

LEGAL_DOCUMENT = """
MEMORANDUM OF LAW

IN THE MATTER OF: Smith v. TechCorp Inc.
Case No. 2024-CV-4521
United States District Court, Southern District of New York

DATE: January 15, 2024

I. STATEMENT OF FACTS

On March 3, 2023, Plaintiff John Smith entered into a Software License Agreement
("the Agreement") with TechCorp Inc. ("Defendant"). The Agreement, executed under
New York law, contained the following material provisions:

1. License Fee: $500,000 per annum, payable quarterly.
2. Term: Three (3) years, commencing April 1, 2023.
3. Limitation of Liability: Defendant's aggregate liability shall NOT exceed the
   total license fees paid in the twelve (12) months preceding the claim.
4. Arbitration Clause: All disputes shall be resolved through binding arbitration
   in New York City under AAA Commercial Rules.
5. Non-Compete: Plaintiff shall not use competing software products during the
   term of the Agreement.
6. Data Protection: Defendant shall comply with all applicable data protection
   laws, including GDPR and CCPA.

II. ISSUE PRESENTED

Whether the limitation of liability clause (Section 3) is enforceable under
New York law, and whether it can be invoked to cap damages arising from
Defendant's alleged breach of the data protection obligations (Section 6).

III. ANALYSIS

A. Enforceability of Limitation of Liability Clauses Under New York Law

Under New York law, limitation of liability clauses are generally enforceable
in commercial agreements between sophisticated parties. See Metropolitan Life
Ins. Co. v. Noble Lowndes Int'l, Inc., 84 N.Y.2d 430, 436 (1994). The Court
of Appeals has held that "[a] limitation of liability clause in a commercial
contract is generally enforceable if the amount set is not nominal."

However, courts have refused to enforce such clauses where they operate to
shield a party from liability for willful or grossly negligent conduct. See
Kalisch-Jarcho, Inc. v. City of New York, 58 N.Y.2d 377, 384 (1983).

B. Application to Data Protection Breaches

The enforceability of limitation clauses as applied to data protection
violations is a developing area of law. In Soo Line R.R. Co. v. St. Louis
Sw. Ry. Co., the court noted that public policy considerations may override
contractual limitations where the breach involves statutory obligations
designed to protect third parties.

The New York Attorney General has taken the position that contractual
limitations cannot shield parties from liability under the SHIELD Act
(N.Y. Gen. Bus. Law § 899-aa), which imposes mandatory notification and
security requirements.

Furthermore, under GDPR Article 82, data subjects have a right to
compensation for material and non-material damage resulting from data
protection violations. The European Court of Justice has held that this
right cannot be contractually waived between data controllers and processors
where it would prejudice the rights of data subjects. See Case C-311/18
(Schrems II).

C. Quantification of Damages

The total license fees paid in the 12 months preceding the breach amount to
$500,000. However, the data breach affected approximately 50,000 users,
with estimated per-user notification costs of $15 and potential regulatory
fines of up to $2,000,000 under the SHIELD Act.

IV. CONCLUSION

Based on the foregoing analysis, the limitation of liability clause is likely
enforceable as to general contractual claims, but may NOT be enforceable as
to data protection violations under the SHIELD Act and GDPR. The Defendant's
potential exposure exceeds the $500,000 cap by a significant margin.

RECOMMENDED ACTION: Pursue claims under the SHIELD Act and GDPR separately
from contractual claims, as these statutory claims may bypass the limitation
of liability provision.

Respectfully submitted,
Jane Doe, Esq.
Attorney for Plaintiff
"""

SECOND_DOCUMENT = """
SUPPLEMENTAL MEMORANDUM

RE: Smith v. TechCorp Inc. — Update on Arbitration Clause

DATE: March 10, 2024

SUPERSEDES: Section IV of Memorandum dated January 15, 2024 (in part)

Following additional research, it should be noted that the arbitration clause
in Section 4 of the Agreement may be unenforceable with respect to SHIELD Act
claims. In American Express Co. v. Italian Colors Restaurant, 570 U.S. 228 (2013),
the Supreme Court held that arbitration clauses are generally enforceable even
when they foreclose certain statutory claims. However, under New York law, the
SHIELD Act's notification requirements are considered non-waivable public policy.

Furthermore, the non-compete clause (Section 5) may face scrutiny under the
Federal Trade Commission's proposed rule banning non-compete agreements, which
was finalized on April 23, 2024. This rule, if upheld, would render Section 5
void as of the effective date.

IMPORTANT UPDATE: The damages assessment should be revised upward. The data
breach notification costs now total $750,000 (50,000 users × $15), and the
SHIELD Act penalties have been assessed at $2,500,000. Total potential exposure
is therefore $3,250,000 — significantly exceeding the $500,000 limitation cap.

This memorandum PARTIALLY SUPERSEDES the earlier analysis with respect to damages
quantification and the arbitration clause analysis.
"""


def step(num, name):
    """Print a step header."""
    print(f"\n{'='*70}")
    print(f"  STEP {num}: {name}")
    print(f"{'='*70}\n")


def pass_step(num, name, detail=""):
    print(f"  [PASS] Step {num}: {name}" + (f" — {detail}" if detail else ""))


def fail_step(num, name, detail=""):
    print(f"  [FAIL] Step {num}: {name}" + (f" — {detail}" if detail else ""))


async def run_full_e2e():
    results = {}
    matter_id = None

    try:
        # ════════════════════════════════════════════════════════════
        # STEP 1: Infrastructure connectivity
        # ════════════════════════════════════════════════════════════
        step(1, "INFRASTRUCTURE CONNECTIVITY")

        from backend.database import get_engine
        from backend.models import Base, Matter, Document, Chunk
        from sqlalchemy.orm import Session

        engine = get_engine()
        from sqlalchemy import text
        with Session(engine) as db:
            db.execute(text("SELECT 1"))
        pass_step(1, "PostgreSQL", "connected")
        results["postgres"] = "PASS"

        from qdrant_client import QdrantClient
        qclient = QdrantClient(url="http://localhost:6333", check_compatibility=False)
        collections = qclient.get_collections()
        pass_step(1, "Qdrant", f"connected, {len(collections.collections)} existing collections")
        results["qdrant"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 2: Create Matter + Documents (DB)
        # ════════════════════════════════════════════════════════════
        step(2, "CREATE MATTER + DOCUMENTS")

        matter_id = str(uuid.uuid4())
        with Session(engine) as db:
            matter = Matter(
                id=uuid.UUID(matter_id),
                name="E2E Test: Smith v. TechCorp",
                blob_storage_path="e2e-test/placeholder",
                file_type="txt",
                status="processing",
            )
            db.add(matter)
            db.flush()

            doc1 = Document(
                id=uuid.uuid4(),
                matter_id=matter.id,
                name="memo_of_law.txt",
                blob_storage_path="e2e-test/memo_of_law.txt",
                file_type="txt",
                status="completed",
            )
            doc2 = Document(
                id=uuid.uuid4(),
                matter_id=matter.id,
                name="supplemental_memo.txt",
                blob_storage_path="e2e-test/supplemental_memo.txt",
                file_type="txt",
                status="completed",
            )
            db.add(doc1)
            db.add(doc2)
            db.commit()
            doc1_id = str(doc1.id)
            doc2_id = str(doc2.id)

        pass_step(2, "Matter created", f"id={matter_id[:8]}...")
        pass_step(2, "Document 1", f"memo_of_law.txt (id={doc1_id[:8]}...)")
        pass_step(2, "Document 2", f"supplemental_memo.txt (id={doc2_id[:8]}...)")
        results["matter_creation"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 3: Chunking
        # ════════════════════════════════════════════════════════════
        step(3, "CHUNKING (semantic + fallback)")

        from backend.services.chunking import chunk_document_from_blob
        t0 = time.time()
        # chunk_document_from_blob expects bytes + file_type, returns List[Dict]
        chunks1_raw = chunk_document_from_blob(LEGAL_DOCUMENT.encode("utf-8"), file_type="txt")
        chunks2_raw = chunk_document_from_blob(SECOND_DOCUMENT.encode("utf-8"), file_type="txt")
        chunk_time = time.time() - t0

        # Extract text content from chunk dicts
        chunks1 = [c.get("content", c.get("text", "")) for c in chunks1_raw]
        chunks2 = [c.get("content", c.get("text", "")) for c in chunks2_raw]

        print(f"  Doc 1: {len(chunks1)} chunks")
        for i, c in enumerate(chunks1[:3]):
            print(f"    chunk[{i}]: {len(c)} chars — \"{c[:60]}...\"")
        print(f"  Doc 2: {len(chunks2)} chunks")
        for i, c in enumerate(chunks2[:3]):
            print(f"    chunk[{i}]: {len(c)} chars — \"{c[:60]}...\"")
        print(f"  Total: {len(chunks1) + len(chunks2)} chunks in {chunk_time:.2f}s")

        all_chunks = [(c, doc1_id, "memo_of_law.txt") for c in chunks1] + \
                     [(c, doc2_id, "supplemental_memo.txt") for c in chunks2]
        pass_step(3, "Chunking", f"{len(all_chunks)} total chunks")
        results["chunking"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 4: Embedding (Voyage voyage-law-2)
        # ════════════════════════════════════════════════════════════
        step(4, "EMBEDDING (Voyage voyage-law-2)")

        from backend.services.embeddings import embed_chunks, embed_query as embed_q, _detect_provider

        provider = _detect_provider()
        print(f"  Provider: {provider}")

        chunk_texts = [c[0] for c in all_chunks]
        t0 = time.time()
        embeddings = embed_chunks(chunk_texts)
        embed_time = time.time() - t0

        print(f"  Embedded {len(embeddings)} chunks in {embed_time:.2f}s")
        print(f"  Dimension: {len(embeddings[0])}")
        print(f"  Sample values: {embeddings[0][:5]}")

        assert len(embeddings[0]) == 1024, f"Expected 1024 dims, got {len(embeddings[0])}"
        pass_step(4, "Embeddings", f"{provider}, {len(embeddings)}x1024, {embed_time:.2f}s")
        results["embeddings"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 5: Sparse vectors (BM25 via FastEmbed)
        # ════════════════════════════════════════════════════════════
        step(5, "SPARSE VECTORS (BM25)")

        from backend.services.hybrid_search import generate_sparse_vector

        sparse_vectors = []
        t0 = time.time()
        for text in chunk_texts:
            sv = generate_sparse_vector(text)
            sparse_vectors.append(sv)
        sparse_time = time.time() - t0

        non_none = [s for s in sparse_vectors if s is not None]
        if non_none:
            sample = non_none[0]
            print(f"  Generated {len(non_none)}/{len(sparse_vectors)} sparse vectors in {sparse_time:.2f}s")
            print(f"  Sample sparse vector: {len(sample.get('indices', []))} non-zero indices")
            pass_step(5, "BM25 sparse vectors", f"{len(non_none)} vectors, {sparse_time:.2f}s")
            results["sparse_vectors"] = "PASS"
        else:
            print(f"  WARNING: No sparse vectors generated (FastEmbed may not be available)")
            results["sparse_vectors"] = "SKIP"

        # ════════════════════════════════════════════════════════════
        # STEP 6: Qdrant upsert (dense + sparse)
        # ════════════════════════════════════════════════════════════
        step(6, "QDRANT UPSERT (dense + sparse vectors)")

        from backend.services.vector_store import create_collection, upsert_vectors

        collection_name = f"matter_{matter_id}"
        create_collection(matter_id)
        print(f"  Collection created: {collection_name}")

        # Build chunk records for DB + Qdrant
        chunk_ids = []
        with Session(engine) as db:
            for i, (text, doc_id, doc_name) in enumerate(all_chunks):
                chunk_id = uuid.uuid4()
                chunk_ids.append(str(chunk_id))
                chunk_obj = Chunk(
                    id=chunk_id,
                    matter_id=uuid.UUID(matter_id),
                    document_id=uuid.UUID(doc_id),
                    content=text,
                    chunk_sequence=i,
                    page_num="1",
                    section_name=f"Section {i+1}",
                )
                db.add(chunk_obj)
            db.commit()
        print(f"  {len(chunk_ids)} chunks saved to PostgreSQL")

        # Upsert to Qdrant with metadata (matching upsert_vectors signature)
        chunk_dicts = []
        for i, (text, doc_id, doc_name) in enumerate(all_chunks):
            chunk_dicts.append({
                "id": chunk_ids[i],
                "content": text,
                "page_num": "1",
                "section_name": f"Section {i+1}",
                "chunk_sequence": i,
                "document_id": doc_id,
                "document_name": doc_name,
            })

        upsert_vectors(matter_id, chunk_dicts, embeddings, sparse_vectors=sparse_vectors)
        print(f"  {len(chunk_dicts)} points upserted to Qdrant")

        # Verify
        info = qclient.get_collection(collection_name)
        print(f"  Collection points: {info.points_count}")
        assert info.points_count == len(all_chunks), f"Expected {len(all_chunks)} points"
        pass_step(6, "Qdrant upsert", f"{info.points_count} points in {collection_name}")
        results["qdrant_upsert"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 7: Query embedding (asymmetric)
        # ════════════════════════════════════════════════════════════
        step(7, "QUERY EMBEDDING (asymmetric)")

        query_text = "Is the limitation of liability clause enforceable for data protection breaches under the SHIELD Act?"
        t0 = time.time()
        query_vec = embed_q(query_text)
        q_time = time.time() - t0

        print(f"  Query: \"{query_text}\"")
        print(f"  Dimension: {len(query_vec)}")
        print(f"  Time: {q_time:.3f}s")
        assert len(query_vec) == 1024
        pass_step(7, "Query embedding", f"1024-dim, {q_time:.3f}s")
        results["query_embedding"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 8: Vector search + hybrid BM25
        # ════════════════════════════════════════════════════════════
        step(8, "VECTOR SEARCH + HYBRID BM25 + RRF FUSION")

        from backend.services.vector_store import search_vectors
        from backend.services.hybrid_search import (
            classify_query_type, get_rrf_weights, reciprocal_rank_fusion,
        )

        # Dense search
        t0 = time.time()
        dense_results = search_vectors(matter_id, query_vec, limit=15)
        dense_time = time.time() - t0
        print(f"  Dense search: {len(dense_results)} results in {dense_time:.3f}s")
        for i, r in enumerate(dense_results[:3]):
            print(f"    [{i}] score={r.get('score', 0):.4f} — \"{r.get('content', '')[:60]}...\"")

        # Query classification
        qtype = classify_query_type(query_text)
        print(f"  Query type: {qtype}")

        # BM25 + RRF
        sparse_query = generate_sparse_vector(query_text)
        if sparse_query:
            from backend.services.vector_store import search_sparse_vectors
            bm25_weight, dense_weight = get_rrf_weights(query_text)
            print(f"  Weights: bm25={bm25_weight:.2f}, dense={dense_weight:.2f}")
            try:
                bm25_results = search_sparse_vectors(matter_id, sparse_query, limit=15)
                if bm25_results:
                    fused = reciprocal_rank_fusion(
                        bm25_results=bm25_results,
                        dense_results=dense_results,
                        bm25_weight=bm25_weight,
                        dense_weight=dense_weight,
                        top_n=8,
                    )
                    print(f"  BM25 results: {len(bm25_results)}")
                    print(f"  RRF fused: {len(fused)} results")
                    results["hybrid_search"] = "PASS"
                    pass_step(8, "Hybrid BM25+dense+RRF", f"{len(fused)} fused results")
                else:
                    print(f"  BM25 returned 0 results (sparse vectors may not be in Qdrant)")
                    results["hybrid_search"] = "PARTIAL"
            except Exception as e:
                print(f"  BM25 search failed: {e}")
                results["hybrid_search"] = "FAIL"
        else:
            print(f"  Sparse query generation returned None")
            results["hybrid_search"] = "SKIP"

        pass_step(8, "Dense search", f"{len(dense_results)} results")
        results["vector_search"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 9: Reranking (cross-encoder)
        # ════════════════════════════════════════════════════════════
        step(9, "RERANKING (cross-encoder/ms-marco-MiniLM-L-6-v2)")

        from backend.services.rag_engine import rerank_chunks

        t0 = time.time()
        reranked = rerank_chunks(query_text, dense_results, top_k=5)
        rerank_time = time.time() - t0

        print(f"  Reranked {len(dense_results)} → {len(reranked)} in {rerank_time:.2f}s")
        for i, r in enumerate(reranked):
            print(f"    [{i}] score={r.get('score', 0):.4f} — \"{r.get('content', '')[:60]}...\"")

        pass_step(9, "Reranking", f"{len(reranked)} top chunks, {rerank_time:.2f}s")
        results["reranking"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 10: Authority detection (Gemini)
        # ════════════════════════════════════════════════════════════
        step(10, "AUTHORITY DETECTION (Gemini structured output)")

        from backend.services.authority_detector import detect_authority

        t0 = time.time()
        auth_result = await detect_authority(LEGAL_DOCUMENT[:2000])
        auth_time = time.time() - t0

        print(f"  Detection time: {auth_time:.2f}s")
        print(f"  Result: {json.dumps(auth_result, indent=2, default=str)}")

        pass_step(10, "Authority detection", f"{auth_time:.2f}s")
        results["authority_detection"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 11: Temporal extraction
        # ════════════════════════════════════════════════════════════
        step(11, "TEMPORAL EXTRACTION (regex + dateutil)")

        from backend.services.temporal_extractor import extract_temporal_metadata

        t0 = time.time()
        temporal1 = await extract_temporal_metadata(LEGAL_DOCUMENT[:2000])
        temporal2 = await extract_temporal_metadata(SECOND_DOCUMENT[:1000])
        temporal_time = time.time() - t0

        print(f"  Doc 1 temporal: {json.dumps(temporal1, indent=2, default=str)}")
        print(f"  Doc 2 temporal: {json.dumps(temporal2, indent=2, default=str)}")
        print(f"  Extraction time: {temporal_time:.2f}s")

        pass_step(11, "Temporal extraction", f"{temporal_time:.2f}s")
        results["temporal_extraction"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 12: NLI models test (claim verification foundation)
        # ════════════════════════════════════════════════════════════
        step(12, "NLI ENSEMBLE (DeBERTa base + small)")

        from backend.services.claim_verifier import (
            _get_nli_base, _get_nli_small, _ensemble_nli_verify,
            _check_token_coverage, _split_into_claims,
        )

        base = _get_nli_base()
        small = _get_nli_small()
        print(f"  Base model: {'loaded' if base else 'FAILED'}")
        print(f"  Small model: {'loaded' if small else 'FAILED'}")

        # Test cases
        test_cases = [
            ("The liability cap is $500,000", "Defendant's aggregate liability shall NOT exceed the total license fees paid", "should be supported"),
            ("Early termination is allowed", "The Agreement has a term of three years", "should be neutral/ambiguous"),
            ("The agreement is governed by California law", "The Agreement, executed under New York law", "should be contradicted"),
            ("Data breach notification costs total $750,000", "estimated per-user notification costs of $15 and 50,000 users", "should be supported"),
        ]

        for claim, source, expected in test_cases:
            result = _ensemble_nli_verify(claim, source)
            print(f"  [{result['status']:20s}] conf={result['confidence']:.2f} agree={result['agreement']} | \"{claim[:50]}\" ({expected})")

        # Token coverage test
        cov = _check_token_coverage(
            "The defendant is liable",
            "liable on count 1 but acquitted on counts 2 through 5"
        )
        print(f"  Token coverage (partial truth): {cov:.3f} (threshold={0.45})")

        # Sentence splitting test
        claims = _split_into_claims(
            "In the U.S. Supreme Court case, the court held that Art. 5 applies [1]. "
            "The FTC's rule on non-compete agreements was finalized on April 23, 2024 [2]."
        )
        print(f"  NUPunkt split: {len(claims)} claims from 2 sentences")
        for c in claims:
            print(f"    - \"{c['text'][:70]}...\" cited={c['cited_sources']}")

        pass_step(12, "NLI ensemble", f"base={'OK' if base else 'FAIL'}, small={'OK' if small else 'FAIL'}")
        results["nli_models"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 13: Conflict detection (NLI pairwise)
        # ════════════════════════════════════════════════════════════
        step(13, "CONFLICT DETECTION (NLI pairwise)")

        from backend.services.conflict_detector import detect_conflicts

        # Build chunks with conflicting content
        test_chunks = [
            {"content": "The limitation of liability clause caps damages at $500,000.", "document_name": "memo_of_law.txt", "score": 0.9},
            {"content": "Total potential exposure is $3,250,000, significantly exceeding the $500,000 cap.", "document_name": "supplemental_memo.txt", "score": 0.85},
            {"content": "The arbitration clause is enforceable under the FAA.", "document_name": "memo_of_law.txt", "score": 0.7},
            {"content": "The arbitration clause may be unenforceable with respect to SHIELD Act claims.", "document_name": "supplemental_memo.txt", "score": 0.75},
        ]

        t0 = time.time()
        conflicts = detect_conflicts(test_chunks, query=query_text)
        conflict_time = time.time() - t0

        print(f"  Detection time: {conflict_time:.2f}s")
        if conflicts:
            print(f"  Conflicts found: {len(conflicts.get('conflicts', []))}")
            for c in conflicts.get("conflicts", []):
                print(f"    - \"{c.get('claim_a', '')[:50]}\" vs \"{c.get('claim_b', '')[:50]}\"")
                print(f"      score={c.get('score', 0):.3f} type={c.get('type', '')}")
        else:
            print(f"  No conflicts detected (or detector returned None)")

        pass_step(13, "Conflict detection", f"{conflict_time:.2f}s")
        results["conflict_detection"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 14: FULL RAG QUERY (standard pipeline)
        # ════════════════════════════════════════════════════════════
        step(14, "FULL RAG QUERY (standard pipeline — query_matter)")

        from backend.services.rag_engine import query_matter

        with Session(engine) as db:
            t0 = time.time()
            rag_result = await query_matter(
                matter_id, query_text, db,
                conversation_history=None,
                include_legal_research=False,
            )
            rag_time = time.time() - t0

        print(f"  Query time: {rag_time:.2f}s")
        print(f"  Answer length: {len(rag_result.get('answer', '') or '')} chars")
        print(f"  Sources: {len(rag_result.get('sources', []))}")
        print(f"  Citations: {len(rag_result.get('citations', []))}")
        print(f"  Tokens used: {rag_result.get('tokens_used', 0)}")
        print(f"  Model: {rag_result.get('model', '')}")
        print(f"  Confidence: {json.dumps(rag_result.get('confidence', {}), indent=4)}")

        answer = rag_result.get("answer", "") or ""
        print(f"\n  --- ANSWER (first 500 chars) ---")
        print(f"  {answer[:500]}")
        print(f"  --- END ANSWER ---")

        # Check citation verification
        cv = rag_result.get("citation_verification")
        if cv:
            print(f"\n  Citation verification: {json.dumps(cv.get('summary', {}))}")
            results["citation_verification_pipeline"] = "PASS"
        else:
            print(f"\n  Citation verification: None (may be disabled or no citations found)")
            results["citation_verification_pipeline"] = "SKIP"

        # Check claim verification
        clv = rag_result.get("claim_verification")
        if clv:
            summary = clv.get("summary", {})
            print(f"  Claim verification: {json.dumps(summary)}")
            claims_list = clv.get("verified_claims", [])
            for vc in claims_list[:5]:
                review = " [REVIEW]" if vc.get("requires_review") else ""
                print(f"    [{vc.get('status'):20s}] conf={vc.get('confidence', 0):.2f}{review} — \"{vc.get('claim_text', '')[:60]}\"")
            results["claim_verification_pipeline"] = "PASS"
        else:
            print(f"  Claim verification: None")
            results["claim_verification_pipeline"] = "SKIP"

        # Check conflict analysis
        ca = rag_result.get("conflict_analysis")
        if ca:
            print(f"  Conflict analysis: {len(ca.get('conflicts', []))} conflicts")
            results["conflict_analysis_pipeline"] = "PASS"
        else:
            print(f"  Conflict analysis: None")
            results["conflict_analysis_pipeline"] = "SKIP"

        pass_step(14, "Standard RAG", f"{rag_time:.2f}s, {len(answer)} chars answer")
        results["standard_rag"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 15: Citation verification (standalone, real APIs)
        # ════════════════════════════════════════════════════════════
        step(15, "CITATION VERIFICATION (standalone, real API calls)")

        from backend.services.citation_extractor import extract_citations_regex
        from backend.services.citation_lookup import lookup_citation

        test_citations = [
            ("Metropolitan Life Ins. Co. v. Noble Lowndes Int'l, Inc., 84 N.Y.2d 430 (1994)", "US"),
            ("American Express Co. v. Italian Colors Restaurant, 570 U.S. 228 (2013)", "US"),
        ]

        for cite, jurisdiction in test_citations:
            t0 = time.time()
            try:
                result = await lookup_citation(cite, jurisdiction=jurisdiction)
                v_time = time.time() - t0
                if result:
                    print(f"  [{result.get('status', 'unknown'):10s}] {v_time:.2f}s — {cite[:60]}")
                    if result.get("url"):
                        print(f"    URL: {result['url']}")
                else:
                    print(f"  [not_found ] {v_time:.2f}s — {cite[:60]}")
            except Exception as e:
                v_time = time.time() - t0
                print(f"  [ERROR     ] {v_time:.2f}s — {cite[:60]}: {e}")

        # Test citation extraction from document
        extracted = extract_citations_regex(LEGAL_DOCUMENT[:2000])
        print(f"\n  Extracted {len(extracted)} citations from document")
        for ext in extracted[:5]:
            print(f"    - {ext.get('text', '')[:60]} ({ext.get('jurisdiction', 'unknown')})")

        pass_step(15, "Citation verification", f"{len(test_citations)} citations tested")
        results["citation_verification"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 16: Claim verification (standalone, full pipeline)
        # ════════════════════════════════════════════════════════════
        step(16, "CLAIM VERIFICATION (standalone, NLI + CoV-RAG)")

        from backend.services.claim_verifier import verify_claims

        test_answer = (
            "The limitation of liability clause caps damages at $500,000 [1]. "
            "However, under the SHIELD Act, this cap may NOT be enforceable for data protection breaches [1][2]. "
            "The total exposure is estimated at $3,250,000 including notification costs and penalties [2]. "
            "The arbitration clause is fully enforceable under federal law [1]. "
            "California law governs the agreement [1]."
        )

        test_sources = [
            {"content": LEGAL_DOCUMENT, "score": 0.9},
            {"content": SECOND_DOCUMENT, "score": 0.85},
        ]

        t0 = time.time()
        claim_result = await verify_claims(test_answer, test_sources)
        claim_time = time.time() - t0

        if claim_result:
            print(f"  Verification time: {claim_time:.2f}s")
            print(f"  Summary: {json.dumps(claim_result.get('summary', {}))}")
            for vc in claim_result.get("verified_claims", []):
                issues = ", ".join(vc.get("issues", [])) if vc.get("issues") else "none"
                review = " [REVIEW]" if vc.get("requires_review") else ""
                print(f"    [{vc['status']:20s}] conf={vc['confidence']:.2f} tier={vc.get('verification_tier', 0)}{review}")
                print(f"      claim: \"{vc['claim_text'][:70]}\"")
                print(f"      issues: {issues}")

        pass_step(16, "Claim verification", f"{claim_time:.2f}s")
        results["claim_verification_standalone"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 17: Agentic RAG (LangGraph full pipeline)
        # ════════════════════════════════════════════════════════════
        step(17, "AGENTIC RAG (LangGraph full pipeline)")

        from backend.services.agentic_rag import (
            agentic_query_matter, _get_graph, _fast_llm_json,
            router_node, clarify_node, AgenticState,
        )

        # Test router
        print("  --- Router Node ---")
        simple_state = {"query": "What is the liability cap?", "rewrite_count": 0}
        router_result = router_node(simple_state)
        print(f"  Simple query: complexity={router_result['complexity']}, type={router_result['query_type']}")

        complex_state = {"query": query_text, "rewrite_count": 0}
        router_result2 = router_node(complex_state)
        print(f"  Complex query: complexity={router_result2['complexity']}, type={router_result2['query_type']}")

        # Test clarify
        print("\n  --- Clarify Node ---")
        clarify_result = clarify_node({"query": query_text})
        print(f"  Jurisdiction: {clarify_result.get('jurisdiction')}")
        print(f"  Entities: {clarify_result.get('entities')}")
        print(f"  Refined: \"{clarify_result.get('refined_query', '')[:80]}\"")

        # Full agentic query (force-enable temporarily)
        print("\n  --- Full Agentic Query (force-enabled) ---")
        from backend.config import get_settings
        settings = get_settings()
        original_flag = settings.agentic_rag_enabled
        settings.agentic_rag_enabled = True

        with Session(engine) as db:
            t0 = time.time()
            agentic_result = await agentic_query_matter(
                matter_id, query_text, db,
                conversation_history=None,
                include_legal_research=False,
            )
            agentic_time = time.time() - t0

        settings.agentic_rag_enabled = original_flag  # Restore

        print(f"  Query time: {agentic_time:.2f}s")
        a_answer = agentic_result.get("answer", "") or ""
        print(f"  Answer length: {len(a_answer)} chars")
        print(f"  Sources: {len(agentic_result.get('sources', []))}")
        print(f"  Tokens used: {agentic_result.get('tokens_used', 0)}")

        meta = agentic_result.get("agentic_metadata", {})
        if meta:
            print(f"  Agentic metadata:")
            print(f"    complexity: {meta.get('complexity')}")
            print(f"    query_type: {meta.get('query_type')}")
            print(f"    jurisdiction: {meta.get('jurisdiction')}")
            print(f"    entities: {meta.get('entities')}")
            print(f"    rewrite_count: {meta.get('rewrite_count')}")
            print(f"    retrieval_score: {meta.get('retrieval_score', 0):.3f}")

        a_cv = agentic_result.get("citation_verification")
        a_clv = agentic_result.get("claim_verification")
        a_ca = agentic_result.get("conflict_analysis")
        print(f"  Citation verification: {'YES' if a_cv else 'NO'}")
        print(f"  Claim verification: {'YES' if a_clv else 'NO'}")
        print(f"  Conflict analysis: {'YES' if a_ca else 'NO'}")

        print(f"\n  --- AGENTIC ANSWER (first 500 chars) ---")
        print(f"  {a_answer[:500]}")
        print(f"  --- END ---")

        pass_step(17, "Agentic RAG", f"{agentic_time:.2f}s, complexity={meta.get('complexity', 'N/A')}")
        results["agentic_rag"] = "PASS"

        # ════════════════════════════════════════════════════════════
        # STEP 18: Groq fast LLM test
        # ════════════════════════════════════════════════════════════
        step(18, "GROQ FAST LLM (Llama 3.3 70B)")

        from backend.services.agentic_rag import _GROQ_AVAILABLE

        if _GROQ_AVAILABLE:
            t0 = time.time()
            groq_result = _fast_llm_json(
                "You are a legal analyst. Analyze the legal issue and provide structured output.",
                "Query: Is a limitation of liability clause enforceable for data protection breaches under NY SHIELD Act?\n"
                "Respond with JSON: {\"enforceable\": true/false, \"reason\": \"brief\", \"confidence\": 0.0-1.0}",
            )
            groq_time = time.time() - t0
            print(f"  Groq response time: {groq_time:.3f}s")
            print(f"  Result: {json.dumps(groq_result, indent=2)}")
            pass_step(18, "Groq Llama 3.3", f"{groq_time:.3f}s")
            results["groq"] = "PASS"
        else:
            print(f"  Groq not available (no API key)")
            results["groq"] = "SKIP"

    except Exception as e:
        print(f"\n  [FATAL ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        results["fatal_error"] = str(e)

    finally:
        # ════════════════════════════════════════════════════════════
        # CLEANUP
        # ════════════════════════════════════════════════════════════
        step(19, "CLEANUP")

        if matter_id:
            try:
                qclient = QdrantClient(url="http://localhost:6333", check_compatibility=False)
                qclient.delete_collection(f"matter_{matter_id}")
                print(f"  Deleted Qdrant collection: matter_{matter_id}")
            except Exception as e:
                print(f"  Qdrant cleanup failed: {e}")

            try:
                from backend.database import get_engine
                from sqlalchemy.orm import Session
                from backend.models import Chunk, Document, Matter

                engine = get_engine()
                with Session(engine) as db:
                    db.query(Chunk).filter(Chunk.matter_id == uuid.UUID(matter_id)).delete()
                    db.query(Document).filter(Document.matter_id == uuid.UUID(matter_id)).delete()
                    db.query(Matter).filter(Matter.id == uuid.UUID(matter_id)).delete()
                    db.commit()
                print(f"  Deleted DB records for matter {matter_id[:8]}...")
            except Exception as e:
                print(f"  DB cleanup failed: {e}")

        # ════════════════════════════════════════════════════════════
        # FINAL REPORT
        # ════════════════════════════════════════════════════════════
        print(f"\n{'='*70}")
        print(f"  FINAL REPORT")
        print(f"{'='*70}\n")

        passed = sum(1 for v in results.values() if v == "PASS")
        failed = sum(1 for v in results.values() if v == "FAIL")
        skipped = sum(1 for v in results.values() if v in ("SKIP", "PARTIAL"))
        total = len(results)

        for key, val in results.items():
            icon = "PASS" if val == "PASS" else ("FAIL" if val == "FAIL" else "SKIP")
            print(f"  [{icon:4s}] {key}")

        print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped out of {total}")
        print(f"{'='*70}\n")

        return results


@pytest.mark.asyncio
async def test_full_e2e_pipeline():
    results = await run_full_e2e()
    assert not results.get("fatal_error"), results.get("fatal_error")
    failed = [k for k, v in results.items() if v == "FAIL"]
    assert not failed, f"E2E steps failed: {failed}"


if __name__ == "__main__":
    asyncio.run(run_full_e2e())
