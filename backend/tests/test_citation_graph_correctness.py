"""Exhaustive correctness harness for the citation graph (Gaps 1-6).

Runs against LIVE Postgres (lexintel_postgres) with a throwaway Matter/Document
that is fully cleaned up. ALL LLM + network entry points are monkeypatched so the
suite uses ZERO Gemini/Groq/CourtListener quota and is deterministic.

Covered invariants:
  Gap 1: no temporally-impossible negative case_cites edges (chronology guard)
  Gap 2: negation false-positive downgraded by independent verifier; window dedup
  Gap 3: parallel cites merge to one node via cluster_id
  Gap 4: no self-edges (canonical-key + id guard)
  Gap 5: PageRank/build_networkx_graph attributes cross-matter edge endpoints
  Gap 6: canonical citation key preserves the page-number space, merges variants

Run: cd backend && ../backend/venv/bin/pytest tests/test_citation_graph_correctness.py -v
"""
import asyncio
import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import backend.services.citation_graph as cg
from backend.models import Matter, Document, CitationNode, CitationEdge


def _real_database_url() -> str:
    """Read the real DATABASE_URL from env / backend/.env.

    conftest.py mocks get_settings to point at a non-existent 'test' DB, so this
    harness (which needs LIVE Postgres) resolves the real URL independently.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("DATABASE_URL not found for live-Postgres harness")


# Skip the entire module at collection time if no live Postgres is configured,
# rather than raising RuntimeError and breaking `pytest --collect-only`.
def _db_available() -> bool:
    try:
        _real_database_url()
        return True
    except (RuntimeError, OSError):
        return False


if not _db_available():
    pytestmark = pytest.mark.skip(  # type: ignore[assignment]
        reason="Live Postgres not configured (DATABASE_URL missing / backend/.env absent)."
    )


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def _engine():
    return create_engine(_real_database_url(), pool_pre_ping=True)


@pytest.fixture()
def db(_engine):
    Session = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def matter(db):
    """A throwaway matter + document, cleaned up (with its graph rows) after."""
    mid = uuid.uuid4()
    did = uuid.uuid4()
    m = Matter(id=mid, name="CORRECTNESS TEST", status="ready", file_type="txt",
               blob_storage_path="x", is_deleted=False)
    db.add(m)
    db.flush()
    d = Document(id=did, matter_id=mid, name="memo", status="ready",
                 file_type="txt", blob_storage_path="x")
    db.add(d)
    db.commit()
    yield mid, did
    db.query(CitationEdge).filter(CitationEdge.matter_id == mid).delete()
    db.query(CitationNode).filter(CitationNode.matter_id == mid).delete()
    db.query(Document).filter(Document.id == did).delete()
    db.query(Matter).filter(Matter.id == mid).delete()
    db.commit()


def _patch_no_network(monkeypatch, triples, *, cluster_for=None):
    """Patch every LLM/network entry point. ``triples`` is what the relationship
    extractor returns; ``cluster_for`` maps a citation string -> cluster_id so we
    can simulate CourtListener parallel-cite resolution with no network."""
    async def fake_rel(_text):
        return list(triples)

    async def fake_classify(*a, **k):
        return {"treatment": "CITES", "confidence": 0.5}

    async def fake_extract_all(*a, **k):
        return []  # no per-chunk document_cites noise in these tests

    async def fake_enrich(db, node):
        if cluster_for:
            cid = cluster_for.get(node.citation_text)
            if cid and not node.courtlistener_id:
                node.courtlistener_id = cid
                db.flush()
        return False

    async def fake_verify(treatment, evidence, citing, cited):
        return ("unknown", 0.0)  # neutral: keep the negative edge as-is

    monkeypatch.setattr(cg, "extract_relationships_llm", fake_rel)
    monkeypatch.setattr(cg, "classify_treatment", fake_classify)
    monkeypatch.setattr(cg, "extract_all_citations", fake_extract_all)
    monkeypatch.setattr(cg, "enrich_node_from_courtlistener", fake_enrich)
    import backend.services.treatment_verifier as tv
    monkeypatch.setattr(tv, "verify_negative_treatment", fake_verify)


def _index(db, mid, did, triples, **patch_kw):
    chunks = [{"content": "stub", "section_name": "", "page_num": 1}]
    return asyncio.run(
        cg.extract_and_index_citations(db, mid, did, "memo", "US", chunks)
    )


# Fictional reporter citations: real reporter ('U.S.' parses via eyecite) but
# volumes/pages that do NOT exist in CourtListener, so tests are fully isolated
# from the global node table and control their own years via the triples.
NEWER = "9001 U.S. 10"   # the citing/overruling case
OLDER = "9000 U.S. 20"   # the cited/overruled case
THIRD = "9002 U.S. 30"


# ──────────────────────────────────────────────────────────────────────────
# Gap 6 — canonical citation key (pure)
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("347 U. S. 483", "347 U.S. 483"),   # spaced reporter -> canonical, page space kept
    ("347 U.S. 483", "347 U.S. 483"),
    ("347 US 483", "347 U.S. 483"),       # dotless -> canonical (regex could never)
    ("410 U. S. 113", "410 U.S. 113"),
])
def test_canonical_key_preserves_page_space(raw, expected):
    assert cg._canonical_citation_key(raw) == expected


def test_canonical_key_parallel_cites_distinct_by_string():
    # eyecite alone cannot merge parallel reporters — that needs cluster_id.
    assert cg._canonical_citation_key("410 U.S. 113") != cg._canonical_citation_key("93 S. Ct. 705")


# ──────────────────────────────────────────────────────────────────────────
# Gap 1 — chronology guard
# ──────────────────────────────────────────────────────────────────────────
def test_basic_overrules_makes_bad_law(db, matter, monkeypatch):
    mid, did = matter
    triples = [{
        "citing_citation": NEWER, "citing_name": "NewCase", "citing_year": 2000,
        "cited_citation": OLDER, "cited_name": "OldCase", "cited_year": 1950,
        "treatment": "OVERRULES", "evidence": "overruled OldCase",
    }]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did, triples)
    edges = db.query(CitationEdge).filter(
        CitationEdge.matter_id == mid, CitationEdge.edge_kind == "case_cites").all()
    assert len(edges) == 1 and edges[0].treatment == "OVERRULES"
    assert cg.is_good_law(db, OLDER)["is_good_law"] is False


def test_no_temporally_impossible_negative_edges(db, matter, monkeypatch):
    # Older case (1950) cannot overrule newer (2000): edge must be DROPPED.
    mid, did = matter
    triples = [{
        "citing_citation": OLDER, "citing_name": "OldCase", "citing_year": 1950,
        "cited_citation": NEWER, "cited_name": "NewCase", "cited_year": 2000,
        "treatment": "OVERRULES", "evidence": "(impossible direction)",
    }]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did, triples)
    edges = db.query(CitationEdge).filter(
        CitationEdge.matter_id == mid, CitationEdge.edge_kind == "case_cites").all()
    assert len(edges) == 0, "temporally-impossible negative edge must be dropped"
    # And the newer, controlling case must NOT be flagged bad law.
    assert cg.is_good_law(db, NEWER)["is_good_law"] is not False


def test_chronology_abstains_on_unknown_year(db, matter, monkeypatch):
    # Years unknown (fictional cites have no CourtListener data) -> keep edge but
    # cap confidence (abstain, don't drop).
    mid, did = matter
    triples = [{
        "citing_citation": NEWER, "citing_name": "NewCase", "citing_year": None,
        "cited_citation": OLDER, "cited_name": "OldCase", "cited_year": None,
        "treatment": "OVERRULES", "evidence": "overruled OldCase",
    }]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did, triples)
    edges = db.query(CitationEdge).filter(
        CitationEdge.matter_id == mid, CitationEdge.edge_kind == "case_cites").all()
    assert len(edges) == 1
    assert (edges[0].confidence or 1.0) <= 0.4, "abstain => capped confidence"


# ──────────────────────────────────────────────────────────────────────────
# Gap 4 — self-edge guard
# ──────────────────────────────────────────────────────────────────────────
def test_no_self_edge_from_parallel_form(db, matter, monkeypatch):
    mid, did = matter
    # Same fictional cite in two spacings -> canonical key equal -> self-edge.
    triples = [{
        "citing_citation": "9001 U. S. 10", "cited_citation": "9001 U.S. 10",
        "treatment": "OVERRULES", "evidence": "x",
    }]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did, triples)
    edges = db.query(CitationEdge).filter(CitationEdge.matter_id == mid).all()
    assert all(e.source_id != e.target_id for e in edges)
    assert len(edges) == 0


def test_create_edge_refuses_same_id(db, matter, monkeypatch):
    mid, did = matter
    node = cg.find_or_create_node(db, THIRD, node_type="case", matter_id=mid)
    e = cg.create_edge(db, source_id=node.id, target_id=node.id,
                       treatment="CITES", edge_kind="case_cites")
    assert e is None


def test_edge_dedup_is_matter_scoped(db, matter, monkeypatch):
    """Same relationship in TWO matters must produce an edge row in EACH.

    Nodes are global, but get_matter_graph filters edges by matter_id, so a
    global edge-dedup would make the relationship visible in only the first
    matter. Regression for that bug.
    """
    mid, did = matter
    # Global nodes (no matter), reused across both matters.
    src = cg.find_or_create_node(db, NEWER, node_type="case", year=2000)
    tgt = cg.find_or_create_node(db, OLDER, node_type="case", year=1950)
    # Second throwaway matter.
    mid2 = uuid.uuid4()
    did2 = uuid.uuid4()
    db.add(Matter(id=mid2, name="SCOPED2", status="ready", file_type="txt",
                  blob_storage_path="x", is_deleted=False))
    db.flush()
    db.add(Document(id=did2, matter_id=mid2, name="m2", status="ready",
                    file_type="txt", blob_storage_path="x"))
    db.commit()
    try:
        e1 = cg.create_edge(db, source_id=src.id, target_id=tgt.id,
                            treatment="OVERRULES", edge_kind="case_cites",
                            matter_id=mid, source_node=src, cited_node=tgt)
        e2 = cg.create_edge(db, source_id=src.id, target_id=tgt.id,
                            treatment="OVERRULES", edge_kind="case_cites",
                            matter_id=mid2, source_node=src, cited_node=tgt)
        assert e1 is not None and e2 is not None, "each matter must get its own edge"
        # ...but the SAME matter still dedups.
        e3 = cg.create_edge(db, source_id=src.id, target_id=tgt.id,
                            treatment="OVERRULES", edge_kind="case_cites",
                            matter_id=mid, source_node=src, cited_node=tgt)
        assert e3 is None, "same matter must still dedup"
    finally:
        db.query(CitationEdge).filter(CitationEdge.matter_id == mid2).delete()
        db.query(Document).filter(Document.id == did2).delete()
        db.query(Matter).filter(Matter.id == mid2).delete()
        db.commit()


# ──────────────────────────────────────────────────────────────────────────
# Gap 3 — parallel-cite merge via cluster_id
# ──────────────────────────────────────────────────────────────────────────
def test_parallel_cites_merge_to_one_node(db, matter, monkeypatch):
    mid, did = matter
    # Two triples cite the SAME case via parallel reporters; enrichment maps both
    # to cluster '9999' -> they must collapse to a single node.
    # Fictional parallel reporters for one case; enrichment maps both to the same
    # cluster -> they must collapse to a single node.
    triples = [
        {"citing_citation": NEWER, "citing_name": "NewCase", "citing_year": 2000,
         "cited_citation": "9000 U.S. 20", "cited_name": "OldCase", "cited_year": 1950,
         "treatment": "OVERRULES", "evidence": "overruled OldCase"},
        {"citing_citation": NEWER, "citing_name": "NewCase", "citing_year": 2000,
         "cited_citation": "9500 S. Ct. 99", "cited_name": "OldCase", "cited_year": 1950,
         "treatment": "OVERRULES", "evidence": "overruled OldCase (parallel cite)"},
    ]
    cluster_for = {"9000 U.S. 20": "TEST9999", "9500 S. Ct. 99": "TEST9999"}
    _patch_no_network(monkeypatch, triples, cluster_for=cluster_for)
    _index(db, mid, did, triples)
    merged = db.query(CitationNode).filter(
        CitationNode.matter_id == mid, CitationNode.courtlistener_id == "TEST9999").all()
    assert len(merged) == 1, "parallel cites must merge onto one cluster node"


# ──────────────────────────────────────────────────────────────────────────
# Gap 2 — negation verifier downgrades false positive
# ──────────────────────────────────────────────────────────────────────────
def test_negation_false_positive_downgraded(db, matter, monkeypatch):
    mid, did = matter
    triples = [{
        "citing_citation": NEWER, "citing_name": "A", "citing_year": 2000,
        "cited_citation": OLDER, "cited_name": "B", "cited_year": 1980,
        "treatment": "OVERRULES", "evidence": "distinguished B but did not overrule it",
    }]
    _patch_no_network(monkeypatch, triples)
    # Override verifier to REJECT the negative claim.
    import backend.services.treatment_verifier as tv
    async def reject(*a, **k):
        return ("rejected", 0.85)
    monkeypatch.setattr(tv, "verify_negative_treatment", reject)
    _index(db, mid, did, triples)
    edges = db.query(CitationEdge).filter(
        CitationEdge.matter_id == mid, CitationEdge.edge_kind == "case_cites").all()
    assert len(edges) == 1
    assert edges[0].treatment == "CITES", "rejected negative must downgrade to CITES"
    assert cg.is_good_law(db, OLDER)["is_good_law"] is not False


def test_window_dedup_no_duplicate_edges(db, matter, monkeypatch):
    # Same triple twice (e.g. from overlapping windows) -> exactly 1 edge.
    mid, did = matter
    t = {"citing_citation": NEWER, "citing_name": "NewCase", "citing_year": 2000,
         "cited_citation": OLDER, "cited_name": "OldCase", "cited_year": 1950,
         "treatment": "OVERRULES", "evidence": "overruled OldCase"}
    _patch_no_network(monkeypatch, [t, dict(t)])
    _index(db, mid, did, [t, dict(t)])
    edges = db.query(CitationEdge).filter(
        CitationEdge.matter_id == mid, CitationEdge.edge_kind == "case_cites").all()
    assert len(edges) == 1


# ──────────────────────────────────────────────────────────────────────────
# Gap 5 + API parity
# ──────────────────────────────────────────────────────────────────────────
def test_api_db_edge_parity_no_dangling(db, matter, monkeypatch):
    mid, did = matter
    triples = [
        {"citing_citation": NEWER, "citing_name": "NewCase", "citing_year": 2000,
         "cited_citation": OLDER, "cited_name": "OldCase", "cited_year": 1950,
         "treatment": "OVERRULES", "evidence": "overruled OldCase"},
        {"citing_citation": NEWER, "citing_name": "NewCase", "citing_year": 2000,
         "cited_citation": THIRD, "cited_name": "ThirdCase", "cited_year": 1960,
         "treatment": "OVERRULES", "evidence": "overruled ThirdCase"},
    ]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did, triples)
    g = cg.get_matter_graph(db, mid)
    node_ids = {n["id"] for n in g["nodes"]}
    dangling = [e for e in g["edges"]
                if e["source_id"] not in node_ids or e["target_id"] not in node_ids]
    assert dangling == [], "every API edge endpoint must be a returned node"
    db_edges = db.query(CitationEdge).filter(CitationEdge.matter_id == mid).count()
    assert len(g["edges"]) == db_edges, "API edge count must equal DB edge count"


def test_pagerank_attributes_all_edge_endpoints(db, matter, monkeypatch):
    mid, did = matter
    triples = [{
        "citing_citation": NEWER, "citing_name": "NewCase", "citing_year": 2000,
        "cited_citation": OLDER, "cited_name": "OldCase", "cited_year": 1950,
        "treatment": "OVERRULES", "evidence": "overruled OldCase"}]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did, triples)
    graph = cg.build_networkx_graph(db, matter_id=mid, edge_kind="case_cites")
    for u, v in graph.edges():
        assert u in graph.nodes and v in graph.nodes, "no orphan edge endpoints"
