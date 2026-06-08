"""Exhaustive PIPELINE edge-case harness for the citation graph.

Complements ``test_citation_graph_correctness.py`` (Gaps 1-6) by covering the
edge-case matrix categories that the original 16-test suite does NOT touch:

  Cat 1  Empty / malformed documents (0 nodes/edges, no crash, error handling)
  Cat 2  Citation extraction (statute/regulation node_type mapping)
  Cat 3  Relationship / treatment classification (multi-object, cycle,
         chronology, unknown verb, dedup, wrapper keys)
  Cat 4  Good-law combinatorics (OVERRULES+FOLLOWS, doc_cites-only, unknown)
  Cat 5  Graph scale & structure (0/1/2 node, parity, mixed edge_kinds)
  Cat 7  Multi-matter isolation (edge dedup matter-scoped, no leak)

Runs against LIVE Postgres with a throwaway Matter/Document fully cleaned up.
ALL LLM + network entry points are monkeypatched so the suite uses ZERO
Gemini/Groq/CourtListener quota and is deterministic.

Patterns mirror test_citation_graph_correctness.py exactly: module-scoped
_engine reading the real DATABASE_URL, a per-test throwaway Matter+Document
fixture with cleanup, FICTIONAL reporter cites ('9001 U.S. 10' / '9500 S. Ct.
99') to stay isolated from the global node table, and _patch_no_network
monkeypatching cg.extract_relationships_llm / cg.classify_treatment /
cg.extract_all_citations / cg.enrich_node_from_courtlistener and
treatment_verifier.verify_negative_treatment.

Run: cd backend && PYTHONPATH=.. venv/bin/python -m pytest \
     tests/test_citation_graph_edge_cases.py -v --no-header -p no:cacheprovider
"""
import asyncio
import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import backend.services.citation_graph as cg
from backend.services import citation_extractor as ce
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


def _make_matter(db, name="EDGECASE TEST"):
    """Create a throwaway matter + document; returns (mid, did)."""
    mid = uuid.uuid4()
    did = uuid.uuid4()
    m = Matter(id=mid, name=name, status="ready", file_type="txt",
               blob_storage_path="x", is_deleted=False)
    db.add(m)
    db.flush()
    d = Document(id=did, matter_id=mid, name="memo", status="ready",
                 file_type="txt", blob_storage_path="x")
    db.add(d)
    db.commit()
    return mid, did


def _cleanup_matter(db, mid, did):
    db.query(CitationEdge).filter(CitationEdge.matter_id == mid).delete()
    db.query(CitationNode).filter(CitationNode.matter_id == mid).delete()
    db.query(Document).filter(Document.id == did).delete()
    db.query(Matter).filter(Matter.id == mid).delete()
    db.commit()


@pytest.fixture()
def matter(db):
    """A throwaway matter + document, cleaned up (with its graph rows) after."""
    mid, did = _make_matter(db)
    yield mid, did
    _cleanup_matter(db, mid, did)


def _patch_no_network(monkeypatch, triples, *, cluster_for=None, found_citations=None,
                      verify=None):
    """Patch every LLM/network entry point.

    ``triples``           : what the relationship extractor returns.
    ``cluster_for``       : maps citation string -> cluster_id (parallel-cite sim).
    ``found_citations``   : per-chunk citations extract_all_citations returns
                            (default [] — no document_cites noise).
    ``verify``            : optional async verifier override; default neutral.
    """
    async def fake_rel(_text):
        return list(triples)

    async def fake_classify(*a, **k):
        return {"treatment": "CITES", "confidence": 0.5}

    async def fake_extract_all(*a, **k):
        return list(found_citations) if found_citations is not None else []

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
    monkeypatch.setattr(tv, "verify_negative_treatment", verify or fake_verify)


def _index(db, mid, did, chunks=None):
    if chunks is None:
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
FOURTH = "9003 U.S. 40"


def _case_cites(db, mid):
    return db.query(CitationEdge).filter(
        CitationEdge.matter_id == mid, CitationEdge.edge_kind == "case_cites").all()


def _triple(citing, cited, treatment, *, cy=2000, dy=1950, ev="x",
            citing_name="C", cited_name="D"):
    return {
        "citing_citation": citing, "citing_name": citing_name, "citing_year": cy,
        "cited_citation": cited, "cited_name": cited_name, "cited_year": dy,
        "treatment": treatment, "evidence": ev,
    }


# ══════════════════════════════════════════════════════════════════════════
# Category 1 — Empty / malformed documents
# ══════════════════════════════════════════════════════════════════════════
def test_empty_001_empty_chunks_no_edges(db, matter, monkeypatch):
    """EMPTY-001: empty chunk list -> only the doc node, 0 edges."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    result = _index(db, mid, did, chunks=[])
    assert result["edges_created"] == 0
    edges = db.query(CitationEdge).filter(CitationEdge.matter_id == mid).all()
    assert len(edges) == 0
    nodes = db.query(CitationNode).filter(CitationNode.matter_id == mid).all()
    assert len(nodes) == 1, "only the document node should exist"


def test_empty_001b_empty_content_string(db, matter, monkeypatch):
    """EMPTY-001 variant: single chunk with empty content string."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    result = _index(db, mid, did, chunks=[{"content": ""}])
    assert result["edges_created"] == 0
    assert db.query(CitationEdge).filter(CitationEdge.matter_id == mid).count() == 0


def test_empty_002_whitespace_only(db, matter, monkeypatch):
    """EMPTY-002: whitespace-only content -> chunk skipped, mining skipped, 0 edges."""
    mid, did = matter

    called = {"rel": False}

    async def spy_rel(_text):
        called["rel"] = True
        return []

    _patch_no_network(monkeypatch, [])
    monkeypatch.setattr(cg, "extract_relationships_llm", spy_rel)
    result = _index(db, mid, did, chunks=[{"content": "   \n\t  "}])
    assert result["edges_created"] == 0
    assert called["rel"] is False, "joined_text empty -> relationship mining skipped"


def test_empty_003_prose_zero_citations(db, matter, monkeypatch):
    """EMPTY-003: prose with no citations -> doc node only, 0 edges."""
    mid, did = matter
    _patch_no_network(monkeypatch, [], found_citations=[])
    result = _index(db, mid, did,
                    chunks=[{"content": "This memo discusses general policy with no cites."}])
    assert result["edges_created"] == 0
    assert _case_cites(db, mid) == []


def test_empty_004_binary_garbage_no_crash(db, matter, monkeypatch):
    """EMPTY-004: binary garbage content must not crash; extractors return []."""
    mid, did = matter
    _patch_no_network(monkeypatch, [], found_citations=[])
    result = _index(db, mid, did, chunks=[{"content": "\x00\xff\x01garbage\x07\x1b"}])
    assert result["errors"] == 0
    assert result["edges_created"] == 0


def test_empty_005_extraction_raises_midchunk(db, matter, monkeypatch):
    """EMPTY-005: extract_all_citations raises on one chunk -> errors++, loop continues."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    state = {"n": 0}

    async def flaky_extract(content, *a, **k):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("boom on chunk 2")
        return []

    monkeypatch.setattr(cg, "extract_all_citations", flaky_extract)
    chunks = [{"content": "alpha"}, {"content": "beta"}, {"content": "gamma"}]
    result = _index(db, mid, did, chunks=chunks)
    assert result["errors"] >= 1, "raising chunk must increment errors"
    assert state["n"] == 3, "loop must continue over remaining chunks"


def test_empty_006_commit_failure_returns_zeroed(db, matter, monkeypatch):
    """EMPTY-006: db.commit failure -> rollback + zeroed-count envelope w/ errors+1."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])

    real_commit = db.commit
    state = {"raised": False}

    def boom_commit():
        # Fail exactly once (the indexing commit), then restore so the fixture
        # cleanup can still commit normally.
        if not state["raised"]:
            state["raised"] = True
            db.commit = real_commit
            raise RuntimeError("commit kaboom")
        return real_commit()

    monkeypatch.setattr(db, "commit", boom_commit)
    result = _index(db, mid, did, chunks=[{"content": "stub"}])
    assert result == {"nodes_created": 0, "nodes_updated": 0,
                      "edges_created": 0, "errors": result["errors"]}
    assert result["errors"] >= 1


# ══════════════════════════════════════════════════════════════════════════
# Category 2 — Citation extraction (node_type mapping)
# ══════════════════════════════════════════════════════════════════════════
def test_ext_005_statute_and_regulation_node_types(db, matter, monkeypatch):
    """EXT-005: type 'statute'->statute, 'regulation'->regulation, else 'case'."""
    mid, did = matter
    found = [
        {"raw_text": "42 U.S.C. 9983", "type": "statute", "case_name": "Some Act"},
        {"raw_text": "29 C.F.R. 9604", "type": "regulation", "case_name": "Some Reg"},
        {"raw_text": NEWER, "type": "case_law", "case_name": "Some Case"},
    ]
    _patch_no_network(monkeypatch, [], found_citations=found)
    _index(db, mid, did, chunks=[{"content": "cites 42 U.S.C. 9983 and 29 C.F.R. 9604 and a case"}])

    def ntype(text):
        n = db.query(CitationNode).filter(
            CitationNode.citation_text == cg._canonical_citation_key(text)).first()
        return n.node_type if n else None

    assert ntype("42 U.S.C. 9983") == "statute"
    assert ntype("29 C.F.R. 9604") == "regulation"
    assert ntype(NEWER) == "case"


def test_ext_009_duplicate_cites_one_doc_edge(db, matter, monkeypatch):
    """EXT-009: same cite twice in a chunk -> one node + one document_cites edge."""
    mid, did = matter
    found = [
        {"raw_text": NEWER, "type": "case_law", "case_name": "Dup"},
        {"raw_text": NEWER, "type": "case_law", "case_name": "Dup"},
    ]
    _patch_no_network(monkeypatch, [], found_citations=found)
    _index(db, mid, did, chunks=[{"content": f"see {NEWER} ... and again {NEWER}"}])
    nodes = db.query(CitationNode).filter(
        CitationNode.citation_text == cg._canonical_citation_key(NEWER)).all()
    assert len(nodes) == 1
    doc_edges = db.query(CitationEdge).filter(
        CitationEdge.matter_id == mid, CitationEdge.edge_kind == "document_cites").all()
    assert len(doc_edges) == 1, "duplicate cite must dedup to a single document_cites edge"


# ══════════════════════════════════════════════════════════════════════════
# Category 3 — Relationship / treatment classification
# ══════════════════════════════════════════════════════════════════════════
def test_rel_003_multi_object_overrule_two_edges(db, matter, monkeypatch):
    """REL-003: 'Dobbs overruled Roe and Casey' -> two triples -> two edges, both bad."""
    mid, did = matter
    triples = [
        _triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950, cited_name="Roe"),
        _triple(NEWER, THIRD, "OVERRULES", cy=2000, dy=1960, cited_name="Casey"),
    ]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    edges = _case_cites(db, mid)
    assert len(edges) == 2
    assert cg.is_good_law(db, OLDER)["is_good_law"] is False
    assert cg.is_good_law(db, THIRD)["is_good_law"] is False


def test_rel_007_cycle_two_edges_no_infinite_loop(db, matter, monkeypatch):
    """REL-007: A<->B cycle (consistent chronology) -> both edges; is_good_law &
    PageRank converge (no infinite loop, finite scores)."""
    mid, did = matter
    # A=NEWER(2000), B=OLDER(1990). A overrules B (ok: 2000>=1990).
    # B overrules a THIRD older sibling to make a cited<->citing cycle without a
    # chronology violation: B(1990) overrules C(1980); C... we instead build a
    # genuine 2-cycle with both years known and consistent by using equal years.
    triples = [
        _triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=2000, cited_name="B"),
        _triple(OLDER, NEWER, "OVERRULES", cy=2000, dy=2000, cited_name="A"),
    ]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    edges = _case_cites(db, mid)
    assert len(edges) == 2, "equal-year cycle: both negative edges pass chronology (>=)"
    # is_good_law must terminate (no infinite loop) for both endpoints.
    assert cg.is_good_law(db, NEWER)["is_good_law"] is False
    assert cg.is_good_law(db, OLDER)["is_good_law"] is False
    # PageRank must converge / fall back and return finite scores for both.
    scores = cg.compute_authority_scores(db, mid)
    assert len(scores) >= 2
    assert all(isinstance(v, float) and v == v for v in scores.values())


def test_rel_008_chronology_violation_dropped(db, matter, monkeypatch):
    """REL-008: older(1950) OVERRULES newer(2000) -> edge dropped, newer not bad."""
    mid, did = matter
    triples = [_triple(OLDER, NEWER, "OVERRULES", cy=1950, dy=2000)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    assert _case_cites(db, mid) == []
    assert cg.is_good_law(db, NEWER)["is_good_law"] is not False


def test_rel_010_chronology_guard_disabled_keeps_edge(db, matter, monkeypatch):
    """REL-010: guard disabled -> impossible negative edge is NOT dropped."""
    mid, did = matter
    triples = [_triple(OLDER, NEWER, "OVERRULES", cy=1950, dy=2000)]
    _patch_no_network(monkeypatch, triples)
    # MagicMock settings -> set the attribute to False explicitly.
    settings = cg.get_settings()
    monkeypatch.setattr(settings, "citation_graph_chronology_guard_enabled", False,
                        raising=False)
    _index(db, mid, did)
    assert len(_case_cites(db, mid)) == 1, "guard off -> impossible edge kept"


def test_rel_011_unknown_treatment_verb_skipped(db, matter, monkeypatch):
    """REL-011: bogus treatment 'MENTIONED' -> _canonical_treatment None -> dropped."""
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "MENTIONED")]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    assert _case_cites(db, mid) == []


@pytest.mark.parametrize("alias,canonical", [
    ("overruled", "OVERRULES"),
    ("OVERRULING", "OVERRULES"),
    ("follows", "FOLLOWS"),
    ("distinguished", "DISTINGUISHES"),
    ("cited", "CITES"),
])
def test_rel_012_treatment_aliases(db, matter, monkeypatch, alias, canonical):
    """REL-012: tense/voice aliases canonicalize and create an edge w/ canonical token."""
    mid, did = matter
    # Use consistent chronology so negative aliases survive the chronology guard.
    triples = [_triple(NEWER, OLDER, alias, cy=2000, dy=1950)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    edges = _case_cites(db, mid)
    assert len(edges) == 1
    assert edges[0].treatment == canonical


def test_rel_013_positive_only_follows_good(db, matter, monkeypatch):
    """REL-013: FOLLOWS only -> status 'good', is_good_law True."""
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "FOLLOWS", cy=2000, dy=1950)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    res = cg.is_good_law(db, OLDER)
    assert res["status"] == "good"
    assert res["is_good_law"] is True


def test_rel_014_cautionary_only_caution(db, matter, monkeypatch):
    """REL-014: QUESTIONS only -> status 'caution', is_good_law None."""
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "QUESTIONS", cy=2000, dy=1950)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    res = cg.is_good_law(db, OLDER)
    assert res["status"] == "caution"
    assert res["is_good_law"] is None


def test_rel_015_partial_triple_dropped(db, matter, monkeypatch):
    """REL-015: triple missing cited_citation/name -> dropped by _parse... ."""
    # This is exercised at the parse layer (extract_relationships_llm is stubbed
    # in indexing), so assert _parse_relationship_items directly.
    items = [{"citing_citation": NEWER, "citing_name": "A",
              "cited_citation": "", "cited_name": "", "treatment": "OVERRULES"}]
    assert cg._parse_relationship_items(items) == []


def test_rel_017_wrapper_key_unwrapping(db, matter, monkeypatch):
    """REL-017: triples wrapped under non-standard keys are unwrapped by parser."""
    triple = _triple(NEWER, OLDER, "OVERRULES")
    for wrapper in ({"inter_case_relationships": [triple]},
                    {"relationships": [triple]},
                    {"foo": [triple]}):
        parsed = cg._parse_relationship_items(wrapper)
        assert len(parsed) == 1, f"wrapper {list(wrapper)[0]!r} must parse one triple"
        assert parsed[0]["treatment"] == "OVERRULES"


# ══════════════════════════════════════════════════════════════════════════
# Category 4 — Good-law logic
# ══════════════════════════════════════════════════════════════════════════
def test_gl_002_overrules_plus_follows_negative_wins(db, matter, monkeypatch):
    """GL-002: incoming OVERRULES + FOLLOWS -> negative wins, status 'bad', False."""
    mid, did = matter
    triples = [
        _triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950, citing_name="Over"),
        _triple(THIRD, OLDER, "FOLLOWS", cy=2010, dy=1950, citing_name="Foll"),
    ]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    res = cg.is_good_law(db, OLDER)
    assert res["status"] == "bad"
    assert res["is_good_law"] is False


def test_gl_003_no_incoming_case_cites_unknown_flag(db, matter, monkeypatch):
    """GL-003: node with no incoming case_cites -> status 'good' but flag None."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    # Create an isolated node with no incoming edges.
    cg.find_or_create_node(db, OLDER, node_type="case", matter_id=mid, year=1950)
    db.commit()
    res = cg.is_good_law(db, OLDER)
    assert res["status"] == "good"
    assert res["is_good_law"] is None


def test_gl_004_document_cites_only_never_flips_good_law(db, matter, monkeypatch):
    """GL-004 (CRITICAL): a document_cites edge tagged OVERRULES must NOT make a
    case bad law. is_good_law filters edge_kind=='case_cites'."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    doc = cg.find_or_create_node(db, "memo-doc", node_type="case", matter_id=mid)
    target = cg.find_or_create_node(db, OLDER, node_type="case", matter_id=mid, year=1950)
    db.flush()
    cg.create_edge(db, source_id=doc.id, target_id=target.id,
                   treatment="OVERRULES", edge_kind="document_cites", matter_id=mid)
    db.commit()
    res = cg.is_good_law(db, OLDER)
    assert res["is_good_law"] is not False, "document_cites must never set bad law"
    assert res["status"] == "good"


def test_gl_005_unknown_citation_envelope(db, matter, monkeypatch):
    """GL-005: citation absent from graph -> status 'unknown', flag None, message."""
    res = cg.is_good_law(db, "9999 U.S. 9999")
    assert res["status"] == "unknown"
    assert res["is_good_law"] is None
    assert "message" in res


def test_gl_008_lookup_uses_canonical_key(db, matter, monkeypatch):
    """GL-008: store canonical, query spaced variant -> found (same node)."""
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    # OLDER stored as canonical '9000 U.S. 20'; query the spaced variant.
    res = cg.is_good_law(db, "9000 U. S. 20")
    assert res["status"] == "bad", "spaced query variant must resolve to canonical node"
    assert res["is_good_law"] is False


# ══════════════════════════════════════════════════════════════════════════
# Category 5 — Graph scale & structure
# ══════════════════════════════════════════════════════════════════════════
def test_scale_001_zero_node_matter(db, matter, monkeypatch):
    """SCALE-001: matter with no rows -> empty envelope, compute returns {}."""
    mid, did = matter
    g = cg.get_matter_graph(db, mid)
    assert g["nodes"] == [] and g["edges"] == []
    assert g["stats"] == {"total_nodes": 0, "total_edges": 0}
    assert cg.compute_authority_scores(db, mid) == {}


def test_scale_002_one_node_no_edge(db, matter, monkeypatch):
    """SCALE-002: 1 node, 0 edges -> node returned, total_edges 0, PageRank ~1.0."""
    mid, did = matter
    cg.find_or_create_node(db, NEWER, node_type="case", matter_id=mid, year=2000)
    db.commit()
    g = cg.get_matter_graph(db, mid)
    assert g["stats"]["total_edges"] == 0
    assert g["stats"]["total_nodes"] == 1


def test_scale_003_two_nodes_one_edge_parity(db, matter, monkeypatch):
    """SCALE-003: 2 nodes 1 edge -> API edge count == DB; no dangling endpoints."""
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    g = cg.get_matter_graph(db, mid)
    node_ids = {n["id"] for n in g["nodes"]}
    dangling = [e for e in g["edges"]
                if e["source_id"] not in node_ids or e["target_id"] not in node_ids]
    assert dangling == []
    db_edges = db.query(CitationEdge).filter(CitationEdge.matter_id == mid).count()
    assert len(g["edges"]) == db_edges


def test_scale_005_disconnected_components(db, matter, monkeypatch):
    """SCALE-005: two separate case_cites pairs -> all 4 nodes + 2 edges returned."""
    mid, did = matter
    triples = [
        _triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950),
        _triple(THIRD, FOURTH, "OVERRULES", cy=2005, dy=1960),
    ]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    g = cg.get_matter_graph(db, mid)
    # 4 case nodes + the document node.
    case_texts = {cg._canonical_citation_key(c) for c in (NEWER, OLDER, THIRD, FOURTH)}
    present = {n["citation_text"] for n in g["nodes"]}
    assert case_texts.issubset(present)
    assert g["stats"]["total_edges"] == 2


def test_scale_007_mixed_edge_kinds_coexist_and_dedup(db, matter, monkeypatch):
    """SCALE-007: same (src,tgt) with different edge_kind coexist; exact dup dedups."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    src = cg.find_or_create_node(db, NEWER, node_type="case", matter_id=mid, year=2000)
    tgt = cg.find_or_create_node(db, OLDER, node_type="case", matter_id=mid, year=1950)
    db.flush()
    e1 = cg.create_edge(db, source_id=src.id, target_id=tgt.id, treatment="CITES",
                        edge_kind="document_cites", matter_id=mid)
    e2 = cg.create_edge(db, source_id=src.id, target_id=tgt.id, treatment="CITES",
                        edge_kind="case_cites", matter_id=mid,
                        source_node=src, cited_node=tgt)
    e3 = cg.create_edge(db, source_id=src.id, target_id=tgt.id, treatment="CITES",
                        edge_kind="case_cites", matter_id=mid,
                        source_node=src, cited_node=tgt)
    db.commit()
    assert e1 is not None and e2 is not None, "different edge_kind must coexist"
    assert e3 is None, "exact duplicate must dedup"


def test_scale_008_orphaned_edge_filtered(db, matter, monkeypatch):
    """SCALE-008: endpoint node deleted out from under an edge -> get_matter_graph
    defensively drops the edge (no dangling)."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    src = cg.find_or_create_node(db, NEWER, node_type="case", matter_id=mid, year=2000)
    tgt = cg.find_or_create_node(db, OLDER, node_type="case", matter_id=mid, year=1950)
    db.flush()
    cg.create_edge(db, source_id=src.id, target_id=tgt.id, treatment="CITES",
                   edge_kind="case_cites", matter_id=mid)
    db.commit()
    # Delete the target node, leaving the edge orphaned.
    db.query(CitationNode).filter(CitationNode.id == tgt.id).delete()
    db.commit()
    g = cg.get_matter_graph(db, mid)
    node_ids = {n["id"] for n in g["nodes"]}
    dangling = [e for e in g["edges"]
                if e["source_id"] not in node_ids or e["target_id"] not in node_ids]
    assert dangling == [], "orphaned edge must be filtered out"


def test_scale_009_pagerank_never_raises(db, matter, monkeypatch):
    """SCALE-009: compute_authority_scores never raises; returns finite scores."""
    mid, did = matter
    triples = [
        _triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=2000),
        _triple(OLDER, NEWER, "OVERRULES", cy=2000, dy=2000),
    ]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    scores = cg.compute_authority_scores(db, mid)
    assert all(isinstance(v, float) and v == v for v in scores.values())


# ══════════════════════════════════════════════════════════════════════════
# Category 7 — Multi-matter isolation
# ══════════════════════════════════════════════════════════════════════════
def test_mm_002_shared_node_matter_scoped_edges(db, matter, monkeypatch):
    """MM-002/MM-003: shared global node, matter-scoped edges; no cross-matter leak.

    get_matter_graph(mid1) and (mid2) each show the SAME global node pair but
    ONLY their own matter's edges.
    """
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    # Global nodes (no matter_id) reused across both matters.
    src = cg.find_or_create_node(db, NEWER, node_type="case", year=2000)
    tgt = cg.find_or_create_node(db, OLDER, node_type="case", year=1950)
    db.flush()
    mid2, did2 = _make_matter(db, name="MM2")
    try:
        e1 = cg.create_edge(db, source_id=src.id, target_id=tgt.id,
                            treatment="OVERRULES", edge_kind="case_cites",
                            matter_id=mid, source_node=src, cited_node=tgt)
        e2 = cg.create_edge(db, source_id=src.id, target_id=tgt.id,
                            treatment="OVERRULES", edge_kind="case_cites",
                            matter_id=mid2, source_node=src, cited_node=tgt)
        # Same edge asserted twice in mid -> dedups to None.
        e3 = cg.create_edge(db, source_id=src.id, target_id=tgt.id,
                            treatment="OVERRULES", edge_kind="case_cites",
                            matter_id=mid, source_node=src, cited_node=tgt)
        db.commit()
        assert e1 is not None and e2 is not None, "each matter gets its own edge row"
        assert e3 is None, "same matter must still dedup"

        g1 = cg.get_matter_graph(db, mid)
        g2 = cg.get_matter_graph(db, mid2)
        # Each matter's edge list contains ONLY its own matter_id rows.
        assert all(e["matter_id"] == str(mid) for e in g1["edges"])
        assert all(e["matter_id"] == str(mid2) for e in g2["edges"])
        assert len(g1["edges"]) == 1 and len(g2["edges"]) == 1
        # The shared global node appears in BOTH graphs (unioned via endpoints).
        g1_nodes = {n["id"] for n in g1["nodes"]}
        g2_nodes = {n["id"] for n in g2["nodes"]}
        assert str(src.id) in g1_nodes and str(src.id) in g2_nodes
        assert str(tgt.id) in g1_nodes and str(tgt.id) in g2_nodes
    finally:
        # Edges/nodes created here are global or in mid2; clean both explicitly.
        db.query(CitationEdge).filter(CitationEdge.source_id == src.id).delete()
        _cleanup_matter(db, mid2, did2)
        db.query(CitationNode).filter(CitationNode.id.in_([src.id, tgt.id])).delete()
        db.commit()


def test_mm_004_pagerank_scoped_per_matter(db, matter, monkeypatch):
    """MM-004/MM-005: PageRank scoped per matter; cross-matter endpoint not dropped."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    src = cg.find_or_create_node(db, NEWER, node_type="case", year=2000)
    tgt = cg.find_or_create_node(db, OLDER, node_type="case", year=1950)
    db.flush()
    mid2, did2 = _make_matter(db, name="MM4")
    try:
        cg.create_edge(db, source_id=src.id, target_id=tgt.id, treatment="OVERRULES",
                       edge_kind="case_cites", matter_id=mid,
                       source_node=src, cited_node=tgt)
        db.commit()
        # build_networkx_graph(mid) must union in src/tgt even though their own
        # matter_id is None (global) — no orphan edge endpoints.
        graph = cg.build_networkx_graph(db, matter_id=mid, edge_kind="case_cites")
        assert str(src.id) in graph.nodes and str(tgt.id) in graph.nodes
        for u, v in graph.edges():
            assert u in graph.nodes and v in graph.nodes
        # mid2 has no edges -> empty graph for it.
        g2 = cg.build_networkx_graph(db, matter_id=mid2, edge_kind="case_cites")
        assert g2.number_of_edges() == 0
    finally:
        db.query(CitationEdge).filter(CitationEdge.source_id == src.id).delete()
        _cleanup_matter(db, mid2, did2)
        db.query(CitationNode).filter(CitationNode.id.in_([src.id, tgt.id])).delete()
        db.commit()


# ══════════════════════════════════════════════════════════════════════════
# Category 8 — Key consistency (write-key == read-key, no-regex canonical)
#
# Headline category from the QA matrix. find_or_create_node WRITES citation_text
# via _canonical_citation_key (eyecite). Every reader MUST resolve with the same
# canonical key. These tests prove no path falls back to the banned reporter
# regex (_normalise_text) for citation IDENTITY — including the rag_engine
# good-law injection path (the previously-confirmed silent good-law-loss bug).
# ══════════════════════════════════════════════════════════════════════════
def test_key_001_write_key_equals_read_key_dotless(db, matter, monkeypatch):
    """KEY-001: store via dotless 'NNNN US PP', resolve via canonical + spaced
    + dotted variants -> all hit the same node (write-key == read-key)."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    # Write with the dotless variant eyecite "fixes".
    node = cg.find_or_create_node(db, "9001 US 10", node_type="case",
                                  matter_id=mid, year=2000)
    db.commit()
    canonical = cg._canonical_citation_key("9001 US 10")
    assert canonical == "9001 U.S. 10", "eyecite canonicalises dotless -> dotted"
    assert node.citation_text == canonical, "node stored under the canonical key"
    # Every spelling variant must resolve to the SAME stored node.
    for variant in ("9001 U.S. 10", "9001 U. S. 10", "9001 US 10"):
        found = db.query(CitationNode).filter(
            CitationNode.citation_text == cg._canonical_citation_key(variant)).one()
        assert found.id == node.id, f"{variant!r} must resolve to the canonical node"


def test_key_002_is_good_law_resolves_all_variants(db, matter, monkeypatch):
    """KEY-002: a node made bad-law via a real OVERRULES edge is found by
    is_good_law() through dotless / spaced / dotted query forms (read uses the
    canonical key, matching the writer)."""
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    # OLDER stored canonical; query every variant including the dotless one
    # that the old _normalise_text path would have MISSED.
    for variant in ("9000 U.S. 20", "9000 U. S. 20", "9000 US 20"):
        res = cg.is_good_law(db, variant)
        assert res["is_good_law"] is False, f"variant {variant!r} must resolve to bad law"
        assert res["status"] == "bad"


def test_key_003_rag_engine_graph_context_resolves_dotless(db, matter, monkeypatch):
    """KEY-003 (CONFIRMED-FIX REGRESSION): the rag_engine good-law injection path
    must resolve nodes with _canonical_citation_key, NOT _normalise_text.

    Stores a BAD-LAW node under the canonical key, then feeds _build_graph_context
    the DOTLESS raw_text ('9000 US 20'). Pre-fix, _normalise_text('9000 US 20')
    == '9000 US 20' != stored '9000 U.S. 20' -> node dropped, BAD-LAW warning
    never emitted. Post-fix the block must contain the BAD LAW line."""
    from backend.services.rag_engine import _build_graph_context
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950, cited_name="Older")]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    # Feed the dotless spelling as the chunk-extracted citation.
    block = _build_graph_context(
        db, str(mid), citations=[{"raw_text": "9000 US 20"}], max_citations=5)
    assert block is not None, "dotless cite must resolve to the canonical bad-law node"
    assert "BAD LAW" in block, "the bad-law warning must reach the LLM context"


def test_key_004_rag_engine_dedup_uses_canonical(db, matter, monkeypatch):
    """KEY-004: two spellings of ONE authority in the citation list must dedup to
    a single emitted validity line (canonical-based dedup), not be double-counted."""
    from backend.services.rag_engine import _build_graph_context
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950, cited_name="Older")]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    block = _build_graph_context(
        db, str(mid),
        citations=[{"raw_text": "9000 US 20"},
                   {"raw_text": "9000 U.S. 20"},
                   {"raw_text": "9000 U. S. 20"}],
        max_citations=5)
    assert block is not None
    # Count emitted bullet lines (the header also contains the words "BAD LAW",
    # so count "• " bullets, not the substring). Dedup -> exactly one bullet.
    bullets = [ln for ln in block.splitlines() if ln.lstrip().startswith("•")]
    assert len(bullets) == 1, f"spacing/dot variants must dedup to one line: {bullets!r}"
    assert "BAD LAW" in bullets[0]


def test_key_005_normalise_text_no_longer_mangles_reporters():
    """KEY-005: _normalise_text must NOT corrupt reporter dots/spacing. It is now a
    whitespace-collapse-only fallback (the banned reporter-spacing regex is gone)."""
    # 'F. Supp.' / 'S. Ct.' must be preserved verbatim (only whitespace collapsed),
    # never rewritten to 'F.Supp.' / 'S.Ct.' as the old regex did.
    assert cg._normalise_text("F. Supp.") == "F. Supp."
    assert cg._normalise_text("S. Ct.") == "S. Ct."
    assert cg._normalise_text("347   U.S.   483") == "347 U.S. 483"  # collapse only
    assert cg._normalise_text("  410\tU.S.\n113  ") == "410 U.S. 113"


# ══════════════════════════════════════════════════════════════════════════
# Category 4b — Good-law: CourtListener authoritative override (GL-006)
# ══════════════════════════════════════════════════════════════════════════
def test_gl_006_courtlistener_bad_law_not_flipped_by_positive_edges(db, matter, monkeypatch):
    """GL-006 (CRITICAL): a CourtListener-authoritative bad-law node must STAY bad
    even when the local graph holds ONLY positive (FOLLOWS) edges and no negative
    ones. is_good_law must surface the authoritative flag, never the positive-only
    graph guess, and must NOT mutate the authoritative row from the read path."""
    mid, did = matter
    # FOLLOWS edge would, on its own, make OLDER "good"/True.
    triples = [_triple(NEWER, OLDER, "FOLLOWS", cy=2000, dy=1950)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    # Simulate CourtListener authoritative enrichment: bad_law.
    node = db.query(CitationNode).filter(
        CitationNode.citation_text == cg._canonical_citation_key(OLDER)).one()
    node.is_good_law = False
    node.is_verified = True
    node.courtlistener_id = "9999CL_TESTONLY"
    db.commit()

    res = cg.is_good_law(db, OLDER)
    assert res["is_good_law"] is False, "authoritative bad_law must not flip to True"
    assert res["status"] == "bad"
    # Read path must NOT have mutated the authoritative flag.
    db.refresh(node)
    assert node.is_good_law is False
    assert res["positive_count"] >= 1, "the FOLLOWS edge still exists, just doesn't win"


def test_gl_006b_courtlistener_good_not_flipped_to_unknown(db, matter, monkeypatch):
    """GL-006 variant: authoritative good_law with NO incoming case_cites must stay
    True (surfacing the CL flag), not collapse to None as a bare graph guess would."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    node = cg.find_or_create_node(db, OLDER, node_type="case", matter_id=mid, year=1950)
    node.is_good_law = True
    node.is_verified = True
    node.courtlistener_id = "9998CL_TESTONLY"
    db.commit()
    res = cg.is_good_law(db, OLDER)
    assert res["is_good_law"] is True, "authoritative good_law surfaced despite no edges"
    assert res["status"] == "good"


def test_gl_006c_real_negative_edge_still_downgrades_authoritative(db, matter, monkeypatch):
    """GL-006 boundary: a REAL adverse case_cites edge MUST still produce bad law
    even on a node CL once marked good — adverse edges always count (the guard only
    blocks positive-only flips, never blocks a genuine overrule)."""
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "OVERRULES", cy=2000, dy=1950)]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    node = db.query(CitationNode).filter(
        CitationNode.citation_text == cg._canonical_citation_key(OLDER)).one()
    node.is_good_law = True          # stale authoritative "good"
    node.is_verified = True
    node.courtlistener_id = "9997CL_TESTONLY"
    db.commit()
    res = cg.is_good_law(db, OLDER)
    assert res["is_good_law"] is False, "a real overrule downgrades even a CL-good node"
    assert res["status"] == "bad"


# ══════════════════════════════════════════════════════════════════════════
# Category 2b — Citation key idempotency (pin / foreign / malformed / statute)
# ══════════════════════════════════════════════════════════════════════════
def test_key_006_pin_cite_does_not_fork_node(db, matter, monkeypatch):
    """EXT-007: a pin cite '9000 U.S. 20, 25' must canonicalise to '9000 U.S. 20'
    and resolve to the SAME node as the bare cite — no forked node per pin."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    bare = cg.find_or_create_node(db, OLDER, node_type="case", matter_id=mid, year=1950)
    db.commit()
    assert cg._canonical_citation_key("9000 U.S. 20, 25") == cg._canonical_citation_key(OLDER)
    pinned = cg.find_or_create_node(db, "9000 U.S. 20, 25", node_type="case",
                                    matter_id=mid)
    db.commit()
    assert pinned.id == bare.id, "pin cite must not fork a second node"
    count = db.query(CitationNode).filter(
        CitationNode.citation_text == cg._canonical_citation_key(OLDER),
        CitationNode.matter_id == mid).count()
    assert count == 1


def test_key_007_foreign_cite_key_idempotent_no_crash(db, matter, monkeypatch):
    """EXT-005: a foreign neutral cite '[2099] UKSC 99' eyecite cannot parse falls
    back to the whitespace-collapse key deterministically and idempotently; storing
    it must not crash and must reuse one node across spacing variants."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    foreign = "[2099] UKSC 99"
    k1 = cg._canonical_citation_key(foreign)
    k2 = cg._canonical_citation_key("[2099]   UKSC   99")  # extra whitespace
    assert k1 == k2, "fallback key must collapse whitespace idempotently"
    n1 = cg.find_or_create_node(db, foreign, node_type="case",
                                matter_id=mid, jurisdiction="UK")
    n2 = cg.find_or_create_node(db, "[2099]   UKSC   99", node_type="case",
                                matter_id=mid, jurisdiction="UK")
    db.commit()
    assert n1.id == n2.id, "spacing variants of a foreign cite reuse one node"


def test_key_008_malformed_cite_key_stable_no_crash(db, matter, monkeypatch):
    """EXT-006: a malformed cite with no volume ('U.S. 483') has no FullCaseCitation;
    the key falls back to whitespace-collapse and stays stable / non-crashing."""
    mid, did = matter
    _patch_no_network(monkeypatch, [])
    malformed = "U.S. 483"
    assert cg._canonical_citation_key(malformed) == "U.S. 483"
    node = cg.find_or_create_node(db, malformed, node_type="case", matter_id=mid)
    db.commit()
    assert node.citation_text == "U.S. 483"


def test_ext_005b_statute_regulation_keys_pass_through(db, matter, monkeypatch):
    """EXT-003/004: statute '42 U.S.C. 9983' and regulation '29 C.F.R. 9604' are not
    FullCaseCitations -> canonical key passes them through unchanged (no mangling),
    and they are stored with the correct node_type (not 'case')."""
    mid, did = matter
    found = [
        {"raw_text": "42 U.S.C. 9983", "type": "statute", "case_name": "Act"},
        {"raw_text": "29 C.F.R. 9604", "type": "regulation", "case_name": "Reg"},
    ]
    _patch_no_network(monkeypatch, [], found_citations=found)
    _index(db, mid, did, chunks=[{"content": "cites 42 U.S.C. 9983 and 29 C.F.R. 9604"}])
    assert cg._canonical_citation_key("42 U.S.C. 9983") == "42 U.S.C. 9983"
    assert cg._canonical_citation_key("29 C.F.R. 9604") == "29 C.F.R. 9604"
    statute = db.query(CitationNode).filter(
        CitationNode.citation_text == "42 U.S.C. 9983", CitationNode.matter_id == mid).one()
    reg = db.query(CitationNode).filter(
        CitationNode.citation_text == "29 C.F.R. 9604", CitationNode.matter_id == mid).one()
    assert statute.node_type == "statute"
    assert reg.node_type == "regulation"


# ══════════════════════════════════════════════════════════════════════════
# Category 9 — De-regex robustness: extractor must NOT be brittle on UNSEEN
# inputs (the user's core fear after deleting the hand-rolled US reporter
# regex). eyecite/reporters-db is now the SOLE US extractor; these tests prove
# it covers reporters the old hardcoded list never enumerated, tolerates
# spacing variants, still covers what the old regex caught, keeps foreign
# coverage via the grammar fallback, and never crashes on garbage.
#
# All calls go through _extract_with_eyecite / extract_citations_regex /
# extract_all_citations(use_llm=False) so the suite burns ZERO LLM quota.
# ══════════════════════════════════════════════════════════════════════════

# Reporters that were NOT in the deleted hand-rolled US list (which only knew
# U.S./S.Ct./F.2d/F.3d/F.Supp./WL/LEXIS). These are real reporters covered by
# reporters-db, so eyecite must find them — proving we did NOT lose coverage by
# deleting the enumerated regex.
_UNCOMMON_US_REPORTERS = [
    "200 Vt. 123",     # Vermont Reports
    "15 Cal. 5th 1",   # California Reports, 5th series
    "50 N.E.3d 1",     # North Eastern Reporter, 3d
    "88 A.3d 5",       # Atlantic Reporter, 3d
    "12 P.3d 99",      # Pacific Reporter, 3d
    "300 So. 3d 7",    # Southern Reporter, 3d
]


@pytest.mark.parametrize("cite", _UNCOMMON_US_REPORTERS)
def test_deregex_001_uncommon_reporter_extracted_by_eyecite(cite):
    """DEREGEX-001 (CORE FEAR): a real US reporter NOT in the old hardcoded list
    is still extracted via eyecite, with confidence='library'. Proves deleting
    the enumerated US regex lost NO coverage."""
    res = ce._extract_with_eyecite(cite)
    raws = [r["raw_text"] for r in res]
    assert cite in raws, f"eyecite must extract uncommon reporter {cite!r}; got {raws}"
    found = next(r for r in res if r["raw_text"] == cite)
    assert found["confidence"] == "library"
    assert found["jurisdiction"] == "US"
    assert found["extraction_method"] == "eyecite"


@pytest.mark.parametrize("cite", _UNCOMMON_US_REPORTERS)
def test_deregex_002_uncommon_reporter_in_full_pipeline_no_llm(cite):
    """DEREGEX-002: the uncommon reporter survives the FULL extract_all_citations
    pipeline (use_llm=False) — i.e. eyecite-only path, no LLM quota — and is
    tagged library confidence (not dropped, not double-emitted).

    NOTE: real case-law prose carries a ' v. ' party string, which fires the
    _likely_has_citations gate; we include one so the test reflects realistic
    input. The gate-without-party-string edge is documented separately in
    DEREGEX-010 (xfail)."""
    text = f"In Smith v. Jones, {cite}, the court held for the plaintiff."
    res = asyncio.run(ce.extract_all_citations(text, use_llm=False))
    matches = [r for r in res if r["raw_text"] == cite]
    assert len(matches) == 1, f"{cite!r} must appear exactly once; got {res}"
    assert matches[0]["confidence"] == "library"


@pytest.mark.parametrize("cite", [
    "347 U. S. 483",   # extra-spaced US Reports
    "513 F. 3d 169",   # spaced federal reporter
    "50 N.E. 3d 1",    # spaced regional reporter
    "347 US 483",      # dotless US Reports
    "513 F3d 169",     # dotless federal reporter
])
def test_deregex_003_spacing_and_dotless_variants(cite):
    """DEREGEX-003: spacing / dotless variants of common reporters still parse via
    eyecite (it normalises them), so retrieval is not brittle to OCR/typographic
    noise on the reporters the library knows."""
    res = ce._extract_with_eyecite(cite)
    raws = [r["raw_text"] for r in res]
    assert cite in raws, f"variant {cite!r} must still extract; got {raws}"


@pytest.mark.parametrize("cite", [
    "347 U.S. 483",      # US Reports — old regex caught this
    "513 F.3d 169",      # Federal Reporter 3d — old regex caught this
    "2020 WL 7405262",   # Westlaw — old regex caught this
    "410 U.S. 113",      # US Reports
])
def test_deregex_004_old_regex_targets_still_work(cite):
    """DEREGEX-004: every citation shape the DELETED hand-rolled US regex used to
    catch is still caught — now by eyecite. Guards against regression from the
    de-regex change."""
    res = ce._extract_with_eyecite(cite)
    raws = [r["raw_text"] for r in res]
    assert cite in raws, f"old-regex target {cite!r} must still extract; got {raws}"
    assert all(r["confidence"] == "library" for r in res if r["raw_text"] == cite)


@pytest.mark.parametrize("cite,jurisdiction", [
    ("[2024] UKSC 1", "UK"),
    ("[2024] EWCA Civ 123", "UK"),
    ("[2024] HCA 5", "AU"),
    ("(1992) 175 CLR 1", "AU"),
    ("AIR 2024 SC 123", "IN"),
    ("(2024) 1 SCC 45", "IN"),
    ("[2024] SGCA 1", "SG"),
    ("2024 SCC 1", "CA"),
])
def test_deregex_005_foreign_cites_still_extract_via_grammar(cite, jurisdiction):
    """DEREGEX-005: foreign neutral cites (no library covers them) still extract via
    the documented last-resort grammar fallback, tagged confidence='grammar' and
    the correct jurisdiction — coverage retained for non-US jurisdictions."""
    res = ce.extract_citations_regex(cite)
    matches = [r for r in res if r["raw_text"] == cite]
    assert len(matches) == 1, f"{cite!r} must extract via grammar; got {res}"
    assert matches[0]["jurisdiction"] == jurisdiction
    assert matches[0]["confidence"] == "grammar"
    assert matches[0]["extraction_method"] == "regex"


def test_deregex_006_foreign_cite_in_full_pipeline_no_llm():
    """DEREGEX-006: a UK cite reaches the merged output through extract_all_citations
    (use_llm=False) via the grammar layer — never dropped just because the LLM
    primary is disabled."""
    text = "The tribunal cited [2024] UKSC 1 with approval."
    res = asyncio.run(ce.extract_all_citations(text, use_llm=False))
    raws = [r["raw_text"] for r in res]
    assert "[2024] UKSC 1" in raws, f"UK cite must survive merge; got {raws}"


@pytest.mark.parametrize("garbage", [
    "",
    "   \n\t  ",
    "\x00\xff\x01garbage\x07\x1b",
    "no legal citations whatsoever in this prose",
    "v. v. v. U.S. F.3d ECLI: [] () AIR SCC",   # citation-indicator noise, no real cite
    "}{][)(*&^%$#@!~`",
])
def test_deregex_007_garbage_and_empty_never_crash(garbage):
    """DEREGEX-007: garbage / empty / indicator-noise input must never raise across
    any layer (eyecite, grammar, full pipeline). Returns a list every time."""
    assert isinstance(ce._extract_with_eyecite(garbage), list)
    assert isinstance(ce.extract_citations_regex(garbage), list)
    res = asyncio.run(ce.extract_all_citations(garbage, use_llm=False))
    assert isinstance(res, list)


def test_deregex_008_mixed_common_and_uncommon_both_extracted_no_llm():
    """DEREGEX-008 (HEADLINE): a single document mixing a COMMON reporter the old
    regex knew (347 U.S. 483) and an UNCOMMON one it never enumerated
    (200 Vt. 123) extracts BOTH via eyecite in the full pipeline with zero LLM —
    the exact scenario the live UI check exercises."""
    text = ("In 347 U.S. 483 the Supreme Court held one thing; the Vermont court "
            "in 200 Vt. 123 reached the opposite result and overruled it.")
    res = asyncio.run(ce.extract_all_citations(text, use_llm=False))
    raws = {r["raw_text"] for r in res}
    assert "347 U.S. 483" in raws, f"common reporter missing: {raws}"
    assert "200 Vt. 123" in raws, f"uncommon reporter missing: {raws}"
    # No duplicate US entries for either cite.
    assert sum(r["raw_text"] == "347 U.S. 483" for r in res) == 1
    assert sum(r["raw_text"] == "200 Vt. 123" for r in res) == 1


@pytest.mark.xfail(
    reason="eyecite/reporters-db limitation (NOT introduced by de-regex): dotless "
           "forms of STATE reporters whose abbreviation ends in a dot ('Vt'->'Vt.', "
           "'Cal'->'Cal.') are not recognised without the trailing dot. The proper "
           "dotted forms ('200 Vt. 123', '15 Cal 5th 1' -> use 'Cal.') DO work. "
           "Documented as a real, narrow upstream gap; the old hand-rolled regex "
           "did not cover these reporters at all, so this is not a regression.",
    strict=True,
)
@pytest.mark.parametrize("cite", ["200 Vt 123", "15 Cal 5th 1"])
def test_deregex_009_dotless_state_reporter_known_gap(cite):
    """DEREGEX-009 (DOCUMENTED GAP, xfail): dotless state-reporter abbreviations are
    not parsed by eyecite. Strict xfail: if eyecite ever fixes this upstream the
    test will XPASS and flag us to remove the xfail."""
    res = ce._extract_with_eyecite(cite)
    raws = [r["raw_text"] for r in res]
    assert cite in raws, f"dotless state reporter {cite!r} not extracted; got {raws}"


def test_deregex_010_bare_uncommon_cite_reaches_eyecite():
    """DEREGEX-010 (FIXED): a bare uncommon reporter with no ' v. ' party string
    must still reach eyecite. The old gate (_likely_has_citations) was a hardcoded
    substring list that silently dropped reporters not on the list; it was replaced
    with a non-lossy digit-presence pre-filter, so the pipeline no longer vetoes
    valid citations before extraction runs."""
    text = "The relevant authority is 200 Vt. 123 on this point."
    res = asyncio.run(ce.extract_all_citations(text, use_llm=False))
    raws = [r["raw_text"] for r in res]
    assert "200 Vt. 123" in raws, f"bare uncommon cite dropped by gate; got {raws}"


def test_deregex_010b_digit_free_prose_skipped():
    """The non-lossy pre-filter still fast-paths text that structurally cannot
    contain a citation (no digits) — over-inclusion is fine, under-inclusion is not."""
    res = asyncio.run(ce.extract_all_citations("The parties agreed in good faith.", use_llm=False))
    assert res == []


# ══════════════════════════════════════════════════════════════════════════
# Category 8 — Advocate-revalidation regressions (live-test findings, 2026-06)
#   Real fresh cases (Lawrence/Janus/Citizens United) surfaced these defects.
# ══════════════════════════════════════════════════════════════════════════
def test_adv_001_disposition_triple_dropped_real_one_kept():
    """ADV-001 (P1): a procedural-disposition sentence mined as a case->case
    REVERSES triple ('The judgment of the Court of Appeals ... is reversed') must
    be dropped by _parse_relationship_items (its endpoint is not a citation), while
    a genuine OVERRULES triple is kept. Prevents the phantom 'case' node that was
    falsely flagged BAD LAW and injected into the RAG prompt."""
    junk = [{
        "citing_citation": "LAWRENCE v. TEXAS\nSupreme Court of the United States, 539 U.S. 558 (2003)",
        "cited_citation": "The judgment of the Court of Appeals for the Texas Fourteenth District",
        "treatment": "REVERSES", "evidence": "... is reversed",
    }]
    real = [_triple("Lawrence v. Texas, 539 U.S. 558 (2003)",
                    "Bowers v. Hardwick, 478 U.S. 186 (1986)", "OVERRULES")]
    assert cg._parse_relationship_items(junk) == []
    assert len(cg._parse_relationship_items(real)) == 1


def test_adv_002_looks_like_case_citation():
    """ADV-002 (P1): the authority guard is eyecite-only (library-backed, NOT a
    brittle regex / "v." substring). Accepts real reporter + statutory cites,
    rejects procedural-disposition prose."""
    assert cg._looks_like_case_citation("478 U.S. 186")
    assert cg._looks_like_case_citation("Bowers v. Hardwick, 478 U.S. 186 (1986)")
    assert cg._looks_like_case_citation("42 U.S.C. Section 1983")
    # multi-line opinion header still recognised (eyecite finds the reporter)
    assert cg._looks_like_case_citation(
        "LAWRENCE v. TEXAS\nSupreme Court of the United States, 539 U.S. 558 (2003)")
    assert not cg._looks_like_case_citation(
        "The judgment of the Court of Appeals for the Texas Fourteenth District")
    assert not cg._looks_like_case_citation("the judgment is reversed")
    assert not cg._looks_like_case_citation("")
    # Deliberate tradeoff (documented): a reporter-LESS bare name is NOT admitted
    # by a string guess — it must be resolved via CourtListener. Asserting on a
    # substring is the brittle behaviour we removed. Prose with " v " must not pass.
    assert not cg._looks_like_case_citation("Bowers v. Hardwick")
    assert not cg._looks_like_case_citation("the plaintiff won 5 v 3 on the vote")


def test_adv_003_normalize_case_name_strips_multiline_header():
    """ADV-003 (P2): multi-line opinion headers collapse to a clean single line."""
    assert cg._normalize_case_name(
        "LAWRENCE v. TEXAS\nSupreme Court of the United States, 539 U.S. 558"
    ) == "LAWRENCE v. TEXAS"
    assert cg._normalize_case_name("  Bowers   v.  Hardwick ") == "Bowers v. Hardwick"
    assert cg._normalize_case_name(None) is None
    assert cg._normalize_case_name("\n\n") is None


def test_adv_004_find_or_create_normalizes_name(db, matter):
    """ADV-004 (P2): find_or_create_node stores the normalized single-line name,
    not the leaked multi-line header."""
    mid, _ = matter
    node = cg.find_or_create_node(
        db, NEWER, node_type="case", matter_id=mid,
        name="JANUS v. AFSCME\nSupreme Court of the United States, 585 U.S. 878",
    )
    db.commit()
    assert node.name == "JANUS v. AFSCME"
    assert "\n" not in (node.name or "")


def test_adv_005_controlling_authority_upgraded_to_good(db, matter, monkeypatch):
    """ADV-005 (P3-C): a case that is the SOURCE of an OVERRULES edge and has no
    incoming adverse edge is confidently GOOD law (not merely 'no_adverse'); the
    overruled target is BAD."""
    mid, did = matter
    triples = [_triple(NEWER, OLDER, "OVERRULES", cy=2003, dy=1986,
                       citing_name="Newer", cited_name="Older")]
    _patch_no_network(monkeypatch, triples)
    _index(db, mid, did)
    g = cg.get_matter_graph(db, mid)
    by_key = {n["citation_text"]: n for n in g["nodes"]}
    assert by_key[NEWER]["good_law_status"] == "good"
    assert by_key[OLDER]["good_law_status"] == "bad"


def test_adv_006_non_citation_node_not_flagged_bad(db, matter):
    """ADV-006 (P2-A, defense-in-depth): even if a non-citation prose node somehow
    carries an incoming negative case_cites edge, it must NOT be labelled BAD LAW
    (a reversed lower-court disposition is not a citable precedent)."""
    mid, _ = matter
    src = cg.find_or_create_node(db, NEWER, node_type="case", matter_id=mid, name="Real Case")
    junk = cg.find_or_create_node(
        db, "The judgment of the Court of Appeals for the Texas Fourteenth District",
        node_type="case", matter_id=mid, name="The judgment of the Court of Appeals")
    db.flush()
    cg.create_edge(db, source_id=src.id, target_id=junk.id,
                   treatment="REVERSES", edge_kind="case_cites", matter_id=mid)
    db.commit()
    g = cg.get_matter_graph(db, mid)
    junk_node = next(n for n in g["nodes"] if "judgment of the Court" in (n["citation_text"] or ""))
    assert junk_node["good_law_status"] != "bad"
