"""Unit tests for the citation-graph RAG context injection (May 2026).

Covers `rag_engine._build_graph_context` over a stubbed in-memory graph — no
live DB, no network. Uses plain pytest + monkeypatch (mirrors the style of
test_completion_fixes.py).
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _node(citation_text, *, name=None, authority="0.0", jurisdiction="US",
          document_id=None, matter_id=None):
    """Lightweight CitationNode stand-in (only attributes the logic reads)."""
    return types.SimpleNamespace(
        citation_text=citation_text,
        name=name,
        authority_score=authority,
        jurisdiction=jurisdiction,
        document_id=document_id,
        matter_id=matter_id,
    )


# The tests drive behaviour through a fake `is_good_law` (monkeypatched onto the
# citation_graph module) and a purpose-built DB whose .query().filter().first()
# resolves CitationNodes by the normalised text recorded during the current
# iteration (since raw SQLAlchemy filter expressions can't be introspected here).


class _OrderedDB:
    """Returns nodes for citation_text lookups in a controlled sequence.

    `_build_graph_context` performs (optionally) two queries per raw citation:
    matter-scoped then global. We model this by mapping normalised text -> node
    and using a tiny expression sniffer is unnecessary: the stub query records
    nothing and `first()` consults `current_text`, set by the test harness via
    monkeypatching `_normalise_text` to also stash the value.
    """

    def __init__(self):
        self.nodes = {}
        self.current_text = None

    def add(self, normalised_text, node):
        self.nodes[normalised_text] = node

    def query(self, _model):
        outer = self

        class _Q:
            def filter(self, *args):
                return self

            def first(self):
                return outer.nodes.get(outer.current_text)

        return _Q()


@pytest.fixture
def graph_env(monkeypatch):
    """Wire a stub citation_graph (is_good_law + _canonical_citation_key) and CitationNode."""
    from backend.services import citation_graph as cg

    db = _OrderedDB()

    # _canonical_citation_key is the real code path used by _build_graph_context
    # for both dedup (step 1) and node lookup (step 2). We wrap it so it also
    # records the resolved key on the DB, letting the stub query resolve the
    # right node for each iteration — matching the actual production lookup key.
    real_canonical = cg._canonical_citation_key

    def _tracking_canonical(text):
        out = real_canonical(text)
        db.current_text = out
        return out

    monkeypatch.setattr(cg, "_canonical_citation_key", _tracking_canonical)

    return cg, db, monkeypatch


def _build(db):
    from backend.services.rag_engine import _build_graph_context
    return _build_graph_context


def test_bad_law_emitted(graph_env):
    cg, db, monkeypatch = graph_env
    node = _node("410 U.S. 113", name="Roe v. Wade", authority="0.08")
    db.add("410 U.S. 113", node)

    def fake_is_good_law(_db, text):
        return {
            "status": "bad",
            "negative_treatments": [
                {"treatment": "OVERRULES", "citing_name": "Dobbs v. Jackson"}
            ],
            "positive_count": 0,
            "cautionary_count": 0,
        }

    monkeypatch.setattr(cg, "is_good_law", fake_is_good_law)

    build = _build(db)
    out = build(db, "11111111-1111-1111-1111-111111111111",
                [{"raw_text": "410 U.S. 113"}])
    assert out is not None
    assert "BAD LAW" in out
    assert "Roe v. Wade" in out
    assert "Dobbs" in out
    assert "Authority: high" in out


def test_good_law_emitted(graph_env):
    cg, db, monkeypatch = graph_env
    node = _node("347 U.S. 483", name="Brown v. Board", authority="0.06")
    db.add("347 U.S. 483", node)

    monkeypatch.setattr(cg, "is_good_law", lambda _db, t: {
        "status": "good", "negative_treatments": [],
        "positive_count": 12, "cautionary_count": 0,
    })

    out = _build(db)(db, "m", [{"raw_text": "347 U.S. 483"}])
    assert out is not None
    assert "GOOD LAW" in out
    assert "followed by 12" in out


def test_unknown_status_omitted(graph_env):
    cg, db, monkeypatch = graph_env
    db.add("999 F.3d 1", _node("999 F.3d 1"))
    monkeypatch.setattr(cg, "is_good_law", lambda _db, t: {
        "status": "unknown", "negative_treatments": [],
        "positive_count": 0, "cautionary_count": 0,
    })
    out = _build(db)(db, "m", [{"raw_text": "999 F.3d 1"}])
    assert out is None


def test_node_not_in_graph_skipped(graph_env):
    cg, db, monkeypatch = graph_env
    # No node added → query returns None → skipped → None result.
    monkeypatch.setattr(cg, "is_good_law", lambda _db, t: {"status": "bad"})
    out = _build(db)(db, "m", [{"raw_text": "1 U.S. 1"}])
    assert out is None


def test_empty_citations_returns_none(graph_env):
    cg, db, monkeypatch = graph_env
    monkeypatch.setattr(cg, "is_good_law", lambda _db, t: {"status": "good"})
    assert _build(db)(db, "m", []) is None
    assert _build(db)(db, "m", [{"raw_text": "   "}]) is None


def test_dedup_across_union(graph_env):
    cg, db, monkeypatch = graph_env
    db.add("410 U.S. 113", _node("410 U.S. 113", name="Roe", authority="0.08"))
    calls = {"n": 0}

    def fake(_db, t):
        calls["n"] += 1
        return {"status": "bad", "negative_treatments": [],
                "positive_count": 0, "cautionary_count": 0}

    monkeypatch.setattr(cg, "is_good_law", fake)
    # Same citation in different case / whitespace → deduped to one lookup.
    out = _build(db)(db, "m", [
        {"raw_text": "410 U.S. 113"},
        {"raw_text": "410  U.S.  113"},
        {"raw_text": "410 u.s. 113"},
    ])
    assert out is not None
    assert calls["n"] == 1


def test_caps_at_max_citations(graph_env):
    cg, db, monkeypatch = graph_env
    cites = []
    for i in range(8):
        txt = f"{i} U.S. {i}"
        db.add(txt, _node(txt, name=f"Case{i}", authority=f"0.0{i}"))
        cites.append({"raw_text": txt})

    monkeypatch.setattr(cg, "is_good_law", lambda _db, t: {
        "status": "good", "negative_treatments": [],
        "positive_count": 1, "cautionary_count": 0,
    })

    out = _build(db)(db, "m", cites, max_citations=3)
    # Note: the stub DB resolves every lookup to the *last* normalised text, so
    # this primarily exercises the cap path without erroring.
    assert out is not None
    assert out.count("•") <= 3


def test_prompt_addendum_present_in_both_prompts():
    from backend.services import rag_engine as re_mod
    for prompt in (re_mod.LEGAL_SYSTEM_PROMPT, re_mod.LEGAL_RESEARCH_SYSTEM_PROMPT):
        assert "CITATION VALIDATION:" in prompt
        assert "[BAD LAW]" in prompt


def test_dotless_reporter_dedup_canonical(graph_env):
    """Dotless '410 US 113' and dotted '410 U.S. 113' must resolve to the same
    canonical key and produce exactly one BAD LAW bullet — not two separate lines.

    This exercises the _canonical_citation_key dedup path in _build_graph_context
    (step 1) with a realistic OCR/typographic variant. The stub DB is keyed on the
    canonical dotted form; the dotless input must normalise to the same key so the
    node resolves and the warning is emitted exactly once."""
    cg, db, monkeypatch = graph_env
    node = _node("410 U.S. 113", name="Roe v. Wade", authority="0.08")
    # Register under the canonical (dotted) key that _canonical_citation_key produces.
    db.add("410 U.S. 113", node)

    monkeypatch.setattr(cg, "is_good_law", lambda _db, t: {
        "status": "bad",
        "negative_treatments": [{"treatment": "OVERRULES", "citing_name": "Dobbs v. Jackson"}],
        "positive_count": 0,
        "cautionary_count": 0,
    })

    build = _build(db)
    out = build(
        db,
        "11111111-1111-1111-1111-111111111111",
        [
            {"raw_text": "410 US 113"},    # dotless variant
            {"raw_text": "410 U.S. 113"},  # standard dotted form
        ],
    )
    assert out is not None, "dotless cite must resolve to the canonical bad-law node"
    assert "BAD LAW" in out
    # Dedup: two spelling variants -> exactly one bullet line.
    bullets = [ln for ln in out.splitlines() if ln.lstrip().startswith("•")]
    assert len(bullets) == 1, f"dotless+dotted variants must dedup to one line: {bullets!r}"
