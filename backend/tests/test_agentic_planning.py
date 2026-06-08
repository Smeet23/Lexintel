"""Unit tests for the agentic RAG P0 additions (plan_node, sufficiency signals,
strategy selection, clarify schema). Pure logic only — no live DB / LLM / network.

Style mirrors tests/test_completion_fixes.py: plain pytest + monkeypatch, the
single network-touching helper (_fast_llm_json) is monkeypatched.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.services import agentic_rag as ar


def _chunk(score=0.8, content="contract breach damages remedy clause", **extra):
    base = {
        "chunk_id": f"c-{score}-{content[:6]}",
        "score": score,
        "content": content,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# plan_node
# ---------------------------------------------------------------------------

class TestPlanNode:
    def test_single_subquestion_is_deterministic_no_llm(self, monkeypatch):
        """<=1 sub-question must NOT call the LLM and yields a single-hop plan."""
        def _boom(*a, **k):
            raise AssertionError("plan_node called the LLM for a single-hop query")

        monkeypatch.setattr(ar, "_fast_llm_json", _boom)

        state = {"query": "What are the indemnity terms?", "sub_questions": ["What are the indemnity terms?"]}
        out = ar.plan_node(state)
        plan = out["research_plan"]

        assert plan["source"] == "deterministic"
        assert plan["estimated_complexity"] == "single_hop"
        assert plan["max_iterations"] == 2  # len(1) + 1
        assert len(plan["ordered_steps"]) == 1

    def test_empty_subquestions_falls_back_to_query(self, monkeypatch):
        monkeypatch.setattr(ar, "_fast_llm_json", lambda *a, **k: None)
        state = {"query": "indemnity?", "sub_questions": []}
        plan = ar.plan_node(state)["research_plan"]
        assert plan["sub_questions"] == ["indemnity?"]
        assert plan["source"] == "deterministic"

    def test_multi_hop_uses_llm_plan(self, monkeypatch):
        captured = {}

        def _fake(system, user):
            captured["called"] = True
            return {
                "estimated_complexity": "multi_hop",
                "strategy_hint": "sequential",
                "ordered_steps": [
                    {"sub_question": "Was there a breach?", "strategy": "targeted"},
                    {"sub_question": "What damages apply?", "strategy": "targeted"},
                ],
            }

        monkeypatch.setattr(ar, "_fast_llm_json", _fake)
        state = {
            "query": "breach and damages",
            "sub_questions": ["Was there a breach?", "What damages apply?"],
        }
        plan = ar.plan_node(state)["research_plan"]

        assert captured.get("called") is True
        assert plan["source"] == "llm"
        assert plan["strategy_hint"] == "sequential"
        assert plan["max_iterations"] == 3  # len(2) + 1
        assert len(plan["ordered_steps"]) == 2

    def test_multi_hop_llm_failure_falls_back_deterministic(self, monkeypatch):
        monkeypatch.setattr(ar, "_fast_llm_json", lambda *a, **k: None)
        state = {
            "query": "breach and damages",
            "sub_questions": ["Was there a breach?", "What damages apply?"],
        }
        plan = ar.plan_node(state)["research_plan"]
        assert plan["source"] == "deterministic_fallback"
        assert len(plan["ordered_steps"]) == 2

    def test_max_iterations_capped_by_setting(self, monkeypatch):
        monkeypatch.setattr(ar.settings, "agentic_max_iterations", 3, raising=False)
        monkeypatch.setattr(
            ar,
            "_fast_llm_json",
            lambda *a, **k: {"ordered_steps": [{"sub_question": "x", "strategy": "t"}]},
        )
        state = {"query": "q", "sub_questions": ["a", "b", "c", "d", "e"]}
        plan = ar.plan_node(state)["research_plan"]
        assert plan["max_iterations"] == 3  # min(5+1, 3)


# ---------------------------------------------------------------------------
# _compute_sufficiency_signals
# ---------------------------------------------------------------------------

class TestSufficiencySignals:
    def test_avg_relevance_and_chunk_count(self):
        state = {
            "query": "q",
            "sub_questions": ["q"],
            "chunks": [_chunk(0.9), _chunk(0.7), _chunk(0.5)],
        }
        sig = ar._compute_sufficiency_signals(state)
        assert sig["chunk_count"] == 3
        assert abs(sig["avg_relevance"] - 0.7) < 1e-6

    def test_empty_chunks_zero_signals(self):
        sig = ar._compute_sufficiency_signals({"query": "q", "sub_questions": ["q"], "chunks": []})
        assert sig["avg_relevance"] == 0.0
        assert sig["coverage"] == 0.0
        assert sig["chunk_count"] == 0

    def test_coverage_covered_when_tokens_overlap(self):
        state = {
            "query": "breach of contract damages",
            "sub_questions": ["what damages follow a contract breach"],
            "chunks": [_chunk(content="contract breach entitles the party to damages and remedy")],
        }
        sig = ar._compute_sufficiency_signals(state)
        assert sig["coverage"] == 1.0
        assert sig["sub_questions_uncovered"] == []

    def test_coverage_uncovered_when_no_overlap(self):
        state = {
            "query": "intellectual property",
            "sub_questions": ["how is patent infringement remedied"],
            "chunks": [_chunk(content="the weather report mentioned rain and clouds today")],
        }
        sig = ar._compute_sufficiency_signals(state)
        assert sig["coverage"] == 0.0
        assert len(sig["sub_questions_uncovered"]) == 1

    def test_jurisdiction_unknown_counts_as_match(self):
        state = {
            "query": "q",
            "sub_questions": ["q"],
            "jurisdiction": "US",
            "chunks": [_chunk()],  # no authority_metadata
        }
        sig = ar._compute_sufficiency_signals(state)
        assert sig["jurisdiction_match"] is True

    def test_jurisdiction_mismatch_detected(self):
        state = {
            "query": "q",
            "sub_questions": ["q"],
            "jurisdiction": "US",
            "chunks": [
                _chunk(authority_metadata={"jurisdiction": "UK"}),
                _chunk(authority_metadata={"jurisdiction": "UK"}),
            ],
        }
        sig = ar._compute_sufficiency_signals(state)
        assert sig["jurisdiction_match"] is False

    def test_binding_authority_from_tier(self):
        state = {
            "query": "q",
            "sub_questions": ["q"],
            "chunks": [_chunk(authority_metadata={"tier": "supreme_court"})],
        }
        sig = ar._compute_sufficiency_signals(state)
        assert sig["has_binding_authority"] is True

    def test_binding_authority_from_good_law_case(self):
        state = {
            "query": "q",
            "sub_questions": ["q"],
            "chunks": [_chunk(source_type="case_law", is_good_law=True)],
        }
        sig = ar._compute_sufficiency_signals(state)
        assert sig["has_binding_authority"] is True

    def test_no_binding_authority_when_plain_doc(self):
        sig = ar._compute_sufficiency_signals({"query": "q", "sub_questions": ["q"], "chunks": [_chunk()]})
        assert sig["has_binding_authority"] is False


# ---------------------------------------------------------------------------
# _select_strategy
# ---------------------------------------------------------------------------

class TestSelectStrategy:
    def test_high_relevance_high_coverage_is_sufficient(self):
        sig = {
            "avg_relevance": 0.85,
            "coverage": 0.9,
            "chunk_count": 8,
            "sub_questions_uncovered": [],
            "has_binding_authority": True,
        }
        assert ar._select_strategy({"rewrite_count": 0}, sig) == "sufficient"

    def test_cap_reached_forces_sufficient(self):
        sig = {
            "avg_relevance": 0.1,
            "coverage": 0.0,
            "chunk_count": 1,
            "sub_questions_uncovered": ["x"],
            "has_binding_authority": False,
        }
        state = {"rewrite_count": ar.MAX_REWRITE_ITERATIONS}
        assert ar._select_strategy(state, sig) == "sufficient"

    def test_too_few_chunks_relaxes_filters(self):
        sig = {
            "avg_relevance": 0.6,
            "coverage": 0.9,
            "chunk_count": 1,
            "sub_questions_uncovered": [],
            "has_binding_authority": True,
        }
        assert ar._select_strategy({"rewrite_count": 0, "filters_relaxed": False}, sig) == "filter_relax"

    def test_low_relevance_reformulates(self):
        sig = {
            "avg_relevance": 0.2,
            "coverage": 0.9,
            "chunk_count": 8,
            "sub_questions_uncovered": [],
            "has_binding_authority": True,
        }
        assert ar._select_strategy({"rewrite_count": 0}, sig) == "reformulate"

    def test_low_coverage_targets_uncovered(self):
        sig = {
            "avg_relevance": 0.6,
            "coverage": 0.3,
            "chunk_count": 8,
            "sub_questions_uncovered": ["uncovered sub q"],
            "has_binding_authority": True,
        }
        assert ar._select_strategy({"rewrite_count": 0}, sig) == "targeted"

    def test_missing_binding_authority_goes_external(self, monkeypatch):
        monkeypatch.setattr(ar.settings, "agentic_external_research_enabled", True, raising=False)
        sig = {
            "avg_relevance": 0.6,
            "coverage": 0.9,
            "chunk_count": 8,
            "sub_questions_uncovered": [],
            "has_binding_authority": False,
        }
        state = {"rewrite_count": 0, "external_done": False}
        assert ar._select_strategy(state, sig) == "external"

    def test_marginal_expands(self, monkeypatch):
        monkeypatch.setattr(ar.settings, "agentic_external_research_enabled", True, raising=False)
        sig = {
            "avg_relevance": 0.6,
            "coverage": 0.9,
            "chunk_count": 8,
            "sub_questions_uncovered": [],
            "has_binding_authority": True,  # binding present, so not external
        }
        assert ar._select_strategy({"rewrite_count": 0, "external_done": False}, sig) == "expand"


# ---------------------------------------------------------------------------
# clarify_node extended schema
# ---------------------------------------------------------------------------

class TestClarifyExtendedSchema:
    def test_extended_fields_passed_through(self, monkeypatch):
        monkeypatch.setattr(
            ar,
            "_fast_llm_json",
            lambda *a, **k: {
                "jurisdiction": "US",
                "entities": ["Acme"],
                "refined_query": "acme breach",
                "sub_questions": ["was there a breach", "what damages"],
                "requires_external_research": True,
                "temporal_scope": "current",
            },
        )
        out = ar.clarify_node({"query": "acme breach damages"})
        assert out["sub_questions"] == ["was there a breach", "what damages"]
        assert out["requires_external_research"] is True
        assert out["temporal_scope"] == "current"

    def test_fallback_when_llm_returns_none(self, monkeypatch):
        monkeypatch.setattr(ar, "_fast_llm_json", lambda *a, **k: None)
        out = ar.clarify_node({"query": "some query"})
        assert out["sub_questions"] == ["some query"]
        assert out["requires_external_research"] is False
        assert out["temporal_scope"] is None

    def test_empty_subquestions_defaults_to_query(self, monkeypatch):
        monkeypatch.setattr(
            ar,
            "_fast_llm_json",
            lambda *a, **k: {"sub_questions": [], "refined_query": "x"},
        )
        out = ar.clarify_node({"query": "fallback query"})
        assert out["sub_questions"] == ["fallback query"]
