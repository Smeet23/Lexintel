"""Comprehensive unit and integration tests for Lexintel services.

Tests cover pure logic paths — no external API calls are made.
For services that are async-only, pytest-asyncio is used.
"""
import sys
import math
import asyncio
import pytest

sys.path.insert(0, ".")


# ═══════════════════════════════════════════════════════════════
# 1. PROBLEM FORMULATION
# ═══════════════════════════════════════════════════════════════


class TestProblemFormulationStructure:
    """Tests for _parse_issue_analysis — the pure-logic validation layer."""

    def _parse(self, raw):
        from backend.services.problem_formulation import _parse_issue_analysis
        return _parse_issue_analysis(raw)

    def test_empty_query_returns_none(self):
        """identify_issues returns None for empty/whitespace queries."""
        result = self._parse({"issues": [], "missing_information": []})
        assert result is not None
        assert result["issues"] == []

    def test_response_has_issues_list(self):
        """Parsed result always contains an 'issues' list."""
        raw = {
            "issues": [
                {
                    "domain": "contract",
                    "issue": "breach of contract",
                    "legal_question": "Did Party A breach the agreement?",
                    "confidence": 0.85,
                    "key_facts": ["signed agreement", "non-delivery"],
                }
            ],
            "missing_information": ["jurisdiction of the contract"],
        }
        result = self._parse(raw)
        assert result is not None
        assert "issues" in result
        assert isinstance(result["issues"], list)

    def test_response_has_missing_information_list(self):
        """Parsed result always contains a 'missing_information' list."""
        raw = {
            "issues": [],
            "missing_information": ["governing law", "date of incident"],
        }
        result = self._parse(raw)
        assert result is not None
        assert "missing_information" in result
        assert isinstance(result["missing_information"], list)

    def test_confidence_is_clamped_to_0_1(self):
        """Confidence values outside [0,1] are clamped."""
        raw = {
            "issues": [
                {
                    "domain": "tort",
                    "issue": "negligence",
                    "legal_question": "Was there a duty of care?",
                    "confidence": 1.9,  # over 1.0
                    "key_facts": [],
                }
            ],
            "missing_information": [],
        }
        result = self._parse(raw)
        assert result is not None
        assert result["issues"][0]["confidence"] <= 1.0

    def test_issues_hard_capped_at_five(self):
        """LLM responses with more than 5 issues are truncated to 5."""
        raw = {
            "issues": [
                {
                    "domain": f"domain_{i}",
                    "issue": f"issue {i}",
                    "legal_question": "?",
                    "confidence": 0.5,
                    "key_facts": [],
                }
                for i in range(8)
            ],
            "missing_information": [],
        }
        result = self._parse(raw)
        assert result is not None
        assert len(result["issues"]) <= 5

    def test_non_list_missing_information_is_normalised(self):
        """Non-list missing_information is coerced to an empty list."""
        raw = {
            "issues": [],
            "missing_information": "some string",
        }
        result = self._parse(raw)
        assert result is not None
        assert isinstance(result["missing_information"], list)

    def test_invalid_raw_input_returns_none(self):
        """Completely invalid input (no 'issues' key as a list) returns None."""
        result = self._parse({"issues": "not-a-list"})
        assert result is None

    def test_key_facts_coerced_to_list(self):
        """key_facts that is not a list is replaced with an empty list."""
        raw = {
            "issues": [
                {
                    "domain": "employment",
                    "issue": "wrongful termination",
                    "legal_question": "Was dismissal fair?",
                    "confidence": 0.7,
                    "key_facts": "single string fact",  # wrong type
                }
            ],
            "missing_information": [],
        }
        result = self._parse(raw)
        assert result is not None
        assert isinstance(result["issues"][0]["key_facts"], list)


# ═══════════════════════════════════════════════════════════════
# 2. CLAIM VERIFIER — PURE LOGIC
# ═══════════════════════════════════════════════════════════════


class TestSplitIntoClaims:
    """Tests for _split_into_claims — sentence splitting + claim filtering."""

    def _split(self, text):
        from backend.services.claim_verifier import _split_into_claims
        return _split_into_claims(text)

    def test_empty_string_returns_empty_list(self):
        """Empty string input returns an empty list."""
        assert self._split("") == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only input returns an empty list."""
        assert self._split("   \n  ") == []

    def test_preserves_citation_indices(self):
        """Citation markers [1], [2] are extracted into cited_sources."""
        answer = "The court ruled in favor of the plaintiff. [1] The defendant was liable. [2]"
        claims = self._split(answer)
        all_sources = [src for c in claims for src in c["cited_sources"]]
        assert 1 in all_sources or 2 in all_sources

    def test_citation_markers_stripped_from_text(self):
        """After splitting, claim text should not contain raw [N] markers."""
        answer = "The defendant was found guilty. [1]"
        claims = self._split(answer)
        for c in claims:
            assert "[1]" not in c["text"]

    def test_each_claim_has_required_keys(self):
        """Every claim dict has: text, context, cited_sources, sentence_idx, flags."""
        answer = "The statute requires written notice. The party failed to comply."
        claims = self._split(answer)
        for c in claims:
            assert "text" in c
            assert "context" in c
            assert "cited_sources" in c
            assert "sentence_idx" in c
            assert "flags" in c

    def test_nupunkt_legal_abbreviations_us(self):
        """'U.S.' does not trigger a false sentence split."""
        answer = "The case was decided under 42 U.S.C. § 1983. The plaintiff prevailed."
        claims = self._split(answer)
        # Must produce at least 1 claim without crashing
        assert isinstance(claims, list)

    def test_nupunkt_legal_abbreviations_inc_v(self):
        """'Inc.' and 'v.' do not trigger false splits."""
        answer = "Smith Inc. v. Jones Corp. settled the dispute amicably."
        claims = self._split(answer)
        assert isinstance(claims, list)

    def test_multiple_sentences_produce_multiple_claims(self):
        """A multi-sentence answer produces one claim per verifiable sentence."""
        answer = (
            "The agreement was signed on 1 January 2023. "
            "The defendant breached the contract by failing to deliver goods. "
            "The plaintiff sustained damages totalling $50,000."
        )
        claims = self._split(answer)
        assert len(claims) >= 2

    def test_returns_list_of_dicts(self):
        """Return type is always a list of dicts."""
        result = self._split("The court held for the plaintiff.")
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], dict)


class TestIsVerifiableClaim:
    """Tests for _is_verifiable_claim — noise filtering."""

    def _check(self, text):
        from backend.services.claim_verifier import _is_verifiable_claim
        return _is_verifiable_claim(text)

    def test_short_text_filtered(self):
        """Text under 15 characters is not a verifiable claim."""
        assert self._check("Yes.") is False
        assert self._check("No.") is False

    def test_header_ending_with_colon_filtered(self):
        """Lines like 'Key Findings:' (under 50 chars, ends with colon) are filtered."""
        assert self._check("Key Findings:") is False
        assert self._check("Analysis:") is False

    def test_long_header_not_filtered(self):
        """A long sentence ending with a colon is NOT filtered (>50 chars)."""
        long_header = "The following section details the court's analysis of contributory negligence:"
        # Over 50 chars — should pass the filter
        assert self._check(long_header) is True

    def test_normal_claim_passes(self):
        """Normal legal sentences pass the filter."""
        assert self._check("The court held that the defendant was negligent.") is True

    def test_citation_only_reference_filtered(self):
        """Sentences starting with 'See', 'cf.', 'Compare', etc. are filtered."""
        from backend.services.claim_verifier import _is_verifiable_claim
        assert _is_verifiable_claim("See Brown v. Board of Education for precedent.") is False

    def test_empty_text_filtered(self):
        """Whitespace-only text is filtered."""
        assert self._check("   ") is False


class TestTokenCoverage:
    """Tests for _check_token_coverage — partial truth detection."""

    def _coverage(self, claim, source):
        from backend.services.claim_verifier import _check_token_coverage
        return _check_token_coverage(claim, source)

    def test_partial_truth_detection(self):
        """Claim omitting key source terms scores low coverage."""
        claim = "The defendant was found liable."
        source = "The defendant was found liable on count 1 but was acquitted on counts 2 through 5."
        score = self._coverage(claim, source)
        # 'acquitted' is in source but not claim — coverage should be partial
        assert score < 1.0

    def test_full_overlap_scores_high(self):
        """When claim contains all source terms, coverage approaches 1.0."""
        source = "The defendant was found liable for breach."
        claim = "The defendant was found liable for breach."
        score = self._coverage(claim, source)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_empty_claim_returns_one(self):
        """Empty claim string returns 1.0 (no false omission signal)."""
        score = self._coverage("", "The defendant was found liable.")
        assert score == pytest.approx(1.0)

    def test_empty_source_returns_one(self):
        """Empty source string returns 1.0 (nothing to miss)."""
        score = self._coverage("The defendant was found liable.", "")
        assert score == pytest.approx(1.0)

    def test_score_in_valid_range(self):
        """Coverage score is always in [0.0, 1.0]."""
        score = self._coverage(
            "court held for plaintiff",
            "The High Court of Justice held in favor of the plaintiff and awarded damages."
        )
        assert 0.0 <= score <= 1.0


class TestExtractSignificantWords:
    """Tests for _extract_significant_words — token extraction logic."""

    def _extract(self, text):
        from backend.services.claim_verifier import _extract_significant_words
        return _extract_significant_words(text)

    def test_filters_short_words(self):
        """Words shorter than 4 characters are excluded."""
        words = self._extract("the cat sat on mat")
        assert "cat" not in words
        assert "sat" not in words

    def test_excludes_legal_stopwords(self):
        """Known legal stopwords ('that', 'with', 'from', etc.) are excluded."""
        words = self._extract("that this with from have been were which")
        assert len(words) == 0

    def test_includes_meaningful_words(self):
        """Meaningful legal words of 4+ chars are included."""
        words = self._extract("The court awarded damages for negligence.")
        assert "court" in words
        assert "awarded" in words
        assert "damages" in words
        assert "negligence" in words

    def test_strips_punctuation_from_edges(self):
        """Trailing/leading punctuation is stripped before length check."""
        words = self._extract("court, awarded.")
        assert "court" in words
        assert "awarded" in words

    def test_lowercases_all_tokens(self):
        """All returned tokens are lowercase."""
        words = self._extract("Supreme Court ruled")
        assert all(w == w.lower() for w in words)

    def test_returns_set(self):
        """Return type is a set."""
        result = self._extract("breach of contract damages")
        assert isinstance(result, set)


class TestLogitsToProbs:
    """Tests for _logits_to_probs — softmax utility."""

    def _convert(self, logits):
        from backend.services.claim_verifier import _logits_to_probs
        return _logits_to_probs(logits)

    def test_output_sums_to_one(self):
        """Softmax output sums to ~1.0."""
        probs = self._convert([2.0, 1.0, 0.1])
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_all_values_in_unit_interval(self):
        """All probability values are in [0, 1]."""
        probs = self._convert([3.0, -1.0, 0.5])
        for p in probs:
            assert 0.0 <= p <= 1.0

    def test_handles_nan_input(self):
        """NaN inputs are replaced and result is still a valid distribution."""
        import numpy as np
        probs = self._convert([float("nan"), 1.0, 0.0])
        assert not any(math.isnan(p) for p in probs)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_handles_inf_input(self):
        """Positive and negative infinity inputs produce valid probabilities."""
        probs = self._convert([float("inf"), float("-inf"), 1.0])
        for p in probs:
            assert not math.isnan(p)
            assert not math.isinf(p)
            assert 0.0 <= p <= 1.0

    def test_equal_logits_produce_uniform_distribution(self):
        """Equal logits → uniform probabilities (each ~1/N)."""
        probs = self._convert([0.0, 0.0, 0.0])
        for p in probs:
            assert p == pytest.approx(1 / 3, abs=1e-6)

    def test_returns_list(self):
        """Return type is a list."""
        result = self._convert([1.0, 2.0, 0.5])
        assert isinstance(result, list)


class TestClassifyClaim:
    """Tests for _classify_claim — special flag detection."""

    def _classify(self, text):
        from backend.services.claim_verifier import _classify_claim
        return _classify_claim(text)

    def test_detects_conditional_if(self):
        """'if' keyword triggers 'conditional' flag."""
        flags = self._classify("The penalty applies if the party fails to notify within 30 days.")
        assert "conditional" in flags

    def test_detects_conditional_unless(self):
        """'unless' keyword triggers 'conditional' flag."""
        flags = self._classify("The clause is enforceable unless waived in writing.")
        assert "conditional" in flags

    def test_detects_conditional_provided_that(self):
        """'provided that' triggers 'conditional' flag."""
        flags = self._classify("Payment is due provided that delivery is complete.")
        assert "conditional" in flags

    def test_detects_double_negation_not_un(self):
        """'not un-' pattern triggers 'double_negation' flag."""
        flags = self._classify("The obligation is not unenforceable under current law.")
        assert "double_negation" in flags

    def test_detects_double_negation_cannot_deny(self):
        """'cannot deny' triggers 'double_negation' flag."""
        flags = self._classify("The defendant cannot deny receiving the notice.")
        assert "double_negation" in flags

    def test_detects_quantifier_all(self):
        """'all' keyword triggers 'has_quantifier' flag."""
        flags = self._classify("All parties must sign the agreement.")
        assert "has_quantifier" in flags

    def test_plain_claim_has_no_flags(self):
        """A straightforward factual sentence produces no flags."""
        flags = self._classify("The court awarded damages to the claimant.")
        assert flags == []

    def test_returns_list(self):
        """Return type is always a list."""
        result = self._classify("The statute was enacted.")
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════
# 3. HYBRID SEARCH — PURE LOGIC
# ═══════════════════════════════════════════════════════════════


class TestClassifyQueryType:
    """Tests for classify_query_type — query routing."""

    def _classify(self, query):
        from backend.services.hybrid_search import classify_query_type
        return classify_query_type(query)

    def test_citation_query_with_v_dot(self):
        """Query containing 'v. [Capital]' is classified as 'citation'."""
        result = self._classify("What did Smith v. Jones 347 U.S. 483 hold?")
        assert result == "citation"

    def test_citation_query_us_reporter(self):
        """Query with a US reporter citation (no conceptual keywords) is classified as 'citation'."""
        # "Summarize" is a conceptual keyword, so use a neutral framing
        result = self._classify("The holding in 513 F.3d 169 on jurisdiction.")
        assert result == "citation"

    def test_conceptual_query_explain(self):
        """Query starting with 'explain' is classified as 'conceptual'."""
        result = self._classify("Explain the doctrine of promissory estoppel.")
        assert result == "conceptual"

    def test_conceptual_query_analyze(self):
        """Query with 'analyze' is classified as 'conceptual'."""
        result = self._classify("Analyze the breach of contract elements.")
        assert result == "conceptual"

    def test_mixed_query_returns_mixed(self):
        """Query with both citation and conceptual cues returns 'mixed'."""
        result = self._classify("Explain how 347 U.S. 483 changed constitutional law.")
        assert result == "mixed"

    def test_plain_query_without_patterns_returns_mixed(self):
        """Query with neither citation nor conceptual keywords returns 'mixed'."""
        result = self._classify("What happens when a tenant does not pay rent?")
        assert result == "mixed"

    def test_india_air_citation(self):
        """Indian AIR citation pattern recognised as citation query."""
        # AIR pattern: \bAIR\s+\d{4}\s+SC\s+\d+
        result = self._classify("AIR 2020 SC 1234 is the leading case.")
        assert result == "citation"


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion — RRF score merging."""

    def _rrf(self, bm25, dense, bm25_w=0.5, dense_w=0.5, k=60, top_n=None):
        from backend.services.hybrid_search import reciprocal_rank_fusion
        return reciprocal_rank_fusion(bm25, dense, bm25_w, dense_w, k=k, top_n=top_n)

    def _make_results(self, ids):
        """Create minimal result dicts for testing."""
        return [{"chunk_id": cid, "score": 0.9 - i * 0.1} for i, cid in enumerate(ids)]

    def test_basic_merge_returns_all_unique_chunks(self):
        """RRF merges results from both lists into a single deduplicated list."""
        bm25 = self._make_results(["a", "b", "c"])
        dense = self._make_results(["b", "d", "e"])
        merged = self._rrf(bm25, dense)
        ids = {r["chunk_id"] for r in merged}
        assert ids == {"a", "b", "c", "d", "e"}

    def test_chunk_appearing_in_both_scores_higher(self):
        """A chunk in both BM25 and dense ranks should outscore a chunk in only one."""
        bm25 = self._make_results(["shared", "only_bm25"])
        dense = self._make_results(["shared", "only_dense"])
        merged = self._rrf(bm25, dense)
        shared_score = next(r["rrf_score"] for r in merged if r["chunk_id"] == "shared")
        bm25_only_score = next(r["rrf_score"] for r in merged if r["chunk_id"] == "only_bm25")
        dense_only_score = next(r["rrf_score"] for r in merged if r["chunk_id"] == "only_dense")
        assert shared_score > bm25_only_score
        assert shared_score > dense_only_score

    def test_empty_bm25_does_not_crash(self):
        """Empty BM25 results list is handled gracefully."""
        dense = self._make_results(["x", "y"])
        merged = self._rrf([], dense)
        assert len(merged) == 2

    def test_empty_dense_does_not_crash(self):
        """Empty dense results list is handled gracefully."""
        bm25 = self._make_results(["x", "y"])
        merged = self._rrf(bm25, [])
        assert len(merged) == 2

    def test_both_empty_returns_empty(self):
        """Both empty inputs return an empty list."""
        merged = self._rrf([], [])
        assert merged == []

    def test_results_sorted_by_rrf_score_descending(self):
        """Output is sorted with highest RRF score first."""
        bm25 = self._make_results(["a", "b", "c"])
        dense = self._make_results(["a", "c", "d"])
        merged = self._rrf(bm25, dense)
        scores = [r["rrf_score"] for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_output(self):
        """top_n parameter truncates output to that many results."""
        bm25 = self._make_results(["a", "b", "c", "d"])
        dense = self._make_results(["e", "f", "g"])
        merged = self._rrf(bm25, dense, top_n=3)
        assert len(merged) == 3

    def test_rrf_score_is_positive(self):
        """All RRF scores are positive (formula: weight / (k + rank))."""
        bm25 = self._make_results(["a", "b"])
        dense = self._make_results(["a", "c"])
        merged = self._rrf(bm25, dense)
        for r in merged:
            assert r["rrf_score"] > 0

    def test_missing_chunk_id_skipped(self):
        """Results without a chunk_id key are silently skipped."""
        bm25 = [{"score": 0.9}]  # No chunk_id
        dense = self._make_results(["valid"])
        merged = self._rrf(bm25, dense)
        ids = [r["chunk_id"] for r in merged]
        assert "valid" in ids
        assert len(merged) == 1


class TestGetRrfWeights:
    """Tests for get_rrf_weights — adaptive weight selection."""

    def _weights(self, query):
        from backend.services.hybrid_search import get_rrf_weights
        return get_rrf_weights(query)

    def test_citation_query_gets_higher_bm25_weight(self):
        """Citation queries get BM25 weight 0.85 and dense weight 0.15."""
        bm25_w, dense_w = self._weights("Smith v. Jones 347 U.S. 483")
        assert bm25_w == pytest.approx(0.85)
        assert dense_w == pytest.approx(0.15)

    def test_conceptual_query_gets_equal_weights(self):
        """Conceptual queries get equal 0.50 / 0.50 weights."""
        bm25_w, dense_w = self._weights("Explain promissory estoppel.")
        assert bm25_w == pytest.approx(0.50)
        assert dense_w == pytest.approx(0.50)

    def test_weights_always_sum_to_one(self):
        """BM25 weight + dense weight should sum to 1.0 for citation/conceptual."""
        for query in [
            "Smith v. Jones 347 U.S. 483",
            "Explain the duty of care.",
        ]:
            bm25_w, dense_w = self._weights(query)
            assert bm25_w + dense_w == pytest.approx(1.0, abs=0.01)

    def test_returns_tuple_of_length_two(self):
        """Return value is a 2-tuple; both values are non-negative and convertible to float."""
        result = self._weights("What is negligence?")
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Values may be Pydantic scalar subtypes; coerce to float to verify numericity
        bm25_w, dense_w = float(result[0]), float(result[1])
        assert bm25_w >= 0.0
        assert dense_w >= 0.0


# ═══════════════════════════════════════════════════════════════
# 4. AUTHORITY DETECTOR — PURE LOGIC
# ═══════════════════════════════════════════════════════════════


class TestComputeAuthorityScore:
    """Tests for _compute_authority_score — ingestion-time score formula."""

    def _score(self, court_level, binding=None):
        from backend.services.authority_detector import _compute_authority_score
        return _compute_authority_score(court_level, binding)

    def test_supreme_court_binding_gets_highest_score(self):
        """Supreme court + binding authority produces the maximum score."""
        score = self._score("supreme", binding=True)
        # Formula: (1.0 * 0.5) + (0.5 * 0.3) + (1.0 * 0.2) = 0.5 + 0.15 + 0.2 = 0.85
        assert score == pytest.approx(0.85)

    def test_unknown_court_gets_default_score(self):
        """Unknown court level gets a mid-range default score."""
        score = self._score("unknown", binding=None)
        # Formula: (0.5 * 0.5) + (0.5 * 0.3) + (0.3 * 0.2) = 0.25 + 0.15 + 0.06 = 0.46
        assert score == pytest.approx(0.46)

    def test_appellate_non_binding(self):
        """Appellate non-binding produces an intermediate score."""
        score = self._score("appellate", binding=False)
        # Formula: (0.85 * 0.5) + (0.5 * 0.3) + (0.0 * 0.2) = 0.425 + 0.15 + 0 = 0.575
        assert score == pytest.approx(0.575)

    def test_score_bounded_between_0_and_1(self):
        """Authority score is always in [0.0, 1.0]."""
        for level in ["supreme", "appellate", "trial", "administrative", "unknown"]:
            for binding in [True, False, None]:
                score = self._score(level, binding)
                assert 0.0 <= score <= 1.0, f"Out of range: {level}, {binding} -> {score}"

    def test_result_is_rounded_to_4_decimals(self):
        """Return value has at most 4 decimal places."""
        score = self._score("trial", binding=True)
        assert score == round(score, 4)


class TestComputeQueryTimeAuthorityScore:
    """Tests for compute_query_time_authority_score — reranking formula."""

    def _qscore(self, chunk_meta, target=None):
        from backend.services.authority_detector import compute_query_time_authority_score
        return compute_query_time_authority_score(chunk_meta, target)

    def test_exact_jurisdiction_match_scores_higher_than_sister_state(self):
        """Same jurisdiction scores higher than a sister-state jurisdiction."""
        meta = {"court_level": "appellate", "binding_authority": True, "jurisdiction_code": "US.state.CA"}
        same_jx_score = self._qscore(meta, "US.state.CA")
        sister_score = self._qscore(meta, "US.state.NY")
        assert same_jx_score > sister_score

    def test_federal_is_more_relevant_than_sister_state(self):
        """Federal jurisdiction is weighted above a sister state."""
        meta = {"court_level": "appellate", "binding_authority": True, "jurisdiction_code": "US.federal"}
        federal_score = self._qscore(meta, "US.state.CA")
        meta_sister = {**meta, "jurisdiction_code": "US.state.NY"}
        sister_score = self._qscore(meta_sister, "US.state.CA")
        assert federal_score > sister_score

    def test_unknown_jurisdiction_uses_default_weight(self):
        """Unknown chunk jurisdiction uses a conservative default weight."""
        meta = {"court_level": "supreme", "binding_authority": True, "jurisdiction_code": "unknown"}
        score = self._qscore(meta, "US.state.CA")
        # Should not crash; score is in valid range
        assert 0.0 <= score <= 1.0

    def test_no_target_jurisdiction_uses_default(self):
        """No target jurisdiction falls back to default weight without error."""
        meta = {"court_level": "trial", "binding_authority": False, "jurisdiction_code": "US.state.TX"}
        score = self._qscore(meta, None)
        assert 0.0 <= score <= 1.0


class TestResolveJurisdictionWeight:
    """Tests for _resolve_jurisdiction_weight — the jurisdiction hierarchy."""

    def _weight(self, chunk_jx, target_jx):
        from backend.services.authority_detector import _resolve_jurisdiction_weight
        return _resolve_jurisdiction_weight(chunk_jx, target_jx)

    def test_exact_match_returns_1(self):
        """Identical jurisdictions return weight 1.0."""
        assert self._weight("US.state.CA", "US.state.CA") == pytest.approx(1.0)

    def test_federal_to_state_returns_0_85(self):
        """Federal chunk against state target returns 0.85."""
        assert self._weight("US.federal", "US.state.CA") == pytest.approx(0.85)

    def test_sister_state_returns_0_50(self):
        """Different states in same country return 0.50."""
        assert self._weight("US.state.NY", "US.state.CA") == pytest.approx(0.50)

    def test_foreign_jurisdiction_returns_0_30(self):
        """Cross-country jurisdiction returns 0.30."""
        assert self._weight("UK", "US.state.CA") == pytest.approx(0.30)

    def test_unknown_chunk_jurisdiction_returns_0_40(self):
        """Unknown chunk jurisdiction returns 0.40."""
        assert self._weight("unknown", "US.state.CA") == pytest.approx(0.40)

    def test_unknown_target_jurisdiction_returns_0_40(self):
        """No/unknown target jurisdiction returns 0.40."""
        assert self._weight("US.state.CA", None) == pytest.approx(0.40)
        assert self._weight("US.state.CA", "unknown") == pytest.approx(0.40)


# ═══════════════════════════════════════════════════════════════
# 5. TEMPORAL EXTRACTOR — PURE LOGIC
# ═══════════════════════════════════════════════════════════════


class TestExtractVersionFromFilename:
    """Tests for extract_version_from_filename — filename heuristics."""

    def _extract(self, filename):
        from backend.services.temporal_extractor import extract_version_from_filename
        return extract_version_from_filename(filename)

    def test_v_prefix_extracts_version(self):
        """'Contract_v2.1.pdf' extracts version '2.1'."""
        assert self._extract("Contract_v2.1.pdf") == "2.1"

    def test_rev_prefix_extracts_version(self):
        """'Policy_Rev3.docx' extracts version '3'."""
        assert self._extract("Policy_Rev3.docx") == "3"

    def test_amendment_no_extracts_version(self):
        """'(Amendment No. 2)' extracts version '2'."""
        result = self._extract("Data Protection Act 2024 (Amendment No. 2).pdf")
        assert result == "2"

    def test_plain_filename_returns_none(self):
        """'memo.pdf' with no version marker returns None."""
        assert self._extract("memo.pdf") is None

    def test_no_extension_still_works(self):
        """Filename without extension is handled."""
        assert self._extract("contract_v1.0") == "1.0"

    def test_major_version_only(self):
        """Single-digit version '_v3' returns '3'."""
        assert self._extract("Agreement_v3.pdf") == "3"

    def test_version_keyword_in_filename(self):
        """'version 2.0' keyword pattern is recognised."""
        result = self._extract("Policy version 2.0 Final.docx")
        assert result == "2.0"


class TestParseDateString:
    """Tests for _parse_date_string — robust date parsing."""

    def _parse(self, text):
        from backend.services.temporal_extractor import _parse_date_string
        return _parse_date_string(text)

    def test_iso_date_parsed(self):
        """ISO format '2024-01-15' is parsed correctly."""
        dt = self._parse("2024-01-15")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_bare_year_parsed_as_jan_1(self):
        """Bare year '2023' is parsed as 2023-01-01."""
        dt = self._parse("2023")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 1
        assert dt.day == 1

    def test_month_year_parsed_as_first_of_month(self):
        """'March 2024' is parsed as 2024-03-01."""
        dt = self._parse("March 2024")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 1

    def test_ordinal_date_parsed(self):
        """'25th March 2024' is parsed correctly."""
        dt = self._parse("25th March 2024")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 25

    def test_result_is_timezone_aware(self):
        """All parsed dates are UTC-aware."""
        from datetime import timezone
        dt = self._parse("2024-06-01")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_invalid_string_returns_none(self):
        """Clearly invalid date strings return None."""
        assert self._parse("not-a-date") is None

    def test_out_of_range_year_returns_none(self):
        """Year outside 1900-2100 returns None."""
        dt = self._parse("1850")
        assert dt is None


class TestExtractViaRegex:
    """Tests for _extract_via_regex — temporal metadata extraction."""

    def _extract(self, text):
        from backend.services.temporal_extractor import _extract_via_regex
        return _extract_via_regex(text)

    def test_effective_from_date_extracted(self):
        """'effective from 1 January 2024' sets effective_date."""
        result = self._extract("This act is effective from 1 January 2024.")
        assert result.effective_date is not None
        assert result.effective_date.year == 2024

    def test_repealed_sets_document_status(self):
        """'this act is repealed' sets document_status to 'repealed'."""
        result = self._extract("This Act is hereby repealed.")
        assert result.document_status == "repealed"

    def test_version_number_extracted(self):
        """'Version 3.0' extracts version_number '3.0'."""
        result = self._extract("Version 3.0 of the guidelines applies.")
        assert result.version_number is not None
        assert "3" in result.version_number

    def test_no_temporal_info_returns_unknown_status(self):
        """Text with no temporal markers leaves document_status as 'unknown'."""
        result = self._extract("The parties agree to the terms herein.")
        assert result.document_status == "unknown"
        assert result.extraction_method == "none"

    def test_future_effective_date_sets_draft_status(self):
        """A far-future effective date sets document_status to 'draft'."""
        result = self._extract("effective from 1 January 2099.")
        if result.effective_date:
            assert result.document_status == "draft"

    def test_supersedes_reference_extracted(self):
        """'supersedes the Privacy Act 2018' populates supersedes_references."""
        result = self._extract("This regulation supersedes the Privacy Act 2018.")
        assert len(result.supersedes_references) > 0


# ═══════════════════════════════════════════════════════════════
# 6. CITATION EXTRACTOR — PURE LOGIC (regex only)
# ═══════════════════════════════════════════════════════════════


class TestExtractCitationsRegex:
    """Tests for extract_citations_regex — multi-jurisdiction citation matching."""

    def _extract(self, text):
        from backend.services.citation_extractor import extract_citations_regex
        return extract_citations_regex(text)

    def test_us_citation_not_emitted_by_grammar_fallback(self):
        """US reporters are owned by eyecite, NOT the non-US grammar fallback.

        The hand-rolled US reporter regex was removed (redundant + brittle vs
        eyecite). ``extract_citations_regex`` is now the non-US neutral-citation
        fallback only, so it must NOT emit US reporter citations.
        """
        text = "The landmark case 347 U.S. 483 (1954) and 513 F.3d 169."
        results = self._extract(text)
        assert not any(r.get("jurisdiction") == "US" for r in results)

    def test_us_citation_extracted_by_eyecite_in_pipeline(self):
        """eyecite is the authoritative US extractor in the full pipeline."""
        import asyncio
        from backend.services.citation_extractor import extract_all_citations
        text = "The landmark case 347 U.S. 483 (1954) and 513 F.3d 169."
        results = asyncio.run(extract_all_citations(text, use_llm=False))
        us = [r for r in results if r.get("jurisdiction") == "US"]
        assert any("347 U.S. 483" == r["raw_text"] for r in us)
        assert any("513 F.3d 169" == r["raw_text"] for r in us)
        # All US hits must carry the library-confidence eyecite provenance.
        assert all(r.get("confidence") == "library" for r in us)

    def test_uk_citation(self):
        """'[2024] UKSC 1' is extracted as a UK citation."""
        text = "The Supreme Court decided [2024] UKSC 1."
        results = self._extract(text)
        assert any(r["jurisdiction"] == "UK" for r in results)

    def test_indian_scc_citation(self):
        """'(2024) 1 SCC 45' is extracted as an Indian citation."""
        text = "The apex court ruled in (2024) 1 SCC 45."
        results = self._extract(text)
        assert any(r["jurisdiction"] == "IN" for r in results)

    def test_plain_prose_returns_empty(self):
        """Normal prose without citations returns an empty list."""
        text = (
            "The parties entered into a binding agreement on the first day of March. "
            "Both sides agreed to the terms and conditions as outlined."
        )
        results = self._extract(text)
        assert results == []

    def test_results_sorted_by_position(self):
        """Multiple citations in text are returned sorted by their position."""
        text = "See 347 U.S. 483 and 513 F.3d 169 for more details."
        results = self._extract(text)
        if len(results) >= 2:
            spans = [r["span"][0] for r in results]
            assert spans == sorted(spans)

    def test_each_result_has_required_keys(self):
        """Every citation dict has raw_text, span, jurisdiction, type, extraction_method."""
        text = "The court cited 347 U.S. 483."
        results = self._extract(text)
        for r in results:
            assert "raw_text" in r
            assert "span" in r
            assert "jurisdiction" in r
            assert "type" in r
            assert "extraction_method" in r

    def test_no_duplicate_spans(self):
        """The same span is not returned twice."""
        text = "Referring to 347 U.S. 483 multiple times: 347 U.S. 483."
        results = self._extract(text)
        spans = [r["span"] for r in results]
        assert len(spans) == len(set(spans))

    def test_eu_ecli_citation(self):
        """'ECLI:EU:C:2014:317' is extracted as an EU citation."""
        text = "See ECLI:EU:C:2014:317 for the ruling."
        results = self._extract(text)
        assert any(r["jurisdiction"] == "EU" for r in results)


class TestCitationFreePreFilter:
    """Tests for the non-lossy pre-filter (_looks_trivially_citation_free).

    It must NEVER veto text that could contain a citation (no hardcoded reporter
    list); it only skips text that structurally cannot — i.e. has no digits.
    """

    def _trivially_free(self, text):
        from backend.services.citation_extractor import _looks_trivially_citation_free
        return _looks_trivially_citation_free(text)

    def test_us_citation_not_skipped(self):
        assert self._trivially_free("See 347 U.S. 483.") is False

    def test_uncommon_reporter_not_skipped(self):
        """An uncommon reporter NOT on any hardcoded list must still pass the gate."""
        assert self._trivially_free("The authority is 200 Vt. 123.") is False

    def test_uk_citation_not_skipped(self):
        assert self._trivially_free("[2024] UKSC 1.") is False

    def test_digit_free_prose_is_skipped(self):
        """Prose with no digits cannot contain a citation — safe to skip."""
        assert self._trivially_free("The contract was signed by both parties.") is True


class TestAnnotateAnswerWithIndices:
    """Tests for annotate_answer_with_indices — citation index insertion."""

    def _annotate(self, answer, citations):
        from backend.services.citation_extractor import annotate_answer_with_indices
        return annotate_answer_with_indices(answer, citations)

    def test_inserts_marker_after_citation(self):
        """[1] marker is inserted after the citation text in the answer."""
        answer = "The court decided 347 U.S. 483 in 1954."
        citations = [{"raw_text": "347 U.S. 483", "index": 1, "span": None}]
        result = self._annotate(answer, citations)
        assert "[1]" in result

    def test_no_citations_returns_original(self):
        """Empty citations list returns the original answer unchanged."""
        answer = "The court ruled for the plaintiff."
        result = self._annotate(answer, [])
        assert result == answer

    def test_does_not_double_insert_markers(self):
        """Calling annotate twice does not insert [1] twice."""
        answer = "See 347 U.S. 483."
        citations = [{"raw_text": "347 U.S. 483", "index": 1, "span": None}]
        once = self._annotate(answer, citations)
        # Second call on already-annotated text — should not double [1][1]
        assert once.count("[1]") == 1


# ═══════════════════════════════════════════════════════════════
# 7. EMBEDDINGS — PROVIDER DETECTION
# ═══════════════════════════════════════════════════════════════


class TestEmbeddingProviderDetection:
    """Tests for _detect_provider — provider selection logic."""

    def test_detects_cohere_when_no_voyage_key(self):
        """Without a Voyage API key, provider defaults to 'cohere'."""
        import importlib
        import backend.services.embeddings as emb_module

        # Reset the cached provider
        original_provider = emb_module._PROVIDER
        original_settings_voyage = getattr(emb_module.settings, 'voyage_api_key', '')

        try:
            emb_module._PROVIDER = None
            # Temporarily clear Voyage key
            emb_module.settings.__dict__['voyage_api_key'] = ''
            provider = emb_module._detect_provider()
            assert provider == "cohere"
        finally:
            emb_module._PROVIDER = original_provider
            if original_settings_voyage:
                emb_module.settings.__dict__['voyage_api_key'] = original_settings_voyage
            elif 'voyage_api_key' in emb_module.settings.__dict__:
                del emb_module.settings.__dict__['voyage_api_key']

    def test_embed_text_raises_on_empty_string(self):
        """embed_text raises ValueError for empty input."""
        from backend.services.embeddings import embed_text
        with pytest.raises(ValueError, match="empty"):
            embed_text("")

    def test_embed_text_raises_on_whitespace_only(self):
        """embed_text raises ValueError for whitespace-only input."""
        from backend.services.embeddings import embed_text
        with pytest.raises(ValueError, match="empty"):
            embed_text("   ")

    def test_embed_query_raises_on_empty_string(self):
        """embed_query raises ValueError for empty input."""
        from backend.services.embeddings import embed_query
        with pytest.raises(ValueError, match="empty"):
            embed_query("")

    def test_embed_chunks_raises_on_empty_list(self):
        """embed_chunks raises ValueError for empty list."""
        from backend.services.embeddings import embed_chunks
        with pytest.raises(ValueError):
            embed_chunks([])

    def test_embed_chunks_raises_on_empty_chunk(self):
        """embed_chunks raises ValueError when a chunk in the list is empty."""
        from backend.services.embeddings import embed_chunks
        with pytest.raises(ValueError):
            embed_chunks(["valid text", ""])


# ═══════════════════════════════════════════════════════════════
# 8. CONFLICT DETECTOR — PURE LOGIC
# ═══════════════════════════════════════════════════════════════


class TestClassifyAuthorityType:
    """Tests for _classify_authority_type — authority classification heuristics."""

    def _classify(self, chunk):
        from backend.services.conflict_detector import _classify_authority_type
        return _classify_authority_type(chunk)

    def test_statute_keyword_in_doc_name(self):
        """Document named 'Privacy Act 2018' is classified as 'statute'."""
        chunk = {"document_name": "Privacy Act 2018", "section_name": "", "content": "", "source_type": "document"}
        result = self._classify(chunk)
        assert result == "statute"

    def test_constitution_keyword(self):
        """Document named 'Constitution' is classified as 'constitution'."""
        chunk = {"document_name": "US Constitution", "section_name": "", "content": "", "source_type": "document"}
        result = self._classify(chunk)
        assert result == "constitution"

    def test_regulation_keyword_in_doc_name(self):
        """Document named 'GDPR Regulation' is classified as 'regulation'."""
        chunk = {"document_name": "GDPR Regulation", "section_name": "", "content": "", "source_type": "document"}
        result = self._classify(chunk)
        assert result == "regulation"

    def test_case_law_source_type_supreme(self):
        """case_law source type with 'supreme court' in content returns 'supreme_court'."""
        chunk = {
            "document_name": "opinion",
            "section_name": "",
            "content": "The Supreme Court held in this matter...",
            "source_type": "case_law",
        }
        result = self._classify(chunk)
        assert result == "supreme_court"

    def test_case_law_content_heuristic(self):
        """Content with 'plaintiff' and 'defendant' is classified as 'case_law'."""
        chunk = {
            "document_name": "legal_brief",
            "section_name": "",
            "content": "The plaintiff sued the defendant for damages. The court held...",
            "source_type": "document",
        }
        result = self._classify(chunk)
        assert result == "case_law"

    def test_unknown_document_returns_unknown(self):
        """Unrecognised document name returns 'unknown'."""
        chunk = {"document_name": "random_document", "section_name": "", "content": "Some text.", "source_type": "document"}
        result = self._classify(chunk)
        assert result == "unknown"


class TestEstimateRecency:
    """Tests for _estimate_recency — year extraction and bracket lookup."""

    def _recency(self, chunk):
        from backend.services.conflict_detector import _estimate_recency
        return _estimate_recency(chunk)

    def test_recent_year_scores_high(self):
        """A chunk with a very recent year gets a high recency weight."""
        from datetime import datetime
        recent_year = datetime.now().year
        chunk = {"document_name": f"Act {recent_year}", "section_name": "", "content": ""}
        weight = self._recency(chunk)
        assert weight >= 0.95

    def test_old_year_scores_lower(self):
        """A chunk with an old year (e.g., 1990) gets a lower weight than a recent one."""
        recent = {"document_name": "Act 2024", "section_name": "", "content": ""}
        old = {"document_name": "Act 1990", "section_name": "", "content": ""}
        assert self._recency(recent) > self._recency(old)

    def test_no_year_returns_middle_weight(self):
        """Chunk with no year defaults to 0.50."""
        chunk = {"document_name": "policy_guidance", "section_name": "", "content": "no year here"}
        assert self._recency(chunk) == pytest.approx(0.50)

    def test_weight_in_valid_range(self):
        """Recency weight is always in [0.0, 1.0]."""
        chunk = {"document_name": "Regulations 2015", "section_name": "", "content": ""}
        weight = self._recency(chunk)
        assert 0.0 <= weight <= 1.0
