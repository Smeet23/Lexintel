"""Unit tests for as-of-date temporal queries (May 2026 feature).

Covers pure, infra-free logic:
- rag_engine._detect_temporal_intent  (NL intent → (as_of_date, exclude_superseded))
- vector_store.build_temporal_filter  (sentinel dict shape, base-filter merge, immutability)
- vector_store._build_qdrant_filter   (sentinel translation → Qdrant must/must_not/should)

No live Qdrant or DB is touched; the Qdrant Filter object is introspected directly.
"""
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ═══════════════════════════════════════════════════════════════
# _detect_temporal_intent
# ═══════════════════════════════════════════════════════════════


class TestDetectTemporalIntent:
    def _detect(self, query):
        from backend.services.rag_engine import _detect_temporal_intent
        return _detect_temporal_intent(query)

    def test_default_query_is_current_law(self):
        as_of, exclude = self._detect("What are the data protection obligations?")
        assert as_of is None
        assert exclude is True

    def test_empty_query(self):
        as_of, exclude = self._detect("")
        assert as_of is None
        assert exclude is True

    def test_as_of_explicit_date(self):
        as_of, exclude = self._detect("What was the law as of 1 January 2020?")
        assert as_of is not None
        assert as_of.year == 2020 and as_of.month == 1 and as_of.day == 1
        assert exclude is False  # as-of mode keeps in-force-then docs

    def test_as_of_iso_date(self):
        as_of, exclude = self._detect("Show the rules as of 2019-06-15")
        assert as_of is not None
        assert as_of.year == 2019 and as_of.month == 6 and as_of.day == 15
        assert exclude is False

    def test_before_year(self):
        as_of, exclude = self._detect("What did the statute say before 2018?")
        assert as_of is not None
        assert as_of.year == 2018 and as_of.month == 1 and as_of.day == 1
        assert exclude is False

    def test_prior_to_year(self):
        as_of, exclude = self._detect("the regime prior to 2016")
        assert as_of is not None
        assert as_of.year == 2016
        assert exclude is False

    def test_in_year_midyear_snapshot(self):
        as_of, exclude = self._detect("What was the position in 2015?")
        assert as_of is not None
        assert as_of.year == 2015 and as_of.month == 7 and as_of.day == 1
        assert exclude is False

    def test_under_the_year_act(self):
        as_of, exclude = self._detect("rights under the 1998 act")
        assert as_of is not None
        assert as_of.year == 1998
        assert exclude is False

    def test_current_law_intent(self):
        as_of, exclude = self._detect("What is the current law on privacy?")
        assert as_of is None
        assert exclude is True

    def test_currently_intent(self):
        as_of, exclude = self._detect("What rules currently apply?")
        assert as_of is None
        assert exclude is True

    def test_historical_intent_keeps_superseded(self):
        as_of, exclude = self._detect("Give me the history of this regulation")
        assert as_of is None
        assert exclude is False

    def test_all_versions_intent(self):
        as_of, exclude = self._detect("Show all versions of the policy over time")
        assert as_of is None
        assert exclude is False

    def test_negative_section_number_not_treated_as_year(self):
        # "section 12" has no year-band match → default current-law, no as-of
        as_of, exclude = self._detect("What does section 12 require?")
        assert as_of is None
        assert exclude is True

    def test_citation_number_below_year_band_ignored(self):
        # A bare 3-digit / out-of-band number must not become an as-of date.
        as_of, exclude = self._detect("Refer to clause 42 of the agreement")
        assert as_of is None
        assert exclude is True


# ═══════════════════════════════════════════════════════════════
# build_temporal_filter
# ═══════════════════════════════════════════════════════════════


class TestBuildTemporalFilter:
    def _build(self, **kwargs):
        from backend.services.vector_store import build_temporal_filter
        return build_temporal_filter(**kwargs)

    def test_no_date_current_only(self):
        f = self._build(as_of_date=None, exclude_superseded=True)
        assert f["_temporal_as_of"] is None
        assert f["_temporal_current_only"] is True

    def test_no_date_historical(self):
        f = self._build(as_of_date=None, exclude_superseded=False)
        assert f["_temporal_as_of"] is None
        assert f["_temporal_current_only"] is False

    def test_as_of_date_serialized_to_iso(self):
        d = datetime(2020, 1, 1, tzinfo=timezone.utc)
        f = self._build(as_of_date=d, exclude_superseded=False)
        assert f["_temporal_as_of"] == d.isoformat()
        # current_only meaningless when as-of set
        assert f["_temporal_current_only"] is False

    def test_as_of_overrides_current_only(self):
        d = datetime(2020, 1, 1, tzinfo=timezone.utc)
        f = self._build(as_of_date=d, exclude_superseded=True)
        # Even with exclude_superseded=True, an explicit as-of disables current_only.
        assert f["_temporal_current_only"] is False

    def test_iso_string_as_of_accepted(self):
        f = self._build(as_of_date="2019-06-15T00:00:00+00:00", exclude_superseded=False)
        assert f["_temporal_as_of"] == "2019-06-15T00:00:00+00:00"

    def test_base_filter_merged(self):
        base = {"jurisdiction": "UK"}
        f = self._build(as_of_date=None, exclude_superseded=True, base_filter=base)
        assert f["jurisdiction"] == "UK"
        assert "_temporal_as_of" in f

    def test_base_filter_not_mutated(self):
        base = {"jurisdiction": "UK"}
        self._build(as_of_date=None, exclude_superseded=True, base_filter=base)
        # immutable-style: caller's dict untouched
        assert base == {"jurisdiction": "UK"}


# ═══════════════════════════════════════════════════════════════
# _build_qdrant_filter  (sentinel → Qdrant Filter translation)
# ═══════════════════════════════════════════════════════════════


class TestBuildQdrantFilter:
    def _translate(self, qf):
        from backend.services.vector_store import _build_qdrant_filter
        return _build_qdrant_filter(qf)

    def test_none_filter_returns_none(self):
        assert self._translate(None) is None

    def test_empty_dict_returns_none(self):
        assert self._translate({}) is None

    def test_plain_field_only(self):
        from backend.services.vector_store import build_temporal_filter
        qf = build_temporal_filter(
            as_of_date=None, exclude_superseded=False, base_filter={"jurisdiction": "UK"}
        )
        result = self._translate(qf)
        assert result is not None
        keys = {c.key for c in result.must}
        assert "jurisdiction" in keys
        assert result.must_not is None

    def test_current_only_excludes_superseded_via_must_not(self):
        # current-law filtering must EXCLUDE superseded/repealed (must_not), not
        # REQUIRE document_status=='current' (which would drop 'unknown' chunks).
        from backend.services.vector_store import build_temporal_filter
        qf = build_temporal_filter(as_of_date=None, exclude_superseded=True)
        result = self._translate(qf)
        must_not_vals = {c.match.value for c in (result.must_not or [])}
        assert {"superseded", "repealed"}.issubset(must_not_vals)
        must_status_vals = {
            c.match.value for c in (result.must or [])
            if getattr(c, "key", None) == "document_status"
        }
        assert "current" not in must_status_vals

    def test_as_of_builds_effective_range_and_superseded_mustnot(self, monkeypatch):
        import backend.services.vector_store as vs

        # Ensure undated inclusion is on so the should-branch is present.
        monkeypatch.setattr(vs.settings, "temporal_include_undated", True, raising=False)

        d = datetime(2020, 1, 1, tzinfo=timezone.utc)
        qf = vs.build_temporal_filter(as_of_date=d, exclude_superseded=False)
        result = self._translate(qf)

        # must contains a nested Filter with should = [effective_date range, IsEmpty]
        nested = [c for c in result.must if hasattr(c, "should") and c.should]
        assert len(nested) == 1
        should_keys = []
        for cond in nested[0].should:
            if hasattr(cond, "key"):
                should_keys.append(cond.key)
            elif hasattr(cond, "is_empty"):
                should_keys.append(cond.is_empty.key)
        assert "effective_date" in should_keys

        # must_not excludes superseded_date <= as_of
        assert result.must_not is not None
        mn_keys = {c.key for c in result.must_not}
        assert "superseded_date" in mn_keys

    def test_as_of_without_undated_inclusion_omits_isempty(self, monkeypatch):
        import backend.services.vector_store as vs
        monkeypatch.setattr(vs.settings, "temporal_include_undated", False, raising=False)

        d = datetime(2020, 1, 1, tzinfo=timezone.utc)
        qf = vs.build_temporal_filter(as_of_date=d, exclude_superseded=False)
        result = self._translate(qf)

        # effective_date is a direct FieldCondition (no nested should branch)
        direct = [c for c in result.must if getattr(c, "key", None) == "effective_date"]
        assert len(direct) == 1
        nested = [c for c in result.must if hasattr(c, "should") and c.should]
        assert nested == []

    def test_input_dict_not_mutated(self):
        from backend.services.vector_store import build_temporal_filter
        qf = build_temporal_filter(
            as_of_date=None, exclude_superseded=True, base_filter={"jurisdiction": "UK"}
        )
        snapshot = dict(qf)
        self._translate(qf)
        assert qf == snapshot  # sentinels popped from a copy, not the input
