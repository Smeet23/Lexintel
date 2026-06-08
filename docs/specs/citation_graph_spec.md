# Legal Citation Knowledge Graph -- Implementation Specification

**Status:** Draft
**Author:** Engineering
**Date:** 2026-03-23
**Target:** Lexintel v2.x
**Estimated effort:** 6 weeks (1 engineer)

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Architecture Decision: PostgreSQL + Apache AGE](#2-architecture-decision-postgresql--apache-age)
3. [Graph Schema](#3-graph-schema)
4. [Infrastructure Setup](#4-infrastructure-setup)
5. [Backend Implementation](#5-backend-implementation)
6. [Citation Extraction Pipeline](#6-citation-extraction-pipeline)
7. [LLM Relationship Classification](#7-llm-relationship-classification)
8. [Graph Query Library](#8-graph-query-library)
9. [Graph-Enhanced RAG Retrieval](#9-graph-enhanced-rag-retrieval)
10. [REST API Endpoints](#10-rest-api-endpoints)
11. [Frontend Citation Network Visualization](#11-frontend-citation-network-visualization)
12. [Performance Targets and Indexing](#12-performance-targets-and-indexing)
13. [Testing Strategy](#13-testing-strategy)
14. [Migration and Rollback](#14-migration-and-rollback)
15. [Six-Week Roadmap](#15-six-week-roadmap)
16. [Appendix: File Inventory](#16-appendix-file-inventory)

---

## 1. Motivation

Lexintel's RAG pipeline currently treats each uploaded document as an isolated bag of chunks. When a user asks "Is this case still good law?" or "What precedents support this ruling?", the system can only answer from literal text found in the uploaded PDFs. It cannot:

- Trace citation chains across documents (Case A cites Case B which cites Case C).
- Detect that a case has been overruled, distinguished, or reversed by a later decision.
- Find structurally similar cases that cite the same authorities.
- Boost retrieval by injecting graph-adjacent context into the RAG prompt.

A citation knowledge graph solves all of these. Every time a document is ingested, we extract legal citations, resolve them to canonical nodes, and classify the relationship between citing and cited authorities. The graph accumulates across all matters, becoming more valuable with every upload.

---

## 2. Architecture Decision: PostgreSQL + Apache AGE

### Why Not Neo4j

| Factor | Neo4j | PostgreSQL + AGE |
|--------|-------|------------------|
| License cost | Community Edition is GPL; Enterprise is commercial ($$$) | Apache 2.0, free forever |
| Infrastructure | Separate server, separate backups, separate monitoring | Same PostgreSQL instance we already run |
| Operational complexity | New ops surface (clustering, backup, HA) | Zero additional ops -- same `pg_dump`, same Alembic, same connection pool |
| Query language | Cypher (native) | Cypher (via AGE extension, same syntax) |
| Scale ceiling | Billions of edges | ~10M edges with proper indexing (sufficient for 100K+ legal documents) |
| Upgrade path | N/A | Export with `agtype` to CSV, import into Neo4j if/when needed |
| Transactional consistency | Separate tx boundary from PostgreSQL | Same ACID transaction as relational writes |

### Decision

Use Apache AGE on our existing PostgreSQL instance. Zero marginal cost, zero ops overhead, same backup strategy, and Cypher query compatibility means we can migrate to Neo4j in the future if graph scale exceeds AGE's practical limits.

### Scale Projections

| Metric | Year 1 | Year 3 (projected) |
|--------|--------|---------------------|
| Documents ingested | 5,000 | 50,000 |
| Case nodes | 25,000 | 250,000 |
| Statute nodes | 10,000 | 100,000 |
| Edges | 100,000 | 2,000,000 |
| AGE practical limit | -- | ~10,000,000 edges |

---

## 3. Graph Schema

### 3.1 Node Types

#### Case

Represents a judicial decision (opinion, order, ruling).

```
(:Case {
    citation: STRING,          -- canonical citation ("347 U.S. 483")
    case_name: STRING,         -- "Brown v. Board of Education"
    court: STRING,             -- "Supreme Court of the United States"
    date: STRING,              -- "1954-05-17" (ISO 8601)
    jurisdiction: STRING,      -- "US" | "UK" | "EU" | "IN" | "AU" | "CA" | "SG"
    status: STRING,            -- "good_law" | "caution" | "bad_law" | "unknown"
    document_id: STRING,       -- UUID of Lexintel Document (if uploaded), NULL if external
    matter_id: STRING,         -- UUID of Matter (if uploaded)
    courtlistener_id: STRING,  -- CourtListener opinion ID (for US cases)
    url: STRING,               -- external URL
    created_at: STRING,        -- ISO timestamp
    updated_at: STRING         -- ISO timestamp
})
```

#### Statute

Represents a statutory provision, regulation, or rule.

```
(:Statute {
    citation: STRING,          -- canonical citation ("42 U.S.C. ss 1983")
    title: STRING,             -- "Civil Rights Act"
    jurisdiction: STRING,      -- "US" | "UK" | "EU" | "IN" | "AU" | "CA" | "SG"
    section: STRING,           -- specific section/article number
    in_force: BOOLEAN,         -- whether currently in force
    url: STRING,               -- link to official text
    created_at: STRING,
    updated_at: STRING
})
```

#### Court

Represents a court or tribunal in the judicial hierarchy.

```
(:Court {
    name: STRING,              -- "Supreme Court of the United States"
    short_name: STRING,        -- "SCOTUS"
    jurisdiction: STRING,      -- "US"
    level: INTEGER,            -- hierarchy level (1=highest)
    country: STRING,           -- "United States"
    created_at: STRING
})
```

### 3.2 Edge Types

| Edge | Source | Target | Meaning | Properties |
|------|--------|--------|---------|------------|
| `CITES` | Case | Case or Statute | Source cites target | `{context, page_num, strength}` |
| `OVERRULES` | Case | Case | Source overrules target (target is bad law) | `{date, holding}` |
| `DISTINGUISHES` | Case | Case | Source distinguishes target (limits applicability) | `{basis, context}` |
| `FOLLOWS` | Case | Case | Source follows/applies target as authority | `{context}` |
| `REVERSES` | Case | Case | Appellate reversal of lower court decision | `{date, scope}` |
| `APPLIES` | Case | Statute | Case applies/interprets the statute | `{interpretation, section}` |
| `DECIDED_BY` | Case | Court | Case was decided by this court | `{date}` |

### 3.3 Edge Property Details

```
[:CITES {
    context: STRING,       -- surrounding sentence where citation appears
    page_num: STRING,      -- page in source document
    strength: STRING,      -- "primary" | "supporting" | "cf" | "see_also" | "but_see"
    extraction_method: STRING,  -- "eyecite" | "regex" | "llm"
    created_at: STRING
}]

[:OVERRULES {
    date: STRING,          -- date of overruling decision
    holding: STRING,       -- brief description of what was overruled
    scope: STRING,         -- "full" | "partial"
    created_at: STRING
}]

[:DISTINGUISHES {
    basis: STRING,         -- factual or legal basis for distinction
    context: STRING,       -- surrounding text
    created_at: STRING
}]

[:FOLLOWS {
    context: STRING,       -- how the case follows the authority
    created_at: STRING
}]

[:REVERSES {
    date: STRING,
    scope: STRING,         -- "full" | "partial" | "remand"
    created_at: STRING
}]

[:APPLIES {
    interpretation: STRING,  -- how the court interpreted the statute
    section: STRING,         -- specific section applied
    created_at: STRING
}]
```

---

## 4. Infrastructure Setup

### 4.1 Install Apache AGE

AGE is a PostgreSQL extension. Install method depends on deployment:

**Local development (macOS with Homebrew):**

```bash
# AGE requires building from source against PG headers
# Assumes PostgreSQL 16 installed via Homebrew
brew install cmake

git clone https://github.com/apache/age.git /tmp/age
cd /tmp/age
git checkout release/PG16/1.5.0   # match your PG major version

make PG_CONFIG=$(brew --prefix postgresql@16)/bin/pg_config
sudo make install PG_CONFIG=$(brew --prefix postgresql@16)/bin/pg_config
```

**Docker (add to existing docker-compose or Dockerfile):**

```dockerfile
FROM apache/age:PG16_latest
# AGE is pre-installed; just use this as the PostgreSQL image
```

Or add to `docker-compose.yml`:

```yaml
services:
  postgres:
    image: apache/age:PG16_latest
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: lexintel
      POSTGRES_USER: lexintel
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
```

**Production (managed PostgreSQL):**

- AWS RDS: Not natively supported. Use Aurora with custom extensions or self-managed EC2.
- Azure Database for PostgreSQL Flexible Server: Supports AGE as an extension.
- Self-hosted: Build AGE from source against your PG version.

### 4.2 Database Initialization SQL

Run once after AGE is installed. This will be executed by the Alembic migration (see Section 14).

```sql
-- Enable the AGE extension
CREATE EXTENSION IF NOT EXISTS age;

-- Load AGE into the session (required per-connection)
LOAD 'age';

-- Add ag_catalog to search path so Cypher functions resolve
SET search_path TO ag_catalog, public;

-- Create the legal citation graph
SELECT * FROM ag_catalog.create_graph('legal_cases');
```

### 4.3 Connection Configuration

Add to `backend/config.py`:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    # Citation Graph
    citation_graph_enabled: bool = False    # Feature flag, off by default
    citation_graph_name: str = "legal_cases"
    age_graph_path: str = "ag_catalog"      # AGE schema
```

### 4.4 Per-Connection AGE Initialization

AGE requires `LOAD 'age'` and `SET search_path` on every new database connection. We handle this with a SQLAlchemy event listener in `database.py`:

```python
from sqlalchemy import event, text

def _init_age_connection(dbapi_conn, connection_record):
    """Initialize AGE extension on each new connection."""
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute("LOAD 'age';")
        cursor.execute("SET search_path TO ag_catalog, public;")
    except Exception:
        pass  # AGE not installed -- graph features will fail gracefully
    finally:
        cursor.close()

# In init_db(), after creating the engine:
if settings.citation_graph_enabled:
    event.listen(engine.pool, "connect", _init_age_connection)
```

---

## 5. Backend Implementation

### 5.1 New Files

| File | Purpose |
|------|---------|
| `backend/services/citation_graph.py` | Graph CRUD: create/merge nodes, create edges, update status |
| `backend/services/graph_queries.py` | Read-only Cypher queries: good law check, precedent chains, similar cases |
| `backend/alembic/versions/xxxx_add_citation_graph.py` | Alembic migration to install AGE and create graph |

### 5.2 Modified Files

| File | Change |
|------|--------|
| `backend/config.py` | Add `citation_graph_enabled`, `citation_graph_name` settings |
| `backend/database.py` | Add AGE connection initializer event listener |
| `backend/tasks.py` | Call `extract_and_index_citations()` after embedding step |
| `backend/services/rag_engine.py` | Inject graph context into RAG prompt |
| `backend/requirements.txt` | Add `age` Python driver (or use raw SQL via SQLAlchemy) |
| `backend/schemas.py` | Add graph query/response schemas |
| `backend/main.py` | Register graph API routes |
| `frontend/lib/types.ts` | Add graph types |
| `frontend/lib/api-services.ts` | Add graph API functions |
| `frontend/components/CitationGraph.tsx` | New: D3.js network visualization |

### 5.3 `backend/services/citation_graph.py` -- Full Implementation

```python
"""Legal citation knowledge graph backed by PostgreSQL + Apache AGE.

Provides CRUD operations for Case, Statute, and Court nodes, plus
typed edges (CITES, OVERRULES, DISTINGUISHES, FOLLOWS, REVERSES, APPLIES).

All Cypher queries are executed via AGE's `cypher()` SQL function,
wrapped in SQLAlchemy `text()` calls to share the existing connection pool.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cypher(db: Session, query: str, graph: str = "legal_cases") -> List[Dict]:
    """Execute a Cypher query via AGE and return results as dicts.

    AGE wraps Cypher in a SQL function:
        SELECT * FROM cypher('graph_name', $$ CYPHER $$) AS (col agtype);

    We parse the agtype results into Python dicts.
    """
    # AGE requires results to be cast. We use a generic single-column return
    # and parse agtype on the Python side.
    sql = text(f"""
        SELECT * FROM cypher('{graph}', $$
            {query}
        $$) AS (result agtype);
    """)
    try:
        rows = db.execute(sql).fetchall()
        results = []
        for row in rows:
            val = row[0]
            # agtype comes back as a string representation; parse it
            if val is not None:
                results.append(_parse_agtype(val))
        return results
    except Exception as e:
        logger.error(f"Cypher execution failed: {e}\nQuery: {query}")
        raise


def _parse_agtype(val: Any) -> Any:
    """Parse AGE agtype value into Python native type.

    agtype values are returned as strings that look like JSON but with
    type suffixes. This parser handles the common cases.
    """
    import json

    if val is None:
        return None

    s = str(val)

    # agtype wraps values like: {"key": "value"}::vertex or "string"::text
    # Strip type annotations
    for suffix in ["::vertex", "::edge", "::path", "::text", "::integer",
                    "::float", "::boolean", "::agtype"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break

    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


# ---------------------------------------------------------------------------
# Node Operations
# ---------------------------------------------------------------------------

def merge_case_node(
    db: Session,
    citation: str,
    case_name: Optional[str] = None,
    court: Optional[str] = None,
    date: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    status: str = "unknown",
    document_id: Optional[str] = None,
    matter_id: Optional[str] = None,
    courtlistener_id: Optional[str] = None,
    url: Optional[str] = None,
) -> Dict:
    """Create or update a Case node. MERGE ensures idempotency on citation.

    Args:
        db: SQLAlchemy session (must have AGE loaded on its connection).
        citation: Canonical citation string (e.g. "347 U.S. 483").
        case_name: Full case name.
        court: Court name.
        date: Decision date (ISO 8601).
        jurisdiction: Two-letter jurisdiction code.
        status: "good_law" | "caution" | "bad_law" | "unknown".
        document_id: Lexintel Document UUID if this case was uploaded.
        matter_id: Lexintel Matter UUID.
        courtlistener_id: CourtListener opinion ID.
        url: External URL.

    Returns:
        Dict with the merged node properties.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Escape single quotes in string values for Cypher
    def esc(v: Optional[str]) -> str:
        if v is None:
            return "null"
        return "'" + v.replace("'", "\\'") + "'"

    query = f"""
        MERGE (c:Case {{citation: {esc(citation)}}})
        ON CREATE SET
            c.case_name = {esc(case_name)},
            c.court = {esc(court)},
            c.date = {esc(date)},
            c.jurisdiction = {esc(jurisdiction)},
            c.status = {esc(status)},
            c.document_id = {esc(document_id)},
            c.matter_id = {esc(matter_id)},
            c.courtlistener_id = {esc(courtlistener_id)},
            c.url = {esc(url)},
            c.created_at = {esc(now)},
            c.updated_at = {esc(now)}
        ON MATCH SET
            c.case_name = COALESCE({esc(case_name)}, c.case_name),
            c.court = COALESCE({esc(court)}, c.court),
            c.date = COALESCE({esc(date)}, c.date),
            c.jurisdiction = COALESCE({esc(jurisdiction)}, c.jurisdiction),
            c.status = CASE WHEN {esc(status)} <> 'unknown'
                            THEN {esc(status)} ELSE c.status END,
            c.document_id = COALESCE({esc(document_id)}, c.document_id),
            c.matter_id = COALESCE({esc(matter_id)}, c.matter_id),
            c.courtlistener_id = COALESCE({esc(courtlistener_id)}, c.courtlistener_id),
            c.url = COALESCE({esc(url)}, c.url),
            c.updated_at = {esc(now)}
        RETURN c
    """
    results = _cypher(db, query)
    return results[0] if results else {}


def merge_statute_node(
    db: Session,
    citation: str,
    title: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    section: Optional[str] = None,
    in_force: bool = True,
    url: Optional[str] = None,
) -> Dict:
    """Create or update a Statute node."""
    now = datetime.now(timezone.utc).isoformat()

    def esc(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        return "'" + str(v).replace("'", "\\'") + "'"

    query = f"""
        MERGE (s:Statute {{citation: {esc(citation)}}})
        ON CREATE SET
            s.title = {esc(title)},
            s.jurisdiction = {esc(jurisdiction)},
            s.section = {esc(section)},
            s.in_force = {esc(in_force)},
            s.url = {esc(url)},
            s.created_at = {esc(now)},
            s.updated_at = {esc(now)}
        ON MATCH SET
            s.title = COALESCE({esc(title)}, s.title),
            s.jurisdiction = COALESCE({esc(jurisdiction)}, s.jurisdiction),
            s.in_force = {esc(in_force)},
            s.updated_at = {esc(now)}
        RETURN s
    """
    results = _cypher(db, query)
    return results[0] if results else {}


def merge_court_node(
    db: Session,
    name: str,
    short_name: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    level: Optional[int] = None,
    country: Optional[str] = None,
) -> Dict:
    """Create or update a Court node."""
    now = datetime.now(timezone.utc).isoformat()

    def esc(v):
        if v is None:
            return "null"
        if isinstance(v, int):
            return str(v)
        return "'" + str(v).replace("'", "\\'") + "'"

    query = f"""
        MERGE (ct:Court {{name: {esc(name)}}})
        ON CREATE SET
            ct.short_name = {esc(short_name)},
            ct.jurisdiction = {esc(jurisdiction)},
            ct.level = {esc(level)},
            ct.country = {esc(country)},
            ct.created_at = {esc(now)}
        ON MATCH SET
            ct.short_name = COALESCE({esc(short_name)}, ct.short_name),
            ct.jurisdiction = COALESCE({esc(jurisdiction)}, ct.jurisdiction),
            ct.level = COALESCE({esc(level)}, ct.level)
        RETURN ct
    """
    results = _cypher(db, query)
    return results[0] if results else {}


# ---------------------------------------------------------------------------
# Edge Operations
# ---------------------------------------------------------------------------

def create_edge(
    db: Session,
    source_citation: str,
    source_label: str,
    target_citation: str,
    target_label: str,
    edge_type: str,
    properties: Optional[Dict] = None,
) -> Dict:
    """Create a typed edge between two nodes.

    Args:
        db: SQLAlchemy session.
        source_citation: Citation of the source node.
        source_label: "Case" or "Statute".
        target_citation: Citation of the target node.
        target_label: "Case" or "Statute".
        edge_type: One of CITES, OVERRULES, DISTINGUISHES, FOLLOWS, REVERSES, APPLIES.
        properties: Optional dict of edge properties.

    Returns:
        Dict with the created edge.
    """
    VALID_EDGES = {"CITES", "OVERRULES", "DISTINGUISHES", "FOLLOWS", "REVERSES", "APPLIES"}
    if edge_type not in VALID_EDGES:
        raise ValueError(f"Invalid edge type '{edge_type}'. Must be one of {VALID_EDGES}")

    VALID_LABELS = {"Case", "Statute", "Court"}
    if source_label not in VALID_LABELS or target_label not in VALID_LABELS:
        raise ValueError(f"Invalid label. Must be one of {VALID_LABELS}")

    now = datetime.now(timezone.utc).isoformat()
    props = properties or {}
    props["created_at"] = now

    def esc(v):
        if v is None:
            return "null"
        return "'" + str(v).replace("'", "\\'") + "'"

    # Build property string for the edge
    prop_parts = []
    for k, v in props.items():
        prop_parts.append(f"{k}: {esc(v)}")
    prop_str = "{" + ", ".join(prop_parts) + "}" if prop_parts else ""

    query = f"""
        MATCH (a:{source_label} {{citation: {esc(source_citation)}}})
        MATCH (b:{target_label} {{citation: {esc(target_citation)}}})
        CREATE (a)-[r:{edge_type} {prop_str}]->(b)
        RETURN r
    """
    results = _cypher(db, query)
    return results[0] if results else {}


def merge_cites_edge(
    db: Session,
    source_citation: str,
    target_citation: str,
    target_label: str = "Case",
    context: Optional[str] = None,
    page_num: Optional[str] = None,
    strength: str = "supporting",
    extraction_method: str = "regex",
) -> Dict:
    """Create a CITES edge (idempotent via MERGE).

    This is the most common edge -- a simple citation reference. Uses MERGE
    to avoid duplicates when the same citation is found in multiple chunks.
    """
    now = datetime.now(timezone.utc).isoformat()

    def esc(v):
        if v is None:
            return "null"
        return "'" + str(v).replace("'", "\\'") + "'"

    query = f"""
        MATCH (a:Case {{citation: {esc(source_citation)}}})
        MATCH (b:{target_label} {{citation: {esc(target_citation)}}})
        MERGE (a)-[r:CITES]->(b)
        ON CREATE SET
            r.context = {esc(context)},
            r.page_num = {esc(page_num)},
            r.strength = {esc(strength)},
            r.extraction_method = {esc(extraction_method)},
            r.created_at = {esc(now)}
        RETURN r
    """
    results = _cypher(db, query)
    return results[0] if results else {}


# ---------------------------------------------------------------------------
# Bulk Indexing (called from tasks.py)
# ---------------------------------------------------------------------------

def extract_and_index_citations(
    db: Session,
    document_id: str,
    matter_id: str,
    document_name: str,
    chunks: List[Dict],
    jurisdiction: str = "US",
) -> Dict[str, int]:
    """Extract citations from document chunks and index them in the graph.

    This is the main entry point called from process_document_task().

    Pipeline:
    1. Determine if the document itself is a case (create a source Case node).
    2. For each chunk, extract citations using the existing citation_extractor.
    3. For each extracted citation, MERGE a target node (Case or Statute).
    4. Create CITES edges from the source case to each target.
    5. Batch classify relationships with Gemini (OVERRULES/DISTINGUISHES/etc).

    Args:
        db: Database session with AGE initialized.
        document_id: UUID of the Document being processed.
        matter_id: UUID of the Matter.
        document_name: Filename for deriving case name.
        chunks: List of chunk dicts with 'content', 'page_num' keys.
        jurisdiction: Default jurisdiction code.

    Returns:
        Dict with counts: {"nodes_created": N, "edges_created": N, "relationships_classified": N}
    """
    try:
        from backend.services.citation_extractor import extract_citations_regex
        from backend.services.citation_extractor import extract_all_citations
    except ImportError:
        try:
            from services.citation_extractor import extract_citations_regex
            from services.citation_extractor import extract_all_citations
        except ImportError:
            from .citation_extractor import extract_citations_regex
            from .citation_extractor import extract_all_citations

    stats = {"nodes_created": 0, "edges_created": 0, "relationships_classified": 0}

    # 1. Try to identify the source document as a case
    source_citation = _infer_source_citation(document_name, chunks)
    source_case_name = _infer_case_name(document_name)

    if source_citation:
        merge_case_node(
            db,
            citation=source_citation,
            case_name=source_case_name,
            jurisdiction=jurisdiction,
            document_id=document_id,
            matter_id=matter_id,
        )
        stats["nodes_created"] += 1
    else:
        # Document is not a recognizable case -- still extract citations
        # but without a source node for edges
        logger.info(f"Document {document_name} is not a recognized case; "
                     "extracting citations without source node")

    # 2. Extract citations from all chunks
    all_extracted = []
    for chunk in chunks:
        content = chunk.get("content", "")
        page_num = chunk.get("page_num", "")

        # Use fast regex extraction (no API cost)
        citations = extract_citations_regex(content)

        for cite in citations:
            cite["page_num"] = page_num
            cite["context"] = _extract_context(content, cite.get("span"))
            all_extracted.append(cite)

    logger.info(f"Extracted {len(all_extracted)} raw citations from {len(chunks)} chunks")

    # 3. Deduplicate by citation text
    unique_citations = {}
    for cite in all_extracted:
        key = cite["raw_text"].strip().lower()
        if key not in unique_citations:
            unique_citations[key] = cite
        else:
            # Keep the one with more context
            existing = unique_citations[key]
            if len(cite.get("context", "")) > len(existing.get("context", "")):
                unique_citations[key] = cite

    # 4. Create target nodes and CITES edges
    for cite in unique_citations.values():
        target_citation = cite["raw_text"].strip()
        target_jurisdiction = cite.get("jurisdiction", jurisdiction)

        # Determine if it is a case or statute
        target_label = _classify_citation_type(target_citation, target_jurisdiction)

        if target_label == "Case":
            merge_case_node(
                db,
                citation=target_citation,
                jurisdiction=target_jurisdiction,
                case_name=cite.get("case_name"),
            )
        else:
            merge_statute_node(
                db,
                citation=target_citation,
                jurisdiction=target_jurisdiction,
            )
        stats["nodes_created"] += 1

        # Create CITES edge if we have a source case
        if source_citation:
            merge_cites_edge(
                db,
                source_citation=source_citation,
                target_citation=target_citation,
                target_label=target_label,
                context=cite.get("context"),
                page_num=cite.get("page_num"),
                extraction_method=cite.get("extraction_method", "regex"),
            )
            stats["edges_created"] += 1

    # 5. Batch classify relationships (async, best-effort)
    if source_citation and unique_citations:
        try:
            import asyncio
            classified = asyncio.run(
                _classify_relationships_batch(
                    db, source_citation, list(unique_citations.values())
                )
            )
            stats["relationships_classified"] = classified
        except Exception as e:
            logger.warning(f"Relationship classification failed (non-fatal): {e}")

    db.commit()
    logger.info(f"Citation graph indexing complete: {stats}")
    return stats


def _extract_context(text: str, span: Optional[tuple], window: int = 150) -> str:
    """Extract surrounding context around a citation span."""
    if not span or len(span) < 2:
        return ""
    start = max(0, span[0] - window)
    end = min(len(text), span[1] + window)
    return text[start:end].strip()


def _infer_source_citation(document_name: str, chunks: List[Dict]) -> Optional[str]:
    """Try to infer the primary citation of a document from its name and content.

    Heuristic: look for a reporter citation in the first 2 chunks or the filename.
    Returns None if the document does not appear to be a case.
    """
    import re

    # Check filename for patterns like "347_US_483.pdf"
    name_clean = document_name.replace("_", " ").replace("-", " ")
    us_pattern = r'(\d+\s+(?:U\.?S\.?|F\.?(?:2d|3d|4th)?|S\.?Ct\.?)\s+\d+)'
    match = re.search(us_pattern, name_clean, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Check first 2 chunks for a prominent citation (often in the header)
    for chunk in chunks[:2]:
        content = chunk.get("content", "")[:500]
        match = re.search(us_pattern, content)
        if match:
            return match.group(1).strip()

    # UK/AU/CA/SG neutral citation patterns
    neutral_pattern = r'\[\d{4}\]\s+(?:UKSC|EWCA|EWHC|HCA|SCC|SGCA|SGHC)\s+\d+'
    for chunk in chunks[:2]:
        content = chunk.get("content", "")[:500]
        match = re.search(neutral_pattern, content)
        if match:
            return match.group(0).strip()

    return None


def _infer_case_name(document_name: str) -> Optional[str]:
    """Infer case name from document filename.

    Examples:
        "Brown v Board of Education.pdf" -> "Brown v. Board of Education"
        "123_F3d_456.pdf" -> None
    """
    import re
    name = document_name.rsplit(".", 1)[0]  # Strip extension
    # Look for "X v Y" or "X v. Y" patterns
    if re.search(r'\bv\.?\s', name, re.IGNORECASE):
        return name.replace("_", " ").strip()
    return None


def _classify_citation_type(citation: str, jurisdiction: str) -> str:
    """Classify a citation as Case or Statute based on its format.

    Returns "Case" or "Statute".
    """
    import re

    # Statute indicators
    statute_patterns = [
        r'U\.S\.C\.', r'C\.F\.R\.', r'Stat\.', r'Act\b',
        r'Art\.\s*\d', r'ss\s*\d', r'Section\s+\d',
        r'Directive\s+\d', r'Regulation\s+\d',
    ]
    for pattern in statute_patterns:
        if re.search(pattern, citation, re.IGNORECASE):
            return "Statute"

    # Default: treat as Case
    return "Case"


async def _classify_relationships_batch(
    db: Session,
    source_citation: str,
    citations: List[Dict],
    batch_size: int = 20,
) -> int:
    """Use Gemini to classify citation relationships beyond simple CITES.

    For each citation, the LLM determines if the source:
    - merely CITES the target (default)
    - OVERRULES the target
    - DISTINGUISHES the target
    - FOLLOWS the target
    - REVERSES the target
    - APPLIES the target (for statutes)

    Processes in batches to stay within token limits.

    Returns:
        Number of relationships upgraded from CITES to a more specific type.
    """
    import json
    import google.generativeai as genai

    try:
        from backend.config import get_settings
    except ImportError:
        try:
            from config import get_settings
        except ImportError:
            from ..config import get_settings

    settings = get_settings()
    if not settings.google_api_key:
        return 0

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(model_name=settings.gemini_model)

    classified_count = 0

    for i in range(0, len(citations), batch_size):
        batch = citations[i:i + batch_size]

        citation_list = []
        for idx, cite in enumerate(batch):
            citation_list.append({
                "index": idx,
                "citation": cite["raw_text"],
                "context": cite.get("context", "")[:300],
            })

        prompt = f"""You are a legal citation analyst. Given a source case and a list of cited authorities with their surrounding context, classify each citation relationship.

Source case: {source_citation}

Cited authorities:
{json.dumps(citation_list, indent=2)}

For each cited authority, determine the relationship type. Choose exactly ONE:
- CITES: simple reference/citation (default if unclear)
- OVERRULES: source explicitly overrules the cited case
- DISTINGUISHES: source distinguishes the cited case on facts or law
- FOLLOWS: source approvingly follows/applies the cited case as authority
- REVERSES: source reverses the cited case (appellate reversal)
- APPLIES: source applies/interprets a statute (only for statutes)

Respond with a JSON array. Each element: {{"index": N, "relationship": "TYPE", "confidence": 0.0-1.0}}
Only upgrade from CITES if confidence >= 0.7. Respond ONLY with the JSON array."""

        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=2000,
                ),
            )
            response_text = response.text.strip()

            # Strip markdown code blocks
            import re
            if response_text.startswith("```"):
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)

            parsed = json.loads(response_text)
            if not isinstance(parsed, list):
                continue

            for item in parsed:
                idx = item.get("index")
                relationship = item.get("relationship", "CITES").upper()
                confidence = item.get("confidence", 0.0)

                if idx is None or idx >= len(batch):
                    continue
                if relationship == "CITES" or confidence < 0.7:
                    continue  # Keep the existing CITES edge as-is

                cite = batch[idx]
                target_citation = cite["raw_text"].strip()
                target_label = _classify_citation_type(
                    target_citation, cite.get("jurisdiction", "US")
                )

                # Create the more specific edge (in addition to CITES)
                try:
                    create_edge(
                        db,
                        source_citation=source_citation,
                        source_label="Case",
                        target_citation=target_citation,
                        target_label=target_label,
                        edge_type=relationship,
                        properties={
                            "context": cite.get("context", "")[:200],
                            "confidence": str(confidence),
                        },
                    )
                    classified_count += 1

                    # If OVERRULES, update the target case status
                    if relationship == "OVERRULES":
                        _update_case_status(db, target_citation, "bad_law")
                    elif relationship == "REVERSES":
                        _update_case_status(db, target_citation, "caution")

                except Exception as e:
                    logger.warning(f"Failed to create {relationship} edge: {e}")

        except Exception as e:
            logger.warning(f"Relationship classification batch failed: {e}")

    return classified_count


def _update_case_status(db: Session, citation: str, new_status: str):
    """Update the status field on a Case node."""
    def esc(v):
        return "'" + str(v).replace("'", "\\'") + "'"

    now = datetime.now(timezone.utc).isoformat()
    query = f"""
        MATCH (c:Case {{citation: {esc(citation)}}})
        SET c.status = {esc(new_status)}, c.updated_at = {esc(now)}
        RETURN c
    """
    _cypher(db, query)


# ---------------------------------------------------------------------------
# Graph Statistics
# ---------------------------------------------------------------------------

def get_graph_stats(db: Session) -> Dict:
    """Return summary statistics about the citation graph."""
    stats = {}

    try:
        # Count nodes by label
        for label in ["Case", "Statute", "Court"]:
            result = _cypher(db, f"MATCH (n:{label}) RETURN count(n)")
            stats[f"{label.lower()}_count"] = result[0] if result else 0

        # Count edges by type
        for edge_type in ["CITES", "OVERRULES", "DISTINGUISHES", "FOLLOWS", "REVERSES", "APPLIES"]:
            result = _cypher(db, f"MATCH ()-[r:{edge_type}]->() RETURN count(r)")
            stats[f"{edge_type.lower()}_count"] = result[0] if result else 0

        # Total edges
        result = _cypher(db, "MATCH ()-[r]->() RETURN count(r)")
        stats["total_edges"] = result[0] if result else 0

    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        stats["error"] = str(e)

    return stats
```

### 5.4 `backend/services/graph_queries.py` -- Query Library

```python
"""Read-only Cypher query library for the legal citation knowledge graph.

All functions accept a SQLAlchemy Session (with AGE initialized) and return
Python dicts. No mutations -- this module is safe to call from any read path.
"""
import logging
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

try:
    from backend.services.citation_graph import _cypher
except ImportError:
    try:
        from services.citation_graph import _cypher
    except ImportError:
        from .citation_graph import _cypher


# ---------------------------------------------------------------------------
# Good Law Analysis
# ---------------------------------------------------------------------------

def is_good_law(db: Session, citation: str) -> Dict:
    """Check whether a case is still good law.

    Queries:
    1. Has the case been overruled?
    2. Has the case been reversed?
    3. Has the case been distinguished (weakened but not overruled)?

    Returns:
        {
            "citation": str,
            "status": "good_law" | "bad_law" | "caution" | "unknown",
            "is_overruled": bool,
            "overruled_by": [{"citation": str, "date": str}],
            "is_reversed": bool,
            "reversed_by": [{"citation": str, "date": str}],
            "distinguished_count": int,
            "distinguished_by": [{"citation": str}],
            "followed_count": int,
            "total_citations": int,
        }
    """
    def esc(v):
        return "'" + str(v).replace("'", "\\'") + "'"

    result = {
        "citation": citation,
        "status": "unknown",
        "is_overruled": False,
        "overruled_by": [],
        "is_reversed": False,
        "reversed_by": [],
        "distinguished_count": 0,
        "distinguished_by": [],
        "followed_count": 0,
        "total_citations": 0,
    }

    try:
        # Check for overruling cases
        overrulers = _cypher(db, f"""
            MATCH (overruler:Case)-[:OVERRULES]->(target:Case {{citation: {esc(citation)}}})
            RETURN overruler.citation, overruler.date
        """)
        if overrulers:
            result["is_overruled"] = True
            result["overruled_by"] = [
                {"citation": str(o.get("overruler.citation", "")),
                 "date": str(o.get("overruler.date", ""))}
                for o in overrulers if isinstance(o, dict)
            ]

        # Check for reversals
        reversers = _cypher(db, f"""
            MATCH (reverser:Case)-[:REVERSES]->(target:Case {{citation: {esc(citation)}}})
            RETURN reverser.citation, reverser.date
        """)
        if reversers:
            result["is_reversed"] = True
            result["reversed_by"] = [
                {"citation": str(r.get("reverser.citation", "")),
                 "date": str(r.get("reverser.date", ""))}
                for r in reversers if isinstance(r, dict)
            ]

        # Count distinguishing cases
        distinguished = _cypher(db, f"""
            MATCH (d:Case)-[:DISTINGUISHES]->(target:Case {{citation: {esc(citation)}}})
            RETURN d.citation
            LIMIT 10
        """)
        result["distinguished_count"] = len(distinguished) if distinguished else 0
        result["distinguished_by"] = [
            {"citation": str(d.get("d.citation", ""))}
            for d in (distinguished or []) if isinstance(d, dict)
        ]

        # Count cases that follow this authority
        followed = _cypher(db, f"""
            MATCH (f:Case)-[:FOLLOWS]->(target:Case {{citation: {esc(citation)}}})
            RETURN count(f)
        """)
        result["followed_count"] = followed[0] if followed else 0

        # Total incoming citations
        total = _cypher(db, f"""
            MATCH (c:Case)-[:CITES]->(target:Case {{citation: {esc(citation)}}})
            RETURN count(c)
        """)
        result["total_citations"] = total[0] if total else 0

        # Determine status
        if result["is_overruled"]:
            result["status"] = "bad_law"
        elif result["is_reversed"]:
            result["status"] = "caution"
        elif result["distinguished_count"] > 3:
            result["status"] = "caution"
        elif result["followed_count"] > 0 or result["total_citations"] > 0:
            result["status"] = "good_law"
        else:
            result["status"] = "unknown"

    except Exception as e:
        logger.error(f"Good law check failed for '{citation}': {e}")
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Precedent Chain
# ---------------------------------------------------------------------------

def get_precedent_chain(
    db: Session,
    citation: str,
    max_hops: int = 3,
    limit: int = 50,
) -> Dict:
    """Trace the precedent chain from a case through CITES and FOLLOWS edges.

    Returns a list of authorities reachable within `max_hops` hops, ordered
    by hop distance (nearest authorities first).

    Args:
        db: Database session.
        citation: Starting case citation.
        max_hops: Maximum traversal depth (1-5). Default 3.
        limit: Max results. Default 50.

    Returns:
        {
            "citation": str,
            "chain": [
                {"citation": str, "case_name": str, "hops": int, "path": [str]}
            ],
            "total": int,
        }
    """
    max_hops = min(max(max_hops, 1), 5)  # Clamp to 1..5

    def esc(v):
        return "'" + str(v).replace("'", "\\'") + "'"

    result = {"citation": citation, "chain": [], "total": 0}

    try:
        # Variable-length path query
        chain = _cypher(db, f"""
            MATCH path = (source:Case {{citation: {esc(citation)}}})-[:CITES|FOLLOWS*1..{max_hops}]->(authority:Case)
            WITH authority, min(length(path)) AS hops, collect(DISTINCT path) AS paths
            RETURN authority.citation, authority.case_name, authority.jurisdiction,
                   authority.status, hops
            ORDER BY hops ASC, authority.citation ASC
            LIMIT {limit}
        """)

        if chain:
            for item in chain:
                if isinstance(item, dict):
                    result["chain"].append({
                        "citation": str(item.get("authority.citation", "")),
                        "case_name": item.get("authority.case_name"),
                        "jurisdiction": item.get("authority.jurisdiction"),
                        "status": item.get("authority.status", "unknown"),
                        "hops": item.get("hops", 0),
                    })

        result["total"] = len(result["chain"])

    except Exception as e:
        logger.error(f"Precedent chain failed for '{citation}': {e}")
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Similar Cases (Shared Authority)
# ---------------------------------------------------------------------------

def find_similar_cases(
    db: Session,
    citation: str,
    limit: int = 10,
) -> Dict:
    """Find cases that cite the same authorities as the given case.

    The similarity metric is the count of shared cited authorities.
    Cases that share more cited authorities are ranked higher.

    Args:
        db: Database session.
        citation: Case citation to find similar cases for.
        limit: Max results. Default 10.

    Returns:
        {
            "citation": str,
            "similar": [
                {
                    "citation": str,
                    "case_name": str,
                    "shared_authorities": int,
                    "shared_list": [str],
                }
            ],
        }
    """
    def esc(v):
        return "'" + str(v).replace("'", "\\'") + "'"

    result = {"citation": citation, "similar": []}

    try:
        similar = _cypher(db, f"""
            MATCH (target:Case {{citation: {esc(citation)}}})-[:CITES]->(shared:Case)<-[:CITES]-(similar:Case)
            WHERE similar <> target
            WITH similar, collect(DISTINCT shared.citation) AS shared_cites
            RETURN similar.citation, similar.case_name, similar.jurisdiction,
                   size(shared_cites) AS shared_count, shared_cites
            ORDER BY shared_count DESC
            LIMIT {limit}
        """)

        if similar:
            for item in similar:
                if isinstance(item, dict):
                    result["similar"].append({
                        "citation": str(item.get("similar.citation", "")),
                        "case_name": item.get("similar.case_name"),
                        "jurisdiction": item.get("similar.jurisdiction"),
                        "shared_authorities": item.get("shared_count", 0),
                        "shared_list": item.get("shared_cites", []),
                    })

    except Exception as e:
        logger.error(f"Similar cases query failed for '{citation}': {e}")
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Citation Network (for visualization)
# ---------------------------------------------------------------------------

def get_citation_network(
    db: Session,
    citation: str,
    depth: int = 2,
    max_nodes: int = 100,
) -> Dict:
    """Get the local citation network around a case for D3.js visualization.

    Returns nodes and edges in a format suitable for force-directed graph rendering.

    Args:
        db: Database session.
        citation: Center case citation.
        depth: Traversal depth in both directions. Default 2.
        max_nodes: Maximum nodes to return. Default 100.

    Returns:
        {
            "nodes": [
                {"id": str, "label": str, "type": "Case"|"Statute", "status": str, ...}
            ],
            "edges": [
                {"source": str, "target": str, "type": str, "properties": {...}}
            ],
            "center": str,
        }
    """
    depth = min(max(depth, 1), 3)  # Clamp to 1..3

    def esc(v):
        return "'" + str(v).replace("'", "\\'") + "'"

    result = {"nodes": [], "edges": [], "center": citation}

    try:
        # Get outgoing citations (what this case cites)
        outgoing = _cypher(db, f"""
            MATCH (source:Case {{citation: {esc(citation)}}})-[r]->(target)
            WHERE type(r) IN ['CITES', 'OVERRULES', 'DISTINGUISHES', 'FOLLOWS', 'REVERSES', 'APPLIES']
            RETURN source.citation AS src, type(r) AS rel_type,
                   target.citation AS tgt, labels(target)[0] AS tgt_label,
                   target.case_name AS tgt_name, target.status AS tgt_status
            LIMIT {max_nodes}
        """)

        # Get incoming citations (what cites this case)
        incoming = _cypher(db, f"""
            MATCH (source)-[r]->(target:Case {{citation: {esc(citation)}}})
            WHERE type(r) IN ['CITES', 'OVERRULES', 'DISTINGUISHES', 'FOLLOWS', 'REVERSES', 'APPLIES']
            RETURN source.citation AS src, type(r) AS rel_type,
                   target.citation AS tgt, labels(source)[0] AS src_label,
                   source.case_name AS src_name, source.status AS src_status
            LIMIT {max_nodes}
        """)

        # Build node and edge sets
        node_map = {}

        # Center node
        center_result = _cypher(db, f"""
            MATCH (c:Case {{citation: {esc(citation)}}})
            RETURN c.citation, c.case_name, c.status, c.jurisdiction
        """)
        if center_result and isinstance(center_result[0], dict):
            c = center_result[0]
            node_map[citation] = {
                "id": citation,
                "label": c.get("c.case_name") or citation,
                "type": "Case",
                "status": c.get("c.status", "unknown"),
                "jurisdiction": c.get("c.jurisdiction"),
                "is_center": True,
            }

        # Process outgoing
        for item in (outgoing or []):
            if not isinstance(item, dict):
                continue
            tgt = str(item.get("tgt", ""))
            if tgt and tgt not in node_map:
                node_map[tgt] = {
                    "id": tgt,
                    "label": item.get("tgt_name") or tgt,
                    "type": item.get("tgt_label", "Case"),
                    "status": item.get("tgt_status", "unknown"),
                    "is_center": False,
                }
            result["edges"].append({
                "source": citation,
                "target": tgt,
                "type": item.get("rel_type", "CITES"),
            })

        # Process incoming
        for item in (incoming or []):
            if not isinstance(item, dict):
                continue
            src = str(item.get("src", ""))
            if src and src not in node_map:
                node_map[src] = {
                    "id": src,
                    "label": item.get("src_name") or src,
                    "type": item.get("src_label", "Case"),
                    "status": item.get("src_status", "unknown"),
                    "is_center": False,
                }
            result["edges"].append({
                "source": src,
                "target": citation,
                "type": item.get("rel_type", "CITES"),
            })

        # Second hop (depth=2): for each neighbor, get their connections
        if depth >= 2:
            neighbor_citations = [n for n in node_map if n != citation]
            for neighbor in neighbor_citations[:20]:  # Cap second-hop expansion
                second_hop = _cypher(db, f"""
                    MATCH (a {{citation: {esc(neighbor)}}})-[r]->(b)
                    WHERE type(r) IN ['CITES', 'OVERRULES', 'FOLLOWS']
                      AND b.citation <> {esc(citation)}
                    RETURN a.citation AS src, type(r) AS rel_type,
                           b.citation AS tgt, labels(b)[0] AS tgt_label,
                           b.case_name AS tgt_name, b.status AS tgt_status
                    LIMIT 5
                """)
                for item in (second_hop or []):
                    if not isinstance(item, dict):
                        continue
                    tgt = str(item.get("tgt", ""))
                    if tgt and tgt not in node_map and len(node_map) < max_nodes:
                        node_map[tgt] = {
                            "id": tgt,
                            "label": item.get("tgt_name") or tgt,
                            "type": item.get("tgt_label", "Case"),
                            "status": item.get("tgt_status", "unknown"),
                            "is_center": False,
                        }
                    if tgt:
                        result["edges"].append({
                            "source": neighbor,
                            "target": tgt,
                            "type": item.get("rel_type", "CITES"),
                        })

        result["nodes"] = list(node_map.values())

    except Exception as e:
        logger.error(f"Citation network query failed for '{citation}': {e}")
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# What Overrules This?
# ---------------------------------------------------------------------------

def what_overrules(db: Session, citation: str) -> Dict:
    """Find all cases that overrule or reverse a given case, with full chain.

    Goes beyond direct overruling to find transitive overrulings:
    if A overrules B and C overrules A, C effectively overrules B.

    Returns:
        {
            "citation": str,
            "direct_overrulings": [...],
            "direct_reversals": [...],
            "transitive_overrulings": [...],
        }
    """
    def esc(v):
        return "'" + str(v).replace("'", "\\'") + "'"

    result = {
        "citation": citation,
        "direct_overrulings": [],
        "direct_reversals": [],
        "transitive_overrulings": [],
    }

    try:
        # Direct overrulings
        direct = _cypher(db, f"""
            MATCH (overruler:Case)-[:OVERRULES]->(target:Case {{citation: {esc(citation)}}})
            RETURN overruler.citation, overruler.case_name, overruler.date,
                   overruler.court, overruler.jurisdiction
        """)
        for item in (direct or []):
            if isinstance(item, dict):
                result["direct_overrulings"].append({
                    "citation": str(item.get("overruler.citation", "")),
                    "case_name": item.get("overruler.case_name"),
                    "date": item.get("overruler.date"),
                    "court": item.get("overruler.court"),
                })

        # Direct reversals
        reversals = _cypher(db, f"""
            MATCH (reverser:Case)-[:REVERSES]->(target:Case {{citation: {esc(citation)}}})
            RETURN reverser.citation, reverser.case_name, reverser.date,
                   reverser.court, reverser.jurisdiction
        """)
        for item in (reversals or []):
            if isinstance(item, dict):
                result["direct_reversals"].append({
                    "citation": str(item.get("reverser.citation", "")),
                    "case_name": item.get("reverser.case_name"),
                    "date": item.get("reverser.date"),
                    "court": item.get("reverser.court"),
                })

        # Transitive: cases that overrule cases that follow this one
        transitive = _cypher(db, f"""
            MATCH (overruler:Case)-[:OVERRULES]->(follower:Case)-[:FOLLOWS]->(target:Case {{citation: {esc(citation)}}})
            RETURN overruler.citation, overruler.case_name, follower.citation AS via_citation
        """)
        for item in (transitive or []):
            if isinstance(item, dict):
                result["transitive_overrulings"].append({
                    "citation": str(item.get("overruler.citation", "")),
                    "case_name": item.get("overruler.case_name"),
                    "via": str(item.get("via_citation", "")),
                })

    except Exception as e:
        logger.error(f"What-overrules query failed for '{citation}': {e}")
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Case Detail
# ---------------------------------------------------------------------------

def get_case_detail(db: Session, citation: str) -> Dict:
    """Get full details for a case node including all relationships.

    Returns:
        {
            "citation": str,
            "case_name": str,
            ...node properties...,
            "cites": [...],
            "cited_by": [...],
            "overruled_by": [...],
            "followed_by": [...],
        }
    """
    def esc(v):
        return "'" + str(v).replace("'", "\\'") + "'"

    result = {"citation": citation, "found": False}

    try:
        # Node properties
        node = _cypher(db, f"""
            MATCH (c:Case {{citation: {esc(citation)}}})
            RETURN c
        """)
        if not node:
            return result

        if isinstance(node[0], dict):
            result.update(node[0])
        result["found"] = True

        # Outgoing CITES
        cites = _cypher(db, f"""
            MATCH (c:Case {{citation: {esc(citation)}}})-[:CITES]->(target)
            RETURN target.citation, labels(target)[0] AS label
            LIMIT 50
        """)
        result["cites"] = [
            {"citation": str(i.get("target.citation", "")), "type": i.get("label", "Case")}
            for i in (cites or []) if isinstance(i, dict)
        ]

        # Incoming CITES
        cited_by = _cypher(db, f"""
            MATCH (source:Case)-[:CITES]->(c:Case {{citation: {esc(citation)}}})
            RETURN source.citation, source.case_name
            LIMIT 50
        """)
        result["cited_by"] = [
            {"citation": str(i.get("source.citation", "")),
             "case_name": i.get("source.case_name")}
            for i in (cited_by or []) if isinstance(i, dict)
        ]

        # Incoming OVERRULES
        result["overruled_by"] = [
            {"citation": str(i.get("overruler.citation", "")),
             "case_name": i.get("overruler.case_name")}
            for i in (_cypher(db, f"""
                MATCH (overruler:Case)-[:OVERRULES]->(c:Case {{citation: {esc(citation)}}})
                RETURN overruler.citation, overruler.case_name
            """) or []) if isinstance(i, dict)
        ]

        # Incoming FOLLOWS
        result["followed_by"] = [
            {"citation": str(i.get("f.citation", "")),
             "case_name": i.get("f.case_name")}
            for i in (_cypher(db, f"""
                MATCH (f:Case)-[:FOLLOWS]->(c:Case {{citation: {esc(citation)}}})
                RETURN f.citation, f.case_name
            """) or []) if isinstance(i, dict)
        ]

    except Exception as e:
        logger.error(f"Case detail query failed for '{citation}': {e}")
        result["error"] = str(e)

    return result
```

---

## 6. Citation Extraction Pipeline

### 6.1 Integration Point: `tasks.py`

The citation graph extraction hooks into the existing document processing pipeline immediately after the embedding/indexing step (step 7) and before the status update (step 8).

**Diff for `backend/tasks.py`:**

```python
# At top of file, add import:
try:
    from backend.services.citation_graph import extract_and_index_citations
except ImportError:
    try:
        from services.citation_graph import extract_and_index_citations
    except ImportError:
        extract_and_index_citations = None

# ... existing code ...

# After step 7 (vector indexing), before step 8 (status update), add:

        # 7b. Extract citations and build graph (best-effort, non-blocking)
        if extract_and_index_citations and settings.citation_graph_enabled:
            try:
                logger.info(f"[Task {self.request.id}] Extracting citations for graph")
                # Re-read chunks from DB since we deleted the local list
                db_chunks = db.query(Chunk).filter(
                    Chunk.document_id == UUID(document_id)
                ).order_by(Chunk.chunk_sequence).all()
                chunk_dicts = [
                    {"content": c.content, "page_num": c.page_num or ""}
                    for c in db_chunks
                ]
                graph_stats = extract_and_index_citations(
                    db=db,
                    document_id=document_id,
                    matter_id=matter_id,
                    document_name=document.name,
                    chunks=chunk_dicts,
                    jurisdiction=document.jurisdiction or "US",
                )
                log_activity(
                    db, matter_id, "citation_graph_updated",
                    details=f"Graph: {graph_stats['nodes_created']} nodes, "
                            f"{graph_stats['edges_created']} edges for {document.name}"
                )
            except Exception as e:
                # Non-fatal: graph indexing failure should not fail document processing
                logger.warning(f"[Task {self.request.id}] Citation graph indexing failed (non-fatal): {e}")
```

### 6.2 Extraction Strategy by Jurisdiction

| Jurisdiction | Primary Extractor | Secondary Validator | LLM Enrichment |
|-------------|-------------------|--------------------|-----------------|
| US | `eyecite` (full parser) | regex patterns | Gemini for relationship classification |
| UK | regex (`[YYYY] UKSC N`) | -- | Gemini for case name extraction |
| EU | regex (`Case C-NNN/NN`) | -- | Gemini for case name extraction |
| IN | regex (`AIR YYYY SC N`) | -- | Gemini for case name extraction |
| AU | regex (`[YYYY] HCA N`) | -- | Gemini for case name extraction |
| CA | regex (`YYYY SCC N`) | -- | Gemini for case name extraction |
| SG | regex (`[YYYY] SGCA N`) | -- | Gemini for case name extraction |

The existing `backend/services/citation_extractor.py` already implements all regex patterns and the LLM extraction pipeline. The graph module reuses `extract_citations_regex()` directly and only adds the graph indexing layer on top.

### 6.3 Citation Normalization

Before merging into the graph, citations are normalized to prevent duplicates:

```python
def normalize_citation(raw: str) -> str:
    """Normalize citation to canonical form for deduplication.

    Examples:
        "347 U.S. 483 (1954)" -> "347 U.S. 483"
        "347 U. S. 483"       -> "347 U.S. 483"
        "[2024]  UKSC  1"     -> "[2024] UKSC 1"
    """
    import re
    s = raw.strip()
    # Remove year parentheticals: "(1954)", "(2024)"
    s = re.sub(r'\s*\(\d{4}\)\s*$', '', s)
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s)
    # Normalize "U. S." -> "U.S."
    s = re.sub(r'U\.\s+S\.', 'U.S.', s)
    return s.strip()
```

---

## 7. LLM Relationship Classification

### 7.1 Prompt Design

The relationship classifier uses Gemini with `temperature=0.0` to deterministically classify each citation relationship. The prompt (see `_classify_relationships_batch()` in Section 5.3) follows these principles:

1. **Batch processing**: Up to 20 citations per LLM call to minimize API round-trips.
2. **Context-grounded**: Each citation includes the surrounding 300 characters from the source document, giving the LLM enough context to determine the relationship.
3. **Conservative defaults**: The LLM must express `confidence >= 0.7` to upgrade a relationship from CITES. Below that threshold, CITES is retained.
4. **Structured output**: JSON array response with `index`, `relationship`, and `confidence`.

### 7.2 Cost Analysis

| Component | Cost per Document | Cost per 1,000 Documents |
|-----------|-------------------|--------------------------|
| Citation regex extraction | $0.00 | $0.00 |
| eyecite US validation | $0.00 | $0.00 |
| Relationship classification (Gemini) | ~$0.003 (avg 20 citations, 1 batch) | ~$3.00 |
| Graph MERGE operations | $0.00 (local PostgreSQL) | $0.00 |
| **Total** | **~$0.003** | **~$3.00** |

### 7.3 Failure Handling

Relationship classification is best-effort. If the LLM call fails:

- All CITES edges remain (they were created before classification).
- A warning is logged.
- Document processing continues to completion.
- Classification can be retried later via an admin endpoint.

---

## 8. Graph Query Library

The query library in `graph_queries.py` (Section 5.4) exposes five primary query functions:

### 8.1 Query Summary

| Function | Input | Output | Cypher Pattern | Target Latency |
|----------|-------|--------|----------------|----------------|
| `is_good_law(citation)` | Citation string | Status + overrulers + reversals | `MATCH (x)-[:OVERRULES\|REVERSES]->(target)` | <50ms |
| `get_precedent_chain(citation, max_hops)` | Citation + depth | Authority chain with hop counts | `MATCH path = (s)-[:CITES\|FOLLOWS*1..N]->(a)` | <200ms (3 hops) |
| `find_similar_cases(citation)` | Citation string | Cases sharing cited authorities | `MATCH (t)-[:CITES]->(shared)<-[:CITES]-(similar)` | <300ms |
| `get_citation_network(citation, depth)` | Citation + depth | Nodes + edges for D3.js | Multi-match, bidirectional | <300ms |
| `what_overrules(citation)` | Citation string | Direct + transitive overrulings | `MATCH (x)-[:OVERRULES]->(target)` + transitive | <100ms |

### 8.2 Cypher Query Examples

**Is case still good law?**

```cypher
-- Direct overruling check
MATCH (overruler:Case)-[:OVERRULES]->(target:Case {citation: '123 F.3d 456'})
RETURN overruler.citation, overruler.date

-- Count followers (positive signal)
MATCH (f:Case)-[:FOLLOWS]->(target:Case {citation: '123 F.3d 456'})
RETURN count(f)
```

**Precedent chain (3 hops):**

```cypher
MATCH path = (source:Case {citation: '123 F.3d 456'})-[:CITES|FOLLOWS*1..3]->(authority:Case)
WITH authority, min(length(path)) AS hops
RETURN authority.citation, authority.case_name, hops
ORDER BY hops ASC
LIMIT 50
```

**Similar cases (shared authorities):**

```cypher
MATCH (target:Case {citation: '123 F.3d 456'})-[:CITES]->(shared:Case)<-[:CITES]-(similar:Case)
WHERE similar <> target
WITH similar, collect(DISTINCT shared.citation) AS shared_cites
RETURN similar.citation, similar.case_name, size(shared_cites) AS shared_count
ORDER BY shared_count DESC
LIMIT 10
```

---

## 9. Graph-Enhanced RAG Retrieval

### 9.1 Integration into `rag_engine.py`

When a user asks a question that references a citation (detected via regex or eyecite), the RAG engine fetches graph context and injects it into the Gemini prompt alongside the vector-retrieved chunks.

**Diff for `backend/services/rag_engine.py`:**

```python
# New import at top:
try:
    from backend.services.graph_queries import is_good_law, get_precedent_chain, find_similar_cases
    from backend.config import get_settings
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

# New function:
def _build_graph_context(db: Session, query: str) -> Optional[str]:
    """Extract citations from the user query and fetch graph context.

    Returns a formatted string to prepend to the RAG context, or None.
    """
    if not GRAPH_AVAILABLE:
        return None

    settings = get_settings()
    if not settings.citation_graph_enabled:
        return None

    from backend.services.citation_extractor import extract_citations_regex

    citations = extract_citations_regex(query)
    if not citations:
        return None

    parts = ["CITATION GRAPH CONTEXT:\n"]

    for cite in citations[:3]:  # Max 3 citations to keep context small
        raw = cite["raw_text"]

        # Good law check
        law_status = is_good_law(db, raw)
        if law_status.get("status") != "unknown":
            parts.append(f"\n--- {raw} ---")
            parts.append(f"Status: {law_status['status'].upper()}")
            if law_status.get("is_overruled"):
                overrulers = ", ".join(
                    o["citation"] for o in law_status.get("overruled_by", [])
                )
                parts.append(f"Overruled by: {overrulers}")
            if law_status.get("followed_count", 0) > 0:
                parts.append(f"Followed by {law_status['followed_count']} cases")
            parts.append(f"Cited by {law_status.get('total_citations', 0)} cases")

        # Precedent chain (1 hop only for context)
        chain = get_precedent_chain(db, raw, max_hops=1, limit=5)
        if chain.get("chain"):
            authorities = ", ".join(
                c["citation"] for c in chain["chain"][:5]
            )
            parts.append(f"Key authorities cited: {authorities}")

    context = "\n".join(parts)
    if len(parts) <= 1:
        return None  # No useful graph context found

    return context + "\n\n"

# In query_matter(), after vector retrieval and before calling Gemini:
#
#   graph_context = _build_graph_context(db, question)
#   if graph_context:
#       context = graph_context + context
```

### 9.2 Graph Context Budget

Graph context is limited to prevent crowding out document chunks:

- Maximum 3 citations looked up per query.
- Maximum 5 authorities per precedent chain.
- Graph context string capped at 2,000 characters.
- Graph context comes before document excerpts in the prompt (lower priority for Gemini's attention).

### 9.3 Citation-Aware Query Detection

Not every query benefits from graph context. The system checks:

1. Does the query contain a recognizable legal citation? (regex check)
2. Does the query contain keywords like "good law", "overruled", "precedent", "authority", "still valid"?
3. If neither, skip graph context entirely (zero latency cost).

---

## 10. REST API Endpoints

### 10.1 New Routes

Add to `backend/main.py` (or a new `backend/routers/graph.py`):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/graph/stats` | Graph-wide statistics |
| `GET` | `/graph/case/{citation}` | Full case detail with relationships |
| `GET` | `/graph/case/{citation}/good-law` | Good law status check |
| `GET` | `/graph/case/{citation}/chain` | Precedent chain traversal |
| `GET` | `/graph/case/{citation}/similar` | Similar cases by shared authorities |
| `GET` | `/graph/case/{citation}/network` | Citation network for visualization |
| `GET` | `/graph/case/{citation}/overruled-by` | What overrules this case |
| `GET` | `/matters/{id}/graph` | Citation graph for all documents in a matter |

### 10.2 Schema Definitions

Add to `backend/schemas.py`:

```python
# ============================================
# CITATION GRAPH SCHEMAS
# ============================================

class GraphStatsResponse(BaseModel):
    """Citation graph statistics"""
    case_count: int = 0
    statute_count: int = 0
    court_count: int = 0
    cites_count: int = 0
    overrules_count: int = 0
    distinguishes_count: int = 0
    follows_count: int = 0
    reverses_count: int = 0
    applies_count: int = 0
    total_edges: int = 0


class GoodLawResponse(BaseModel):
    """Good law check result"""
    citation: str
    status: str  # good_law, bad_law, caution, unknown
    is_overruled: bool = False
    overruled_by: list[dict] = []
    is_reversed: bool = False
    reversed_by: list[dict] = []
    distinguished_count: int = 0
    followed_count: int = 0
    total_citations: int = 0


class PrecedentChainItem(BaseModel):
    """Single item in a precedent chain"""
    citation: str
    case_name: Optional[str] = None
    jurisdiction: Optional[str] = None
    status: str = "unknown"
    hops: int = 0


class PrecedentChainResponse(BaseModel):
    """Precedent chain result"""
    citation: str
    chain: list[PrecedentChainItem] = []
    total: int = 0


class SimilarCaseItem(BaseModel):
    """Single similar case"""
    citation: str
    case_name: Optional[str] = None
    jurisdiction: Optional[str] = None
    shared_authorities: int = 0
    shared_list: list[str] = []


class SimilarCasesResponse(BaseModel):
    """Similar cases result"""
    citation: str
    similar: list[SimilarCaseItem] = []


class GraphNode(BaseModel):
    """Node in the citation network"""
    id: str
    label: str
    type: str  # Case, Statute, Court
    status: Optional[str] = None
    jurisdiction: Optional[str] = None
    is_center: bool = False


class GraphEdge(BaseModel):
    """Edge in the citation network"""
    source: str
    target: str
    type: str  # CITES, OVERRULES, DISTINGUISHES, FOLLOWS, REVERSES, APPLIES


class CitationNetworkResponse(BaseModel):
    """Citation network for visualization"""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    center: str
```

### 10.3 Route Implementation

```python
# backend/routers/graph.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from urllib.parse import unquote

from backend.database import get_db
from backend.config import get_settings
from backend.schemas import (
    GraphStatsResponse, GoodLawResponse, PrecedentChainResponse,
    SimilarCasesResponse, CitationNetworkResponse,
)
from backend.services.citation_graph import get_graph_stats
from backend.services.graph_queries import (
    is_good_law, get_precedent_chain, find_similar_cases,
    get_citation_network, what_overrules, get_case_detail,
)

router = APIRouter(prefix="/graph", tags=["Citation Graph"])
settings = get_settings()


def _require_graph():
    """Raise 503 if graph feature is disabled."""
    if not settings.citation_graph_enabled:
        raise HTTPException(
            status_code=503,
            detail="Citation graph feature is not enabled. "
                   "Set CITATION_GRAPH_ENABLED=true in .env"
        )


@router.get("/stats", response_model=GraphStatsResponse)
def graph_stats(db: Session = Depends(get_db)):
    _require_graph()
    return get_graph_stats(db)


@router.get("/case/{citation:path}/good-law", response_model=GoodLawResponse)
def check_good_law(citation: str, db: Session = Depends(get_db)):
    _require_graph()
    citation = unquote(citation)
    return is_good_law(db, citation)


@router.get("/case/{citation:path}/chain", response_model=PrecedentChainResponse)
def precedent_chain(
    citation: str,
    max_hops: int = Query(3, ge=1, le=5),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    _require_graph()
    citation = unquote(citation)
    return get_precedent_chain(db, citation, max_hops=max_hops, limit=limit)


@router.get("/case/{citation:path}/similar", response_model=SimilarCasesResponse)
def similar_cases(
    citation: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    _require_graph()
    citation = unquote(citation)
    return find_similar_cases(db, citation, limit=limit)


@router.get("/case/{citation:path}/network", response_model=CitationNetworkResponse)
def citation_network(
    citation: str,
    depth: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(100, ge=10, le=500),
    db: Session = Depends(get_db),
):
    _require_graph()
    citation = unquote(citation)
    return get_citation_network(db, citation, depth=depth, max_nodes=max_nodes)


@router.get("/case/{citation:path}/overruled-by")
def overruled_by(citation: str, db: Session = Depends(get_db)):
    _require_graph()
    citation = unquote(citation)
    return what_overrules(db, citation)


@router.get("/case/{citation:path}")
def case_detail(citation: str, db: Session = Depends(get_db)):
    _require_graph()
    citation = unquote(citation)
    result = get_case_detail(db, citation)
    if not result.get("found"):
        raise HTTPException(status_code=404, detail=f"Case '{citation}' not found in graph")
    return result
```

---

## 11. Frontend Citation Network Visualization

### 11.1 Dependencies

Add to `frontend/package.json`:

```json
{
  "dependencies": {
    "d3": "^7.9.0",
    "@types/d3": "^7.4.3"
  }
}
```

### 11.2 TypeScript Types

Add to `frontend/lib/types.ts`:

```typescript
// ============================================
// Citation Graph Types
// ============================================

export interface GraphNode {
  id: string
  label: string
  type: "Case" | "Statute" | "Court"
  status?: "good_law" | "caution" | "bad_law" | "unknown"
  jurisdiction?: string
  isCenter: boolean
  // D3 simulation properties (added at runtime)
  x?: number
  y?: number
  fx?: number | null
  fy?: number | null
}

export interface GraphEdge {
  source: string | GraphNode
  target: string | GraphNode
  type: "CITES" | "OVERRULES" | "DISTINGUISHES" | "FOLLOWS" | "REVERSES" | "APPLIES"
}

export interface CitationNetwork {
  nodes: GraphNode[]
  edges: GraphEdge[]
  center: string
}

export interface GoodLawResult {
  citation: string
  status: "good_law" | "bad_law" | "caution" | "unknown"
  isOverruled: boolean
  overruledBy: { citation: string; date?: string }[]
  isReversed: boolean
  reversedBy: { citation: string; date?: string }[]
  distinguishedCount: number
  followedCount: number
  totalCitations: number
}

export interface PrecedentChainItem {
  citation: string
  caseName?: string
  jurisdiction?: string
  status: string
  hops: number
}

export interface SimilarCaseItem {
  citation: string
  caseName?: string
  jurisdiction?: string
  sharedAuthorities: number
  sharedList: string[]
}
```

### 11.3 API Service Functions

Add to `frontend/lib/api-services.ts`:

```typescript
// ============================================
// Citation Graph API Functions
// ============================================

export async function getGraphStats(): Promise<Record<string, number>> {
  const { data } = await api.get("/graph/stats")
  return data
}

export async function checkGoodLaw(citation: string): Promise<import("./types").GoodLawResult> {
  const { data } = await api.get(`/graph/case/${encodeURIComponent(citation)}/good-law`)
  return data
}

export async function getPrecedentChain(
  citation: string,
  maxHops = 3
): Promise<{ citation: string; chain: import("./types").PrecedentChainItem[]; total: number }> {
  const { data } = await api.get(
    `/graph/case/${encodeURIComponent(citation)}/chain`,
    { params: { max_hops: maxHops } }
  )
  return data
}

export async function getSimilarCases(
  citation: string,
  limit = 10
): Promise<{ citation: string; similar: import("./types").SimilarCaseItem[] }> {
  const { data } = await api.get(
    `/graph/case/${encodeURIComponent(citation)}/similar`,
    { params: { limit } }
  )
  return data
}

export async function getCitationNetwork(
  citation: string,
  depth = 2,
  maxNodes = 100
): Promise<import("./types").CitationNetwork> {
  const { data } = await api.get(
    `/graph/case/${encodeURIComponent(citation)}/network`,
    { params: { depth, max_nodes: maxNodes } }
  )
  return data
}
```

### 11.4 `CitationGraph.tsx` Component

```tsx
// frontend/components/CitationGraph.tsx
"use client"

import React, { useRef, useEffect, useState, useCallback } from "react"
import * as d3 from "d3"
import type { GraphNode, GraphEdge, CitationNetwork } from "@/lib/types"
import { getCitationNetwork } from "@/lib/api-services"

// Color scheme for node statuses
const STATUS_COLORS: Record<string, string> = {
  good_law: "#22c55e",   // green-500
  caution: "#f59e0b",    // amber-500
  bad_law: "#ef4444",    // red-500
  unknown: "#6b7280",    // gray-500
}

// Color scheme for edge types
const EDGE_COLORS: Record<string, string> = {
  CITES: "#94a3b8",         // slate-400
  OVERRULES: "#ef4444",     // red-500
  DISTINGUISHES: "#f59e0b", // amber-500
  FOLLOWS: "#22c55e",       // green-500
  REVERSES: "#dc2626",      // red-600
  APPLIES: "#3b82f6",       // blue-500
}

interface CitationGraphProps {
  citation: string
  width?: number
  height?: number
  depth?: number
  onNodeClick?: (citation: string) => void
}

export default function CitationGraph({
  citation,
  width = 800,
  height = 600,
  depth = 2,
  onNodeClick,
}: CitationGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [network, setNetwork] = useState<CitationNetwork | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch network data
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    getCitationNetwork(citation, depth)
      .then((data) => {
        if (!cancelled) {
          setNetwork(data)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message || "Failed to load citation network")
          setLoading(false)
        }
      })

    return () => { cancelled = true }
  }, [citation, depth])

  // D3 force simulation
  useEffect(() => {
    if (!network || !svgRef.current) return

    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    const { nodes, edges } = network

    if (nodes.length === 0) return

    // Create simulation
    const simulation = d3.forceSimulation<GraphNode>(nodes)
      .force("link", d3.forceLink<GraphNode, GraphEdge>(edges)
        .id((d) => d.id)
        .distance(120))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(40))

    // Zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => {
        container.attr("transform", event.transform)
      })
    svg.call(zoom)

    const container = svg.append("g")

    // Arrow markers for directed edges
    const defs = svg.append("defs")
    Object.entries(EDGE_COLORS).forEach(([type, color]) => {
      defs.append("marker")
        .attr("id", `arrow-${type}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 25)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", color)
    })

    // Draw edges
    const link = container.append("g")
      .selectAll("line")
      .data(edges)
      .join("line")
      .attr("stroke", (d) => EDGE_COLORS[d.type as string] || "#94a3b8")
      .attr("stroke-width", (d) => d.type === "OVERRULES" || d.type === "REVERSES" ? 2.5 : 1.5)
      .attr("stroke-dasharray", (d) => d.type === "DISTINGUISHES" ? "5,5" : "none")
      .attr("marker-end", (d) => `url(#arrow-${d.type})`)

    // Edge labels
    const edgeLabels = container.append("g")
      .selectAll("text")
      .data(edges)
      .join("text")
      .text((d) => d.type as string)
      .attr("font-size", "9px")
      .attr("fill", "#64748b")
      .attr("text-anchor", "middle")

    // Draw nodes
    const node = container.append("g")
      .selectAll<SVGGElement, GraphNode>("g")
      .data(nodes)
      .join("g")
      .style("cursor", "pointer")
      .call(d3.drag<SVGGElement, GraphNode>()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on("drag", (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        })
      )

    // Node circles
    node.append("circle")
      .attr("r", (d) => d.isCenter ? 16 : (d.type === "Statute" ? 10 : 12))
      .attr("fill", (d) => STATUS_COLORS[d.status || "unknown"])
      .attr("stroke", (d) => d.isCenter ? "#1e293b" : "#e2e8f0")
      .attr("stroke-width", (d) => d.isCenter ? 3 : 1.5)

    // Node labels
    node.append("text")
      .text((d) => d.label.length > 25 ? d.label.slice(0, 22) + "..." : d.label)
      .attr("dy", 28)
      .attr("text-anchor", "middle")
      .attr("font-size", "11px")
      .attr("fill", "#334155")
      .attr("font-weight", (d) => d.isCenter ? "600" : "400")

    // Click handler
    node.on("click", (_event, d) => {
      if (onNodeClick) onNodeClick(d.id)
    })

    // Tooltip
    node.append("title")
      .text((d) => `${d.label}\n${d.id}\nStatus: ${d.status || "unknown"}`)

    // Tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as GraphNode).x!)
        .attr("y1", (d) => (d.source as GraphNode).y!)
        .attr("x2", (d) => (d.target as GraphNode).x!)
        .attr("y2", (d) => (d.target as GraphNode).y!)

      edgeLabels
        .attr("x", (d) => ((d.source as GraphNode).x! + (d.target as GraphNode).x!) / 2)
        .attr("y", (d) => ((d.source as GraphNode).y! + (d.target as GraphNode).y!) / 2)

      node.attr("transform", (d) => `translate(${d.x},${d.y})`)
    })

    return () => { simulation.stop() }
  }, [network, width, height, onNodeClick])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading citation network...
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 text-red-500">
        {error}
      </div>
    )
  }

  if (!network || network.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        No citation network data available for this case.
      </div>
    )
  }

  return (
    <div className="relative border rounded-lg overflow-hidden bg-white">
      {/* Legend */}
      <div className="absolute top-3 left-3 bg-white/90 rounded-md p-2 text-xs space-y-1 z-10 border shadow-sm">
        <div className="font-semibold mb-1">Legend</div>
        {Object.entries(STATUS_COLORS).map(([status, color]) => (
          <div key={status} className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: color }} />
            <span>{status.replace("_", " ")}</span>
          </div>
        ))}
        <div className="border-t pt-1 mt-1">
          {Object.entries(EDGE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <span className="w-4 h-0.5 inline-block" style={{ backgroundColor: color }} />
              <span>{type}</span>
            </div>
          ))}
        </div>
      </div>

      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
      />
    </div>
  )
}
```

---

## 12. Performance Targets and Indexing

### 12.1 Targets

| Query | Target | Mechanism |
|-------|--------|-----------|
| Single-hop (is_good_law) | <50ms | GIN index on `citation` property |
| 3-hop precedent chain | <200ms | AGE index + depth limit |
| 5-hop precedent chain | <500ms | AGE index + depth limit |
| Similar cases | <300ms | Bounded by shared-authority join |
| Citation network (depth=2) | <300ms | Capped at 100 nodes |
| Full graph stats | <100ms | Label-based COUNT |

### 12.2 AGE Indexing

AGE stores graph data in PostgreSQL tables under the `ag_catalog` schema (one table per label). Create B-tree indexes on frequently queried properties:

```sql
-- After creating the graph, add indexes on the underlying tables.
-- AGE stores each label in a table like: legal_cases."Case"
-- The property accessor for AGE is: agtype_access_operator

-- Index on Case.citation (most queried property)
CREATE INDEX IF NOT EXISTS idx_case_citation
ON legal_cases."Case" USING btree (
    ag_catalog.agtype_access_operator(properties, '"citation"'::agtype)
);

-- Index on Statute.citation
CREATE INDEX IF NOT EXISTS idx_statute_citation
ON legal_cases."Statute" USING btree (
    ag_catalog.agtype_access_operator(properties, '"citation"'::agtype)
);

-- Index on Case.jurisdiction (for filtered queries)
CREATE INDEX IF NOT EXISTS idx_case_jurisdiction
ON legal_cases."Case" USING btree (
    ag_catalog.agtype_access_operator(properties, '"jurisdiction"'::agtype)
);

-- Index on Case.status (for good-law queries)
CREATE INDEX IF NOT EXISTS idx_case_status
ON legal_cases."Case" USING btree (
    ag_catalog.agtype_access_operator(properties, '"status"'::agtype)
);
```

### 12.3 Query Optimization Notes

1. **LIMIT everything.** Every variable-length path query must have an explicit LIMIT to prevent runaway scans. The `graph_queries.py` functions enforce this.

2. **Depth capping.** Precedent chain queries are clamped to `max_hops <= 5`. Beyond 5 hops, the result set explodes combinatorially and the legal relevance drops.

3. **Second-hop expansion cap.** In `get_citation_network()`, second-hop neighbor expansion is capped at 20 neighbors with 5 edges each. This bounds the network query at ~120 total edge lookups.

4. **Connection pooling.** AGE's `LOAD 'age'` is per-connection. SQLAlchemy's pool ensures this runs once per pooled connection, not once per request.

---

## 13. Testing Strategy

### 13.1 Unit Tests: `backend/tests/test_citation_graph.py`

```python
"""Unit tests for citation graph service.

Uses a real PostgreSQL + AGE instance (test database). Tests are transactional
and rolled back after each test.
"""
import pytest
from backend.services.citation_graph import (
    merge_case_node, merge_statute_node, merge_court_node,
    create_edge, merge_cites_edge, get_graph_stats,
    _classify_citation_type, _infer_source_citation, _infer_case_name,
    normalize_citation,
)
from backend.services.graph_queries import (
    is_good_law, get_precedent_chain, find_similar_cases,
    get_citation_network, what_overrules, get_case_detail,
)


class TestCitationNormalization:
    def test_strip_year_parenthetical(self):
        assert normalize_citation("347 U.S. 483 (1954)") == "347 U.S. 483"

    def test_collapse_spaces(self):
        assert normalize_citation("347  U.S.  483") == "347 U.S. 483"

    def test_normalize_us_periods(self):
        assert normalize_citation("347 U. S. 483") == "347 U.S. 483"

    def test_uk_neutral_citation(self):
        assert normalize_citation("[2024]  UKSC  1") == "[2024] UKSC 1"


class TestCitationTypeClassification:
    def test_us_case(self):
        assert _classify_citation_type("347 U.S. 483", "US") == "Case"

    def test_us_statute(self):
        assert _classify_citation_type("42 U.S.C. ss 1983", "US") == "Statute"

    def test_regulation(self):
        assert _classify_citation_type("Regulation 2016/679", "EU") == "Statute"

    def test_generic_case(self):
        assert _classify_citation_type("[2024] UKSC 1", "UK") == "Case"


class TestInferSourceCitation:
    def test_from_filename(self):
        result = _infer_source_citation("347_US_483.pdf", [])
        assert result is not None
        assert "347" in result

    def test_from_chunk_content(self):
        chunks = [{"content": "No. 1, Original. 347 U.S. 483. Argued December 1952."}]
        result = _infer_source_citation("document.pdf", chunks)
        assert result == "347 U.S. 483"

    def test_non_case_document(self):
        chunks = [{"content": "This is a contract between Party A and Party B."}]
        result = _infer_source_citation("contract.pdf", chunks)
        assert result is None


class TestInferCaseName:
    def test_vs_pattern(self):
        assert _infer_case_name("Brown v Board of Education.pdf") == "Brown v Board of Education"

    def test_no_vs_pattern(self):
        assert _infer_case_name("347_US_483.pdf") is None


# --- Integration tests (require AGE-enabled PostgreSQL) ---

@pytest.fixture
def graph_db(db_session):
    """Fixture that provides a session with AGE initialized.
    Wraps each test in a transaction and rolls back."""
    # Assumes db_session is already configured with AGE
    yield db_session
    db_session.rollback()


@pytest.mark.integration
class TestGraphCRUD:
    def test_merge_case_node_creates(self, graph_db):
        result = merge_case_node(
            graph_db, citation="347 U.S. 483",
            case_name="Brown v. Board of Education",
            jurisdiction="US",
        )
        assert result  # Node returned

    def test_merge_case_node_idempotent(self, graph_db):
        merge_case_node(graph_db, citation="347 U.S. 483", case_name="Brown")
        merge_case_node(graph_db, citation="347 U.S. 483", case_name="Brown v. Board")
        # Should not create duplicate

    def test_create_cites_edge(self, graph_db):
        merge_case_node(graph_db, citation="A")
        merge_case_node(graph_db, citation="B")
        result = merge_cites_edge(graph_db, source_citation="A", target_citation="B")
        assert result

    def test_create_overrules_edge(self, graph_db):
        merge_case_node(graph_db, citation="A")
        merge_case_node(graph_db, citation="B")
        result = create_edge(
            graph_db, source_citation="A", source_label="Case",
            target_citation="B", target_label="Case",
            edge_type="OVERRULES",
        )
        assert result


@pytest.mark.integration
class TestGraphQueries:
    def test_good_law_no_overruling(self, graph_db):
        merge_case_node(graph_db, citation="X")
        result = is_good_law(graph_db, "X")
        assert result["status"] in ("unknown", "good_law")
        assert result["is_overruled"] is False

    def test_good_law_overruled(self, graph_db):
        merge_case_node(graph_db, citation="Old")
        merge_case_node(graph_db, citation="New")
        create_edge(
            graph_db, source_citation="New", source_label="Case",
            target_citation="Old", target_label="Case",
            edge_type="OVERRULES",
        )
        result = is_good_law(graph_db, "Old")
        assert result["is_overruled"] is True
        assert result["status"] == "bad_law"

    def test_precedent_chain(self, graph_db):
        merge_case_node(graph_db, citation="A")
        merge_case_node(graph_db, citation="B")
        merge_case_node(graph_db, citation="C")
        merge_cites_edge(graph_db, source_citation="A", target_citation="B")
        merge_cites_edge(graph_db, source_citation="B", target_citation="C")
        result = get_precedent_chain(graph_db, "A", max_hops=3)
        citations = [item["citation"] for item in result["chain"]]
        assert "B" in citations
        assert "C" in citations

    def test_similar_cases(self, graph_db):
        merge_case_node(graph_db, citation="Target")
        merge_case_node(graph_db, citation="Similar")
        merge_case_node(graph_db, citation="SharedAuth")
        merge_cites_edge(graph_db, source_citation="Target", target_citation="SharedAuth")
        merge_cites_edge(graph_db, source_citation="Similar", target_citation="SharedAuth")
        result = find_similar_cases(graph_db, "Target")
        assert len(result["similar"]) >= 1
        assert result["similar"][0]["citation"] == "Similar"

    def test_citation_network(self, graph_db):
        merge_case_node(graph_db, citation="Center")
        merge_case_node(graph_db, citation="Cited")
        merge_cites_edge(graph_db, source_citation="Center", target_citation="Cited")
        result = get_citation_network(graph_db, "Center", depth=1)
        assert len(result["nodes"]) >= 2
        assert len(result["edges"]) >= 1


@pytest.mark.integration
class TestGraphStats:
    def test_stats_returns_counts(self, graph_db):
        merge_case_node(graph_db, citation="StatsTest")
        stats = get_graph_stats(graph_db)
        assert "case_count" in stats
```

### 13.2 E2E Test: `backend/tests/test_citation_graph_e2e.py`

```python
"""End-to-end test: upload a legal document and verify the citation graph is populated."""

@pytest.mark.e2e
def test_document_ingestion_populates_graph(client, sample_legal_pdf):
    """Upload a PDF with known citations, verify graph nodes and edges are created."""
    # 1. Create matter with document
    response = client.post("/matters", files={"files": sample_legal_pdf}, data={"name": "Graph E2E"})
    matter_id = response.json()["id"]

    # 2. Wait for processing
    # ... poll status ...

    # 3. Check graph stats
    stats = client.get("/graph/stats").json()
    assert stats["case_count"] > 0
    assert stats["total_edges"] > 0

    # 4. Check good law for a known citation
    result = client.get("/graph/case/347%20U.S.%20483/good-law").json()
    assert result["citation"] == "347 U.S. 483"
```

---

## 14. Migration and Rollback

### 14.1 Alembic Migration

Create `backend/alembic/versions/xxxx_add_citation_graph.py`:

```python
"""Add citation knowledge graph using Apache AGE extension.

Revision ID: <auto-generated>
Revises: <previous-head>
Create Date: 2026-03-23
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '<auto-generated>'
down_revision = '<previous-head>'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Install AGE extension
    op.execute("CREATE EXTENSION IF NOT EXISTS age;")

    # 2. Load AGE (required for graph creation)
    op.execute("LOAD 'age';")
    op.execute("SET search_path TO ag_catalog, public;")

    # 3. Create the graph (idempotent check)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'legal_cases'
            ) THEN
                PERFORM ag_catalog.create_graph('legal_cases');
            END IF;
        END $$;
    """)

    # 4. Create property indexes on AGE tables
    # Note: AGE tables are created lazily when the first node of each label
    # is inserted. These indexes will be created by a post-migration script
    # or the first time the graph service runs.


def downgrade() -> None:
    # Drop the graph (removes all nodes and edges)
    op.execute("LOAD 'age';")
    op.execute("SET search_path TO ag_catalog, public;")
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'legal_cases'
            ) THEN
                PERFORM ag_catalog.drop_graph('legal_cases', true);
            END IF;
        END $$;
    """)

    # Optionally remove the extension (only if no other graphs exist)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph) THEN
                DROP EXTENSION IF EXISTS age CASCADE;
            END IF;
        END $$;
    """)
```

### 14.2 Post-Migration Index Script

Run after the first batch of data is indexed (AGE creates label tables lazily):

```sql
-- backend/scripts/create_age_indexes.sql

LOAD 'age';
SET search_path TO ag_catalog, public;

-- Only run these after at least one Case and Statute node exist
-- (otherwise the tables don't exist yet)

-- Case citation index
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'legal_cases' AND table_name = 'Case') THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_case_citation
                 ON legal_cases."Case" USING btree (
                     ag_catalog.agtype_access_operator(properties, ''"citation"''::agtype)
                 )';
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_case_jurisdiction
                 ON legal_cases."Case" USING btree (
                     ag_catalog.agtype_access_operator(properties, ''"jurisdiction"''::agtype)
                 )';
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_case_status
                 ON legal_cases."Case" USING btree (
                     ag_catalog.agtype_access_operator(properties, ''"status"''::agtype)
                 )';
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'legal_cases' AND table_name = 'Statute') THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_statute_citation
                 ON legal_cases."Statute" USING btree (
                     ag_catalog.agtype_access_operator(properties, ''"citation"''::agtype)
                 )';
    END IF;
END $$;
```

### 14.3 Rollback Plan

The migration `downgrade()` drops the entire `legal_cases` graph. This is a non-destructive operation for the rest of the application because:

1. The graph feature is gated behind `CITATION_GRAPH_ENABLED=true`.
2. All graph code paths have try/except guards that fail gracefully.
3. No relational tables depend on graph data.
4. The feature flag can be set to `false` immediately, disabling all graph queries without a migration.

Emergency rollback procedure:

```bash
# 1. Disable the feature flag immediately (no deploy needed)
echo "CITATION_GRAPH_ENABLED=false" >> backend/.env

# 2. Run migration downgrade (when ready)
cd backend && alembic downgrade -1
```

---

## 15. Six-Week Roadmap

### Week 1: Infrastructure + Core Graph CRUD

| Day | Task | Output |
|-----|------|--------|
| 1-2 | Install AGE on dev PostgreSQL. Verify Cypher queries work via `psql`. | AGE running locally |
| 3 | Write Alembic migration. Add config settings to `config.py`. | Migration file + config changes |
| 4 | Implement `_cypher()` helper and `_parse_agtype()` in `citation_graph.py`. | Core execution layer |
| 5 | Implement `merge_case_node()`, `merge_statute_node()`, `merge_court_node()`. | Node CRUD |

**Milestone:** Can create and query nodes via Python.

### Week 2: Edge Operations + Extraction Pipeline

| Day | Task | Output |
|-----|------|--------|
| 1 | Implement `create_edge()` and `merge_cites_edge()`. | Edge CRUD |
| 2-3 | Implement `extract_and_index_citations()` bulk indexer. | Ingestion pipeline |
| 4 | Add citation normalization (`normalize_citation()`). | Dedup logic |
| 5 | Integrate into `tasks.py` (step 7b). | End-to-end ingestion working |

**Milestone:** Uploading a document populates the citation graph.

### Week 3: LLM Relationship Classification + Good Law

| Day | Task | Output |
|-----|------|--------|
| 1-2 | Implement `_classify_relationships_batch()` with Gemini. | LLM classifier |
| 3 | Implement `is_good_law()` and `what_overrules()` queries. | Good law analysis |
| 4 | Implement `get_precedent_chain()`. | Chain traversal |
| 5 | Implement `find_similar_cases()`. | Similarity query |

**Milestone:** All five core graph queries functional.

### Week 4: API Endpoints + RAG Integration

| Day | Task | Output |
|-----|------|--------|
| 1-2 | Build REST API routes (`/graph/*`) with schemas. | API layer |
| 3 | Implement `_build_graph_context()` in `rag_engine.py`. | Graph-enhanced RAG |
| 4 | Implement `get_citation_network()` for visualization. | Network data endpoint |
| 5 | Add AGE property indexes. Run benchmark queries. | Performance validation |

**Milestone:** Graph context appears in RAG answers. API endpoints work.

### Week 5: Frontend Visualization

| Day | Task | Output |
|-----|------|--------|
| 1 | Add D3.js dependency. Create TypeScript types. | Frontend setup |
| 2-3 | Build `CitationGraph.tsx` with force simulation. | Interactive graph rendering |
| 4 | Add legend, zoom, drag, click-to-navigate. | UX polish |
| 5 | Integrate graph component into citation panel and matter detail page. | Feature visible in UI |

**Milestone:** Users can see and interact with citation networks.

### Week 6: Testing + Hardening + Documentation

| Day | Task | Output |
|-----|------|--------|
| 1 | Write unit tests (`test_citation_graph.py`). | Unit test suite |
| 2 | Write integration tests (requires AGE in CI). | Integration tests |
| 3 | Write E2E test (upload PDF, verify graph). | E2E coverage |
| 4 | Performance testing under load (1000+ nodes). Optimize slow queries. | Performance report |
| 5 | Code review. Update MEMORY.md. Write deployment runbook. | Ship-ready |

**Milestone:** Feature complete, tested, and documented.

---

## 16. Appendix: File Inventory

### New Files

| Path | Lines (est.) | Purpose |
|------|-------------|---------|
| `backend/services/citation_graph.py` | ~500 | Graph CRUD, bulk indexing, relationship classification |
| `backend/services/graph_queries.py` | ~400 | Read-only Cypher query library |
| `backend/routers/graph.py` | ~120 | REST API routes for graph queries |
| `backend/alembic/versions/xxxx_add_citation_graph.py` | ~60 | Database migration |
| `backend/scripts/create_age_indexes.sql` | ~40 | Post-migration index creation |
| `backend/tests/test_citation_graph.py` | ~250 | Unit + integration tests |
| `backend/tests/test_citation_graph_e2e.py` | ~80 | End-to-end test |
| `frontend/components/CitationGraph.tsx` | ~250 | D3.js network visualization |

### Modified Files

| Path | Change Summary |
|------|---------------|
| `backend/config.py` | Add `citation_graph_enabled`, `citation_graph_name`, `age_graph_path` |
| `backend/database.py` | Add AGE connection event listener |
| `backend/tasks.py` | Add step 7b: citation extraction + graph indexing |
| `backend/services/rag_engine.py` | Add `_build_graph_context()`, inject into query pipeline |
| `backend/schemas.py` | Add graph-related Pydantic models |
| `backend/main.py` | Register `/graph` router |
| `backend/requirements.txt` | No new PyPI packages needed (AGE uses raw SQL via SQLAlchemy) |
| `frontend/lib/types.ts` | Add `GraphNode`, `GraphEdge`, `CitationNetwork`, etc. |
| `frontend/lib/api-services.ts` | Add `getGraphStats()`, `checkGoodLaw()`, etc. |

### Dependency Changes

| Package | Version | Purpose | Cost |
|---------|---------|---------|------|
| Apache AGE (PG extension) | 1.5.0 | Graph database | $0 (Apache 2.0) |
| d3 (npm) | ^7.9.0 | Frontend visualization | $0 (BSD) |
| @types/d3 (npm) | ^7.4.3 | TypeScript definitions | $0 |

No new Python packages are required. AGE is accessed via raw SQL through the existing SQLAlchemy + psycopg2 stack.
