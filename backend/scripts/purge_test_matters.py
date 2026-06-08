"""Hard-purge leftover TEST matters and their cascading rows from the dev DB.

The API only soft-deletes (is_deleted=true); this removes the physical rows for
matters created by automated citation-graph testing so they stop polluting the
graph + counts. Matched by an explicit allow-list of test-name patterns AND
is_deleted=true OR name-pattern, scoped to obvious test fixtures only.

SAFE: prints what it will delete and requires CONFIRM=1 to actually delete.
Run preview:  PYTHONPATH=. backend/venv/bin/python backend/scripts/purge_test_matters.py
Run for real: CONFIRM=1 PYTHONPATH=. backend/venv/bin/python backend/scripts/purge_test_matters.py
"""
import os

from sqlalchemy import text

from backend.database import get_session_factory

# Explicit test-name patterns (case-insensitive LIKE). Conservative on purpose.
_PATTERNS = [
    "%Graph Test%", "%Citation Graph Fix%", "%UI Browser Test%", "%UI Graph Test%",
    "%UI Final Test%", "%Graph E2E Test%", "%Live Robust Test%", "%Groq Fallback Proof%",
    "%Supersession Fixed%",
]

# Tables with a matter_id FK, ordered so that rows referencing OTHER child
# tables are deleted first (documents reference amendment_chains, so documents
# must go before amendment_chains; chunks/queries before documents).
_CHILD_TABLES = [
    "citation_edges", "citation_nodes", "audit_logs", "contract_reviews",
    "conversations", "drafts", "processing_jobs", "saved_precedents",
    "chunks", "queries", "documents", "amendment_chains",
]


def main() -> None:
    db = get_session_factory()()
    try:
        where = " OR ".join("name ILIKE :p%d" % i for i in range(len(_PATTERNS)))
        params = {f"p{i}": pat for i, pat in enumerate(_PATTERNS)}
        rows = db.execute(
            text(f"SELECT id, name FROM matters WHERE {where}"), params
        ).fetchall()
        if not rows:
            print("No test matters match. Nothing to do.")
            return
        print(f"Matched {len(rows)} test matters:")
        for r in rows:
            print(f"  {r[0]}  {r[1]}")
        if os.environ.get("CONFIRM") != "1":
            print("\nPREVIEW ONLY. Re-run with CONFIRM=1 to delete.")
            return

        ids = [r[0] for r in rows]
        for tbl in _CHILD_TABLES:
            db.execute(
                text(f"DELETE FROM {tbl} WHERE matter_id = ANY(:ids)"), {"ids": ids}
            )
        db.execute(text("DELETE FROM matters WHERE id = ANY(:ids)"), {"ids": ids})
        db.commit()
        print(f"\nPurged {len(ids)} test matters and all cascading rows.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
