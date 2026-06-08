"""One-time heal of citation nodes fragmented by the old reporter-spacing regex.

The retired ``_normalise_text`` regex produced non-canonical keys like
'347 U.S.483' (no space before the page), forking a second node for cases that
already existed under the eyecite-canonical '347 U.S. 483'. This script re-keys
every node by the current ``_canonical_citation_key`` and merges duplicates:
edges are re-pointed onto the oldest (canonical) node, missing fields are
backfilled, and the duplicate is deleted. Idempotent — safe to re-run.

Run: cd backend && PYTHONPATH=.. ../backend/venv/bin/python -m backend.scripts.heal_fragmented_nodes
or:  PYTHONPATH=. backend/venv/bin/python backend/scripts/heal_fragmented_nodes.py
"""
from collections import defaultdict

from backend.database import get_session_factory
from backend.models import CitationNode, CitationEdge
from backend.services.citation_graph import _canonical_citation_key


def heal() -> dict:
    Session = get_session_factory()
    db = Session()
    merged = 0
    edges_repointed = 0
    groups: dict[str, list[CitationNode]] = defaultdict(list)
    try:
        for node in db.query(CitationNode).all():
            groups[_canonical_citation_key(node.citation_text)].append(node)

        for key, nodes in groups.items():
            if len(nodes) < 2:
                # Single node: re-key to canonical if it drifted (cheap, idempotent).
                n = nodes[0]
                if n.citation_text != key:
                    n.citation_text = key
                continue
            # Keep the oldest as canonical; merge the rest into it.
            nodes.sort(key=lambda n: (n.created_at is None, n.created_at))
            canonical = nodes[0]
            canonical.citation_text = key
            for dup in nodes[1:]:
                for col in (CitationEdge.source_id, CitationEdge.target_id):
                    n = db.query(CitationEdge).filter(col == dup.id).update(
                        {col: canonical.id}, synchronize_session=False
                    )
                    edges_repointed += n
                for field in ("name", "court", "year", "jurisdiction",
                              "source_url", "courtlistener_id", "is_good_law"):
                    if getattr(canonical, field, None) in (None, "") and \
                       getattr(dup, field, None) not in (None, ""):
                        setattr(canonical, field, getattr(dup, field))
                db.delete(dup)
                merged += 1
        db.commit()
    finally:
        db.close()
    return {"nodes_merged": merged, "edges_repointed": edges_repointed}


if __name__ == "__main__":
    result = heal()
    print(f"HEAL DONE: {result}")
