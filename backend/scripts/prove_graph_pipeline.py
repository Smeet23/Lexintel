"""Prove the case_cites edge pipeline end-to-end, bypassing ONLY the quota-blocked
Gemini text->triples call.

We monkeypatch extract_relationships_llm to return the triples a correct Gemini
extraction WOULD return for the Brown/Dobbs/Miranda memo, then run the REAL
extract_and_index_citations and verify nodes/edges/good-law in Postgres and via
the graph API serializer. Everything downstream of the (quota-blocked) LLM call
is exercised for real.

Run: PYTHONPATH=. backend/venv/bin/python backend/scripts/prove_graph_pipeline.py
"""
import asyncio
import uuid

import backend.services.citation_graph as cg
from backend.database import get_session_factory
from backend.models import Matter, Document, CitationNode, CitationEdge

SessionLocal = get_session_factory()

FAKE_TRIPLES = [
    {"citing_citation": "347 U.S. 483", "citing_name": "Brown v. Board of Education", "citing_year": 1954,
     "cited_citation": "163 U.S. 537", "cited_name": "Plessy v. Ferguson", "cited_year": 1896,
     "treatment": "OVERRULES", "evidence": "the Supreme Court overruled Plessy v. Ferguson"},
    {"citing_citation": "597 U.S. 215", "citing_name": "Dobbs v. Jackson", "citing_year": 2022,
     "cited_citation": "410 U.S. 113", "cited_name": "Roe v. Wade", "cited_year": 1973,
     "treatment": "OVERRULES", "evidence": "the Court overruled Roe v. Wade"},
    {"citing_citation": "597 U.S. 215", "citing_name": "Dobbs v. Jackson", "citing_year": 2022,
     "cited_citation": "505 U.S. 833", "cited_name": "Planned Parenthood v. Casey", "cited_year": 1992,
     "treatment": "OVERRULES", "evidence": "and Planned Parenthood v. Casey"},
    {"citing_citation": "384 U.S. 436", "citing_name": "Miranda v. Arizona", "citing_year": 1966,
     "cited_citation": "378 U.S. 478", "cited_name": "Escobedo v. Illinois", "cited_year": 1964,
     "treatment": "DISTINGUISHES", "evidence": "the Court distinguished Escobedo v. Illinois"},
]

OUT = []


def log(msg):
    OUT.append(str(msg))


async def _fake_rel(_text):
    return FAKE_TRIPLES


async def _fake_classify(*a, **k):
    return {"treatment": "CITES", "confidence": 0.5}


def main():
    cg.extract_relationships_llm = _fake_rel
    cg.classify_treatment = _fake_classify

    db = SessionLocal()
    mid = uuid.uuid4()
    did = uuid.uuid4()
    try:
        m = Matter(id=mid, name="PIPELINE PROOF", status="ready", file_type="txt",
                   blob_storage_path="x", is_deleted=False)
        db.add(m)
        db.flush()
        doc = Document(id=did, matter_id=mid, name="Proof Memo", status="ready",
                       file_type="txt", blob_storage_path="x")
        db.add(doc)
        db.commit()

        chunks = [{
            "content": ("In Brown v. Board of Education, 347 U.S. 483 (1954) the Court "
                        "overruled Plessy v. Ferguson, 163 U.S. 537 (1896)."),
            "section_name": "", "page_num": 1,
        }]
        res = asyncio.run(cg.extract_and_index_citations(db, mid, did, "Proof Memo", "US", chunks))
        log(f"INDEX_RESULT: {res}")

        edges = db.query(CitationEdge).filter(
            CitationEdge.matter_id == mid, CitationEdge.edge_kind == "case_cites").all()
        log(f"CASE_CITES_EDGES={len(edges)}")
        for e in edges:
            s = db.query(CitationNode).filter(CitationNode.id == e.source_id).first()
            t = db.query(CitationNode).filter(CitationNode.id == e.target_id).first()
            sl = f"{s.citation_text}({s.year})" if s else f"<missing {e.source_id}>"
            tl = f"{t.citation_text}({t.year})" if t else f"<missing {e.target_id}>"
            log(f"  EDGE {sl} --{e.treatment}--> {tl}")

        for cite in ["163 U.S. 537", "410 U.S. 113", "347 U.S. 483"]:
            gl = cg.is_good_law(db, cite)
            log(f"GOODLAW {cite}: status={gl['status']} is_good_law={gl['is_good_law']}")

        g = cg.get_matter_graph(db, mid)
        kinds = {}
        for e in g["edges"]:
            kinds[e.get("edge_kind")] = kinds.get(e.get("edge_kind"), 0) + 1
        nid = {n["id"] for n in g["nodes"]}
        dangling = [e for e in g["edges"] if e["source_id"] not in nid or e["target_id"] not in nid]
        log(f"API: nodes={len(g['nodes'])} edges={len(g['edges'])} stats={g['stats']}")
        log(f"API_EDGE_KINDS={kinds}")
        log(f"DANGLING_EDGES={len(dangling)} (must be 0)")
    finally:
        db.query(CitationEdge).filter(CitationEdge.matter_id == mid).delete()
        db.query(CitationNode).filter(CitationNode.matter_id == mid).delete()
        db.query(Document).filter(Document.id == did).delete()
        db.query(Matter).filter(Matter.id == mid).delete()
        db.commit()
        db.close()

    with open("/tmp/proof_result.txt", "w") as f:
        f.write("\n".join(OUT))
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
