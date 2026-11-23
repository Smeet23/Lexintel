"""Ingestion integration checks against a disposable PostgreSQL database."""

from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID
import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from lexintel_ingest.api import app
from lexintel_ingest.db import connection
from lexintel_ingest.embeddings import DIMENSIONS, EmbeddingBatch, EmbeddingFailure
from lexintel_ingest.rank_service import _index_batch
from lexintel_ingest.worker import claim_job, run_once


AGREEMENT = b"""MASTER SERVICES AGREEMENT
Made as of January 15, 2025 between Alpha Ltd ("Supplier") and Beta Ltd ("Customer").
Effective Date: February 1, 2025
ARTICLE VII TERMINATION
7.2 Notice. Either party may terminate with 30 days' written notice.
7.3 Exceptions. See Section 7.2.
7.3(b) Fraud permits immediate termination.
"""
AMENDMENT = b"""AMENDMENT NO. 1
This amendment is to the Master Services Agreement dated January 15, 2025.
Effective Date: March 1, 2025
Section 7.2 is hereby replaced.
7.2 Notice. Either party may terminate with 45 days' written notice.
"""


def render_agreement():
    stream = BytesIO()
    pdf = Canvas(stream)
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawString(60, 780, "MASTER SERVICES AGREEMENT")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(60, 748, "Made as of January 15, 2025 between Alpha Ltd and Beta Ltd.")
    pdf.drawString(60, 722, "Effective Date: February 1, 2025")
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(60, 675, "ARTICLE VII TERMINATION")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(60, 652, "7.2 Notice. Either party may terminate")
    pdf.drawString(60, 636, "with 30 days' written notice.")
    pdf.drawString(60, 606, "See Schedule A for the amounts below.")
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(60, 564, "SCHEDULE A - TERMINATION FEES")

    for y in (540, 505, 470, 435):
        pdf.line(60, y, 485, y)
    for x in (60, 250, 365, 485):
        pdf.line(x, 435, x, 540)
    rows = (
        ("Event", "Amount", "Unit"),
        ("Late delivery", "USD 500", "per day"),
        ("Early termination", "USD 1200*", "per contract"),
    )
    for index, row in enumerate(rows):
        pdf.setFont("Helvetica-Bold" if index == 0 else "Helvetica", 11)
        for x, value in zip((70, 260, 375), row, strict=True):
            pdf.drawString(x, 519 - index * 35, value)
    pdf.drawString(60, 413, "* The fee applies only in the first year.")
    pdf.save()
    return stream.getvalue()


def render_scan(mixed=False):
    raster = Image.new("RGB", (1500, 640), "white")
    draw = ImageDraw.Draw(raster)
    font = ImageFont.load_default(size=54)
    draw.multiline_text((90, 100), "EXHIBIT A DELIVERY RECORD\n\nCourier delivered the notice on May 2, 2025.",
                        fill="black", font=font, spacing=24)
    stream = BytesIO()
    pdf = Canvas(stream)
    if mixed:
        pdf.setFont("Helvetica", 14)
        pdf.drawString(50, 760, "Delivery Letter - Original Content")
    pdf.drawImage(ImageReader(raster), 37, 365, width=520, height=222)
    pdf.save()
    return stream.getvalue()


def render_word():
    from docx import Document

    source = Document()
    source.add_heading("SUPPLY AGREEMENT", 0)
    source.add_paragraph("Effective Date: April 1, 2025")
    source.add_heading("ARTICLE III NOTICES", 1)
    source.add_paragraph("3.1 Notices. Send by registered courier.")
    source.add_paragraph("Obtain a delivery receipt.", style="List Number")
    stream = BytesIO()
    source.save(stream)
    return stream.getvalue()


class OfflineEmbedder:
    def __init__(self, failure_on=None):
        self.requests = []
        self.failure_on = failure_on

    def embed(self, texts):
        self.requests.append(tuple(texts))
        if len(self.requests) == self.failure_on:
            raise EmbeddingFailure("provider_http_429", "Simulated rate limit", retryable=True)
        vectors = []
        for passage in texts:
            digest = sha256(passage.encode()).digest()
            vectors.append([float(digest[i % len(digest)] + 1) / 256 for i in range(DIMENSIONS)])
        return EmbeddingBatch(vectors, sum(len(passage.split()) for passage in texts))


@unittest.skipUnless(os.getenv("LEXINTEL_INTEGRATION_TESTS") == "1", "requires a dedicated PostgreSQL database and parser tools")
class IngestionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.api = TestClient(app)
        self.matter_ids = []
        self.embedder = OfflineEmbedder()
        self.assertEqual(self.api.get("/health").status_code, 200)

    def tearDown(self):
        for matter_id in self.matter_ids:
            self.api.delete(f"/matters/{matter_id}")
        self.api.close()

    def _matter(self, files):
        response = self.api.post(
            "/matters", data={"name": "LexIntel integration fixture"},
            files=[("files", (name, content)) for name, content in files],
        )
        self.assertEqual(response.status_code, 201, response.text)
        matter_id = response.json()["id"]
        self.matter_ids.append(matter_id)
        return matter_id

    def _documents(self, matter_id):
        result = self.api.get(f"/matters/{matter_id}/documents")
        result.raise_for_status()
        return {doc["name"]: doc for doc in result.json()}

    def _artifact(self, matter_id, doc):
        response = self.api.get(f"/matters/{matter_id}/documents/{doc['id']}/artifact")
        self.assertEqual(response.status_code, 200, response.text)
        artifact = response.json()
        for chunk in artifact["chunks"]:
            for span in chunk["source_spans"]:
                page = artifact["pages"][span["page_number"] - 1]
                self.assertEqual(page["text"][span["start"]:span["end"]], span["text"])
        return artifact

    def test_pdf_ocr_docx_and_amendment_workflow(self):
        inputs = {
            "agreement.pdf": render_agreement(),
            "amendment.txt": AMENDMENT,
            "scan.pdf": render_scan(),
            "mixed.pdf": render_scan(mixed=True),
            "agreement.docx": render_word(),
            "bad.pdf": b"%PDF-1.7 malformed fixture",
        }
        matter_id = self._matter(list(inputs.items()))
        for _ in inputs:
            self.assertTrue(run_once(self.embedder))
        documents = self._documents(matter_id)
        self.assertEqual(documents["bad.pdf"]["ingestion_status"], "failed")
        for name in set(inputs) - {"bad.pdf"}:
            self.assertIn(documents[name]["ingestion_status"], {"ready", "ready_with_warnings"}, documents[name])
            self.assertGreater(documents[name]["embedding_requests"], 0)

        contract = self._artifact(matter_id, documents["agreement.pdf"])
        self.assertEqual(contract["profile"]["type"], "agreement")
        self.assertEqual(contract["profile"]["metadata"]["effective_date"]["value"], "2025-02-01")
        self.assertNotIn("layout_unavailable", [warning["code"] for warning in contract["warnings"]], contract["warnings"])
        fee = next(chunk for chunk in contract["chunks"] if "USD 500" in chunk["content"] and chunk["kind"] == "table_row")
        self.assertIn("per day", fee["content"])
        self.assertTrue(any(source["rectangles"] for source in fee["source_spans"]))
        early = next(chunk for chunk in contract["chunks"] if "USD 1200" in chunk["content"] and chunk["kind"] == "table_row")
        self.assertIn("first year", early["content"])

        amendment = self._artifact(matter_id, documents["amendment.txt"])
        self.assertEqual(amendment["profile"]["type"], "amendment")
        self.assertTrue(any(
            ref["relation"] == "replaces" and ref.get("target_document_id") == documents["agreement.pdf"]["id"]
            for ref in amendment["references"]
        ), amendment["references"])

        for filename, mode in (("scan.pdf", "ocr"), ("mixed.pdf", "mixed")):
            scan = self._artifact(matter_id, documents[filename])
            self.assertEqual(scan["pages"][0]["mode"], mode)
            self.assertIn("Courier delivered the notice", scan["pages"][0]["text"])
            self.assertTrue(scan["pages"][0]["character_boxes"])

        word = self._artifact(matter_id, documents["agreement.docx"])
        self.assertEqual(word["rendition"], "rendered_pdf")
        self.assertEqual(word["profile"]["metadata"]["effective_date"]["value"], "2025-04-01")
        self.assertIn("1.", word["pages"][0]["text"])
        preview = self.api.get(f"/matters/{matter_id}/documents/{documents['agreement.docx']['id']}/preview")
        self.assertTrue(preview.content.startswith(b"%PDF"))
        original = self.api.get(f"/matters/{matter_id}/documents/{documents['agreement.docx']['id']}/download")
        self.assertEqual(original.content, inputs["agreement.docx"])

        with connection() as conn:
            hit = conn.execute(
                """
                SELECT c.content, vector_dims(c.embedding) AS dimensions
                FROM document_chunks c JOIN documents d ON c.document_id = d.id
                WHERE d.matter_id = %s AND d.id = %s
                ORDER BY search_text <@> to_bm25query('early termination', 'document_chunks_bm25')
                LIMIT 1
                """,
                (matter_id, documents["agreement.pdf"]["id"]),
            ).fetchone()
        self.assertEqual(hit["dimensions"], 1024)
        self.assertIn("Early termination", hit["content"])

    def test_keyless_parse_duplicate_upload_and_cached_retry(self):
        matter_id = self._matter([("agreement.txt", AGREEMENT)])
        with patch("lexintel_ingest.worker.embedding_key", return_value=""):
            self.assertTrue(run_once())
        doc = self._documents(matter_id)["agreement.txt"]
        self.assertEqual(doc["ingestion_status"], "awaiting_index")
        self.assertEqual(doc["embedding_requests"], 0)
        artifact = self._artifact(matter_id, doc)
        self.assertTrue(artifact["chunks"])

        identical = self.api.post(f"/matters/{matter_id}/documents", files={"file": ("another.txt", AGREEMENT)})
        self.assertEqual(identical.json()["id"], doc["id"])
        retry_url = f"/matters/{matter_id}/documents/{doc['id']}/retry"
        self.assertEqual(self.api.post(retry_url).status_code, 202)

        provider = OfflineEmbedder(failure_on=2)
        with patch("lexintel_ingest.worker.MAX_BATCH_ITEMS", 1):
            self.assertTrue(run_once(provider))
            self.assertEqual(self._documents(matter_id)["agreement.txt"]["ingestion_status"], "retrying")
            with connection() as conn:
                conn.execute("UPDATE ingestion_jobs SET available_at = now() WHERE id = %s", (doc["task_id"],))
            self.assertTrue(run_once(provider))

        finished = self._documents(matter_id)["agreement.txt"]
        self.assertEqual(finished["ingestion_status"], "ready")
        self.assertEqual(provider.requests.count(provider.requests[0]), 1)
        self.assertIsNone(claim_job())
        self.assertEqual(len(self.api.get(f"/matters/{matter_id}/chunks").json()), len(artifact["chunks"]))

    def test_ask_retrieves_keyless_evidence_and_scopes_chat_and_sources(self):
        first = self._matter([("statement.txt", b"The approved payment is exactly fifty dollars.\n")])
        second = self._matter([("unrelated.txt", b"The approved payment is exactly ninety dollars.\n")])
        with patch("lexintel_ingest.worker.embedding_key", return_value=""):
            self.assertTrue(run_once())
            self.assertTrue(run_once())
        first_url = f"/matters/{first}"
        second_url = f"/matters/{second}"
        result = self.api.post(f"{first_url}/ask", json={"question": "What is the approved payment?"})
        self.assertEqual(result.status_code, 200, result.text)
        body = result.json()
        self.assertEqual((body["model"], body["tokens_used"]), ("bm25", 0))
        self.assertEqual(len(body["sources"]), 1)
        self.assertIn("fifty dollars", body["sources"][0]["content"])
        self.assertEqual(body["sources"][0]["pages"], [1])
        self.assertEqual(self._documents(first)["statement.txt"]["embedding_requests"], 0)
        conversation = body["conversation_id"]
        restored = self.api.get(f"{first_url}/conversations/{conversation}").json()["queries"]
        self.assertEqual(restored[0]["citations"], body["sources"])
        self.assertEqual(self.api.get(f"{second_url}/conversations/{conversation}").status_code, 404)
        misuse = self.api.post(f"{second_url}/ask", json={"question": "Payment", "conversation_id": conversation})
        self.assertEqual(misuse.status_code, 404)

        missing = self.api.post(
            f"{first_url}/ask", json={"question": "nonexistentzzfoo", "conversation_id": conversation},
        )
        self.assertEqual(missing.json()["sources"], [])
        self.assertIn("No matching passages", missing.json()["answer"])
        self.assertEqual(self.api.get(first_url).json()["queries_count"], 2)

        foreign = self._documents(second)["unrelated.txt"]["id"]
        artifact = self._artifact(second, {"id": foreign})
        fingerprint = sha256(artifact["chunks"][0]["search_text"].encode()).hexdigest()
        with (
            patch("lexintel_ingest.retrieval.settings", SimpleNamespace(ranker_url="http://ranking:8001")),
            patch("lexintel_ingest.retrieval._ranked_references", return_value=[(UUID(foreign), 0, fingerprint)]),
        ):
            fallback = self.api.post(
                f"{first_url}/ask", json={"question": "approved payment", "conversation_id": conversation},
            ).json()
        self.assertEqual(fallback["model"], "bm25")
        self.assertEqual([source["document_name"] for source in fallback["sources"]], ["statement.txt"])

    def test_local_vectors_follow_passage_replacement(self):
        matter = self._matter([("original.txt", b"The retention period is ten days.\n")])
        with patch("lexintel_ingest.worker.embedding_key", return_value=""):
            self.assertTrue(run_once())
        document = self._documents(matter)["original.txt"]["id"]
        embedder = Mock()
        embedder.encode.return_value.tolist.return_value = [[1.0] + [0.0] * 383]
        self.assertTrue(_index_batch(embedder))
        self.assertFalse(_index_batch(embedder))

        with connection() as conn:
            conn.execute(
                """UPDATE parsed_documents
                   SET artifact = jsonb_set(artifact, '{chunks,0,search_text}', '"Updated retention period"'::jsonb)
                   WHERE document_id = %s""",
                (document,),
            )
            self.assertIsNone(conn.execute(
                "SELECT 1 FROM local_passage_vectors WHERE document_id = %s", (document,)
            ).fetchone())

        self.assertTrue(_index_batch(embedder))
        embedder.encode.assert_called_with(["Updated retention period"], "passage")

    def test_worker_recovers_an_expired_lease_and_honors_cancelled_jobs(self):
        matter_id = self._matter([("expired.txt", AGREEMENT)])
        claim = claim_job()
        self.assertIsNotNone(claim)
        with connection() as conn:
            conn.execute("UPDATE ingestion_jobs SET lease_until = now() - interval '1 second' WHERE id = %s", (claim.id,))
        self.assertTrue(run_once(self.embedder))
        ready = self._documents(matter_id)["expired.txt"]
        self.assertEqual(ready["ingestion_status"], "ready")
        self.assertEqual(ready["attempts"], 2)

        second = self._matter([("cancelled.txt", AMENDMENT)])
        self.assertTrue(self.api.post(f"/matters/{second}/cancel").json()["cancelled"])
        self.assertFalse(run_once(self.embedder))
        cancelled = self._documents(second)["cancelled.txt"]
        self.assertEqual(cancelled["ingestion_status"], "cancelled")
        self.assertEqual(cancelled["embedding_requests"], 0)
