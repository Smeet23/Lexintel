import unittest
from uuid import uuid4

import httpx

from lexintel_ingest.embeddings import DIMENSIONS, EmbeddingFailure, OpenRouterEmbedder
from lexintel_ingest.worker import Job, _index


class EmbeddingProtocolTests(unittest.TestCase):
    def test_embeddings_are_sorted_and_batched_with_document_input_type(self):
        seen = []

        def respond(request):
            seen.append(request.read())
            return httpx.Response(200, json={
                "data": [{"index": 1, "embedding": [1] * DIMENSIONS},
                         {"index": 0, "embedding": [2] * DIMENSIONS}],
                "usage": {"prompt_tokens": 7},
            })

        with httpx.Client(base_url="https://provider.test", transport=httpx.MockTransport(respond)) as client:
            result = OpenRouterEmbedder(client).embed(["First", "Second"])
        self.assertEqual([vector[0] for vector in result.vectors], [2, 1])
        self.assertEqual(result.input_tokens, 7)
        self.assertIn(b'"input_type":"search_document"', seen[0])

    def test_provider_failure_is_retryable_without_logging_body(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(429, text="Sensitive details"))
        with httpx.Client(base_url="https://provider.test", transport=transport) as client:
            with self.assertRaises(EmbeddingFailure) as raised:
                OpenRouterEmbedder(client).embed(["Passage"])
        self.assertTrue(raised.exception.retryable)
        self.assertNotIn("Sensitive", str(raised.exception))


class BudgetTests(unittest.TestCase):
    def test_document_budget_is_checked_before_provider_or_database_work(self):
        job = Job(uuid4(), uuid4(), uuid4(), 1, "large.txt", "txt", "large.txt", "hash")
        chunk = {"search_text": "x" * 2_000_001}
        with self.assertRaises(EmbeddingFailure) as raised:
            _index(job, {"chunks": [chunk]}, None, None)
        self.assertEqual(raised.exception.code, "embedding_input_limit")
