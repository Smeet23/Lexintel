"""Batch document embeddings through OpenRouter with validated outputs."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import httpx

MODEL = "mongodb/voyage-4-large"
DIMENSIONS = 1024
MAX_BATCH_ITEMS = 32
MAX_INPUT_BYTES = 2_000_000


class EmbeddingFailure(Exception):
    def __init__(self, code: str, detail: str, *, retryable: bool = False):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.retryable = retryable


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    input_tokens: int


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingBatch: ...


class OpenRouterEmbedder:
    def __init__(self, client: httpx.Client):
        self.client = client

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        try:
            response = self.client.post(
                "/api/v1/embeddings",
                json={"model": MODEL, "dimensions": DIMENSIONS, "input_type": "search_document", "input": texts},
            )
        except httpx.RequestError as error:
            raise EmbeddingFailure("provider_connection", "Embedding service could not be reached", retryable=True) from error
        if response.status_code >= 400:
            status = response.status_code
            raise EmbeddingFailure(
                f"provider_http_{status}",
                f"Embedding service returned HTTP {status}",
                retryable=status in {408, 429} or status >= 500,
            )

        try:
            payload = response.json()
            entries = payload["data"]
            ordered = sorted(entries, key=lambda record: record["index"])
            if [record["index"] for record in ordered] != list(range(len(texts))):
                raise ValueError("Unexpected response indexes")
            vectors = [record["embedding"] for record in ordered]
            for vector in vectors:
                if len(vector) != DIMENSIONS or not all(isfinite(value) for value in vector):
                    raise ValueError("Unexpected vector shape")
            tokens = int(payload.get("usage", {}).get("prompt_tokens", 0))
            if tokens < 0:
                raise ValueError("Invalid token count")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise EmbeddingFailure("provider_response", "Embedding service returned an invalid response", retryable=True) from error

        return EmbeddingBatch(vectors, tokens)
