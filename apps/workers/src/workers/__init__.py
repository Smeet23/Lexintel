"""Celery worker tasks."""

from .document_extraction import extract_text_from_document

__all__ = [
    "extract_text_from_document",
]
