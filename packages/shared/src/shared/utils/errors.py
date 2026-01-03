"""Shared error definitions for LexIntel."""


class LexIntelError(Exception):
    """Base exception for LexIntel."""
    pass


class PermanentError(LexIntelError):
    """Error that should not be retried."""
    pass


class RetryableError(LexIntelError):
    """Error that can be retried with backoff."""
    pass


class DocumentNotFound(PermanentError):
    """Document not found in database."""
    pass


class FileNotFound(PermanentError):
    """File not found in storage."""
    pass


class ExtractionFailed(PermanentError):
    """Document extraction failed."""
    pass
