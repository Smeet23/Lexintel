"""
Text extraction and chunking service

Handles:
- Extracting text from various file formats (TXT, PDF, DOCX)
- Cleaning extracted text
- Splitting text into overlapping chunks
- Creating DocumentChunk database records
"""

import logging
from typing import List
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from app.models import DocumentChunk

logger = logging.getLogger(__name__)
