# Functional Tabs & Precedents Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Contract Review, Draft Assistant, Audit Log, and Precedents fully functional with real backend endpoints and frontend integration.

**Architecture:** Four new DB tables (contract_reviews, drafts, audit_logs, saved_precedents) + 10 new API endpoints in main.py + a new `contract_review.py` service + a new `draft_service.py` service + audit log helper + frontend API services, hooks, and updated components. All Gemini calls reuse the existing pattern from `rag_engine.py` and `document_summary.py`.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Google Gemini, Cohere embeddings, Qdrant, Next.js 14, TanStack Query, TypeScript

---

### Task 1: Add new database models

**Files:**
- Modify: `backend/models.py`

**Step 1: Add ContractReview, Draft, AuditLog, SavedPrecedent models to models.py**

Add after the `ProcessingJob` class at the end of `backend/models.py`:

```python
class ContractReview(Base):
    """Contract risk analysis result for a document"""
    __tablename__ = "contract_reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    risks = Column(JSON, nullable=False, default=list)
    summary = Column(JSON, nullable=False, default=dict)
    missing_clauses = Column(JSON, nullable=False, default=list)
    overall_score = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<ContractReview(id={self.id}, matter_id={self.matter_id}, score={self.overall_score})>"


class Draft(Base):
    """Generated legal draft document"""
    __tablename__ = "drafts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    document_type = Column(String(100), nullable=False)
    instructions = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<Draft(id={self.id}, matter_id={self.matter_id}, type={self.document_type})>"


class AuditLog(Base):
    """Activity log entry for a matter"""
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    user = Column(String(255), nullable=False, default="System")
    details = Column(Text, nullable=True)
    sources = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    __table_args__ = (
        Index('idx_audit_matter_created', 'matter_id', 'created_at'),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, matter_id={self.matter_id})>"


class SavedPrecedent(Base):
    """User-bookmarked search result from cross-matter search"""
    __tablename__ = "saved_precedents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    query = Column(Text, nullable=False)
    document_name = Column(String(255), nullable=True)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=True)
    chunk_content = Column(Text, nullable=True)
    page_num = Column(Integer, nullable=True)
    section_name = Column(String(255), nullable=True)
    relevance_score = Column(String(10), nullable=True)
    tags = Column(JSON, nullable=True, default=list)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<SavedPrecedent(id={self.id}, title={self.title})>"
```

**Step 2: Commit**

```bash
git add backend/models.py
git commit -m "feat: add ContractReview, Draft, AuditLog, SavedPrecedent models"
```

---

### Task 2: Create Alembic migration

**Files:**
- Create: `backend/alembic/versions/9_add_functional_tabs_tables.py`

**Step 1: Create the migration file**

```python
"""Add contract_reviews, drafts, audit_logs, saved_precedents tables

Revision ID: 9
Revises: 8
Create Date: 2026-03-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision: str = "9"
down_revision: Union[str, None] = "8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "contract_reviews",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False, index=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False, index=True),
        sa.Column("risks", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("summary", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("missing_clauses", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("overall_score", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "drafts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False, index=True),
        sa.Column("document_type", sa.String(100), nullable=False),
        sa.Column("instructions", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("sources", sa.JSON, nullable=True, server_default="[]"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False, index=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("user", sa.String(255), nullable=False, server_default="System"),
        sa.Column("details", sa.Text, nullable=True),
        sa.Column("sources", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
    )
    op.create_index("idx_audit_matter_created", "audit_logs", ["matter_id", "created_at"])

    op.create_table(
        "saved_precedents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("document_name", sa.String(255), nullable=True),
        sa.Column("matter_id", UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("chunk_content", sa.Text, nullable=True),
        sa.Column("page_num", sa.Integer, nullable=True),
        sa.Column("section_name", sa.String(255), nullable=True),
        sa.Column("relevance_score", sa.String(10), nullable=True),
        sa.Column("tags", sa.JSON, nullable=True, server_default="[]"),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("saved_precedents")
    op.drop_table("audit_logs")
    op.drop_table("drafts")
    op.drop_table("contract_reviews")
```

**Step 2: Run migration**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && alembic upgrade head`

**Step 3: Commit**

```bash
git add backend/alembic/versions/9_add_functional_tabs_tables.py
git commit -m "feat: add migration for functional tabs tables"
```

---

### Task 3: Create audit log helper

**Files:**
- Create: `backend/services/audit.py`

**Step 1: Create the audit log helper**

```python
"""Audit log helper — fire-and-forget log entries for matter activity."""
import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session

try:
    from backend.models import AuditLog
except ImportError:
    try:
        from models import AuditLog
    except ImportError:
        from ..models import AuditLog

logger = logging.getLogger(__name__)


def log_activity(
    db: Session,
    matter_id: str,
    action: str,
    details: str | None = None,
    sources: str | None = None,
    user: str = "System",
) -> None:
    """Write an audit log entry. Commits independently — call AFTER the main transaction."""
    try:
        entry = AuditLog(
            id=uuid.uuid4(),
            matter_id=uuid.UUID(matter_id),
            action=action,
            user=user,
            details=details,
            sources=sources,
            created_at=datetime.now(timezone.utc),
        )
        db.add(entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Failed to write audit log: {e}")
```

**Step 2: Commit**

```bash
git add backend/services/audit.py
git commit -m "feat: add audit log helper service"
```

---

### Task 4: Create contract review service

**Files:**
- Create: `backend/services/contract_review.py`

**Step 1: Create the contract review service**

```python
"""Contract review service — analyze document chunks for legal risks using Gemini."""
import json
import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session

import google.generativeai as genai

try:
    from backend.config import get_settings
    from backend.models import Chunk, Document
except ImportError:
    try:
        from config import get_settings
        from models import Chunk, Document
    except ImportError:
        from ..config import get_settings
        from ..models import Chunk, Document

logger = logging.getLogger(__name__)

CONTRACT_REVIEW_PROMPT = """You are a legal contract risk analyst. Analyze the following document and return a JSON object with your analysis.

Evaluate each identifiable clause or section for risk level (high, medium, or low) from the perspective of the party receiving/signing this document.

Return ONLY valid JSON in this exact format:
{
  "risks": [
    {
      "clause": "Clause name with section reference",
      "risk_level": "high|medium|low",
      "explanation": "Brief explanation of the risk",
      "remedy": "Suggested improvement or remedy"
    }
  ],
  "summary": {
    "total_clauses": <number of clauses analyzed>,
    "high_risk": <count>,
    "medium_risk": <count>,
    "low_risk": <count>
  },
  "missing_clauses": ["List of standard legal clauses NOT found in this document"],
  "overall_score": <0-100, where 100 means very safe and 0 means very risky>
}

Common missing clauses to check for: Force Majeure, Limitation of Liability, Indemnification, Confidentiality, Non-Compete, Non-Solicitation, Audit Rights, Governing Law, Dispute Resolution, Termination for Cause, Termination for Convenience, Assignment, Data Protection/Privacy, Insurance Requirements, Warranties, Intellectual Property.

Document content:
"""


async def analyze_contract(
    matter_id: str,
    document_id: str,
    db: Session,
) -> Dict[str, Any]:
    """Analyze a document's chunks for contract risks using Gemini.

    Args:
        matter_id: Matter UUID string
        document_id: Document UUID string
        db: Database session

    Returns:
        Dict with risks, summary, missing_clauses, overall_score
    """
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)

    # Fetch chunks ordered by sequence
    chunks = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id, Chunk.matter_id == matter_id)
        .order_by(Chunk.chunk_sequence.asc().nullslast())
        .all()
    )

    if not chunks:
        return {
            "risks": [],
            "summary": {"total_clauses": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0},
            "missing_clauses": [],
            "overall_score": 0,
        }

    # Build document text from chunks (respect token budget ~50k chars)
    doc_text = ""
    for chunk in chunks:
        header = ""
        if chunk.section_name:
            header = f"\n## {chunk.section_name}\n"
        doc_text += header + chunk.content + "\n\n"
        if len(doc_text) > 50000:
            break

    # Call Gemini
    model = genai.GenerativeModel(model_name=settings.gemini_model)
    response = model.generate_content(
        CONTRACT_REVIEW_PROMPT + doc_text,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )

    # Parse JSON from response
    text = response.text.strip()
    # Handle markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Gemini returned invalid JSON for contract review: {text[:500]}")
        result = {
            "risks": [],
            "summary": {"total_clauses": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0},
            "missing_clauses": [],
            "overall_score": 50,
        }

    return result
```

**Step 2: Commit**

```bash
git add backend/services/contract_review.py
git commit -m "feat: add contract review Gemini analysis service"
```

---

### Task 5: Create draft generation service

**Files:**
- Create: `backend/services/draft_service.py`

**Step 1: Create the draft generation service**

```python
"""Draft generation service — generate legal documents using matter context and Gemini."""
import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session

import google.generativeai as genai

try:
    from backend.config import get_settings
    from backend.services.embeddings import embed_query as embed_query_fn
    from backend.services.vector_store import search_vectors
    from backend.models import Chunk, Document
except ImportError:
    try:
        from config import get_settings
        from services.embeddings import embed_query as embed_query_fn
        from services.vector_store import search_vectors
        from models import Chunk, Document
    except ImportError:
        from ..config import get_settings
        from .embeddings import embed_query as embed_query_fn
        from .vector_store import search_vectors
        from ..models import Chunk, Document

logger = logging.getLogger(__name__)

DRAFT_SYSTEM_PROMPT = """You are an expert legal drafting assistant. Generate a {document_type} based on the provided source material and user instructions.

Requirements:
1. Write in formal legal language appropriate for the document type
2. Include inline source references in the format [Source: document name, Page X] where you rely on specific source material
3. Structure the document with appropriate headings and sections for a {document_type}
4. Be thorough but concise
5. Flag any areas where additional information may be needed with [NOTE: ...]

Source Material:
{context}

User Instructions:
{instructions}

Generate the {document_type}:"""


async def generate_draft(
    matter_id: str,
    document_type: str,
    instructions: str,
    db: Session,
) -> Dict[str, Any]:
    """Generate a legal draft using matter context and Gemini.

    Args:
        matter_id: Matter UUID string
        document_type: Type of document to generate
        instructions: User instructions for the draft
        db: Database session

    Returns:
        Dict with content and sources
    """
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key)

    # Embed the instructions to find relevant chunks
    query_embedding = embed_query_fn(instructions)

    # Search for relevant chunks across the matter
    search_results = search_vectors(
        collection_name=matter_id,
        query_vector=query_embedding,
        limit=10,
    )

    # Build context from retrieved chunks
    sources = []
    context_parts = []
    for result in search_results:
        payload = result.payload or {}
        doc_name = payload.get("document_name", "Unknown")
        page = payload.get("page_num", "?")
        section = payload.get("section_name", "")
        content = payload.get("content", "")

        context_parts.append(f"[{doc_name}, Page {page}]{f' - {section}' if section else ''}:\n{content}")
        sources.append({
            "document_name": doc_name,
            "page_num": str(page),
            "section_name": section,
            "excerpt": content[:200],
        })

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No source material available."

    # Generate draft with Gemini
    prompt = DRAFT_SYSTEM_PROMPT.format(
        document_type=document_type,
        context=context,
        instructions=instructions,
    )

    model = genai.GenerativeModel(model_name=settings.gemini_model)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=8192,
        ),
    )

    return {
        "content": response.text,
        "sources": sources,
    }
```

**Step 2: Commit**

```bash
git add backend/services/draft_service.py
git commit -m "feat: add draft generation service with Gemini"
```

---

### Task 6: Add all backend API endpoints

**Files:**
- Modify: `backend/main.py`

**Step 1: Add imports at top of main.py**

Add to the import blocks (inside the first try block, after the existing imports):

```python
    from backend.models import Matter, Chunk, Query, Document, ContractReview, Draft, AuditLog, SavedPrecedent
    from backend.services.contract_review import analyze_contract
    from backend.services.draft_service import generate_draft
    from backend.services.audit import log_activity
    from backend.services.embeddings import embed_query as embed_query_fn
    from backend.services.vector_store import search_vectors
```

Also add the same to the other two fallback import blocks.

**Step 2: Add audit log calls to existing endpoints**

After the `db.commit()` in `upload_matter` (around line 243), add:
```python
        log_activity(db, str(matter_id), "matter_created", details=f"Created matter '{name}' with {len(files)} document(s)")
```

After the `db.commit()` in `upload_matter_document` (around line 527), add:
```python
        log_activity(db, str(matter_uuid), "document_uploaded", details=f"Uploaded '{file.filename}'")
```

After the `db.commit()` in `delete_document` (around line 673), add:
```python
    log_activity(db, str(matter_uuid), "document_deleted", details=f"Deleted document")
```

After the `db.add(db_query)` / `db.commit()` in `ask_question` (around line 780), add:
```python
            log_activity(db, str(matter_uuid), "query_asked", details=question)
```

After the `db.commit()` in `cancel_matter_processing` (around line 366), add:
```python
    log_activity(db, str(matter_uuid), "matter_cancelled", details="Processing cancelled")
```

**Step 3: Add Contract Review endpoints**

Add after the SSE Progress section, before `if __name__`:

```python
# ============================================
# CONTRACT REVIEW ENDPOINTS
# ============================================

@app.post("/matters/{matter_id}/contract-review", response_model=dict)
async def run_contract_review(
    matter_id: str,
    document_id: str = Body(None, embed=True),
    db: Session = Depends(get_db)
):
    """Run contract risk analysis on a document using Gemini."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    # Resolve document_id: use provided or first document
    if document_id:
        try:
            doc_uuid = UUID(document_id)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document ID")
    else:
        first_doc = db.query(Document).filter(Document.matter_id == matter_uuid).order_by(Document.created_at.asc()).first()
        if not first_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents found for this matter")
        doc_uuid = first_doc.id

    # Delete any existing review for this document (re-run)
    db.query(ContractReview).filter(
        ContractReview.matter_id == matter_uuid,
        ContractReview.document_id == doc_uuid
    ).delete()
    db.commit()

    try:
        result = await analyze_contract(str(matter_uuid), str(doc_uuid), db)
    except Exception as e:
        logger.error(f"Contract review failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Contract review analysis failed")

    # Store result
    review = ContractReview(
        id=uuid.uuid4(),
        matter_id=matter_uuid,
        document_id=doc_uuid,
        risks=result.get("risks", []),
        summary=result.get("summary", {}),
        missing_clauses=result.get("missing_clauses", []),
        overall_score=result.get("overall_score"),
    )
    db.add(review)
    db.commit()

    log_activity(db, str(matter_uuid), "contract_review_run", details=f"Analyzed document for contract risks")

    return {
        "id": str(review.id),
        "matter_id": str(matter_uuid),
        "document_id": str(doc_uuid),
        "risks": review.risks,
        "summary": review.summary,
        "missing_clauses": review.missing_clauses,
        "overall_score": review.overall_score,
        "created_at": review.created_at.isoformat(),
    }


@app.get("/matters/{matter_id}/contract-review", response_model=dict)
async def get_contract_review(
    matter_id: str,
    document_id: str = QueryParam(None),
    db: Session = Depends(get_db)
):
    """Get cached contract review for a matter/document."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    query = db.query(ContractReview).filter(ContractReview.matter_id == matter_uuid)
    if document_id:
        try:
            doc_uuid = UUID(document_id)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document ID")
        query = query.filter(ContractReview.document_id == doc_uuid)

    review = query.order_by(ContractReview.created_at.desc()).first()
    if not review:
        return {"exists": False}

    return {
        "exists": True,
        "id": str(review.id),
        "matter_id": str(review.matter_id),
        "document_id": str(review.document_id),
        "risks": review.risks,
        "summary": review.summary,
        "missing_clauses": review.missing_clauses,
        "overall_score": review.overall_score,
        "created_at": review.created_at.isoformat(),
    }


# ============================================
# DRAFT ASSISTANT ENDPOINTS
# ============================================

@app.post("/matters/{matter_id}/drafts", response_model=dict)
async def create_draft(
    matter_id: str,
    document_type: str = Body(..., embed=True),
    instructions: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Generate a legal draft using matter context."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    if not document_type or not instructions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_type and instructions are required")

    try:
        result = await generate_draft(str(matter_uuid), document_type, instructions, db)
    except Exception as e:
        logger.error(f"Draft generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Draft generation failed")

    draft = Draft(
        id=uuid.uuid4(),
        matter_id=matter_uuid,
        document_type=document_type,
        instructions=instructions,
        content=result["content"],
        sources=result.get("sources", []),
    )
    db.add(draft)
    db.commit()

    log_activity(db, str(matter_uuid), "draft_generated", details=f"Generated {document_type}")

    return {
        "id": str(draft.id),
        "matter_id": str(matter_uuid),
        "document_type": draft.document_type,
        "instructions": draft.instructions,
        "content": draft.content,
        "sources": draft.sources,
        "created_at": draft.created_at.isoformat(),
    }


@app.get("/matters/{matter_id}/drafts", response_model=list)
async def list_drafts(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """List all drafts for a matter."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    drafts = (
        db.query(Draft)
        .filter(Draft.matter_id == matter_uuid)
        .order_by(Draft.created_at.desc())
        .all()
    )

    return [
        {
            "id": str(d.id),
            "document_type": d.document_type,
            "instructions": d.instructions,
            "content": d.content,
            "sources": d.sources,
            "created_at": d.created_at.isoformat(),
        }
        for d in drafts
    ]


@app.get("/matters/{matter_id}/drafts/{draft_id}", response_model=dict)
async def get_draft(
    matter_id: str,
    draft_id: str,
    db: Session = Depends(get_db)
):
    """Get a single draft."""
    try:
        matter_uuid = UUID(matter_id)
        draft_uuid = UUID(draft_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format")

    draft = db.query(Draft).filter(Draft.id == draft_uuid, Draft.matter_id == matter_uuid).first()
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")

    return {
        "id": str(draft.id),
        "matter_id": str(draft.matter_id),
        "document_type": draft.document_type,
        "instructions": draft.instructions,
        "content": draft.content,
        "sources": draft.sources,
        "created_at": draft.created_at.isoformat(),
    }


# ============================================
# AUDIT LOG ENDPOINT
# ============================================

@app.get("/matters/{matter_id}/audit-log", response_model=list)
async def get_audit_log(
    matter_id: str,
    limit: int = QueryParam(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get activity log for a matter."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    logs = (
        db.query(AuditLog)
        .filter(AuditLog.matter_id == matter_uuid)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": str(log.id),
            "action": log.action,
            "user": log.user,
            "details": log.details,
            "sources": log.sources,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


# ============================================
# PRECEDENTS ENDPOINTS
# ============================================

@app.post("/precedents/search", response_model=dict)
async def search_precedents(
    query: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Search across all matters for relevant legal precedents."""
    if not query or len(query) < 3:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query must be at least 3 characters")

    # Get all non-deleted matters to search across
    matters = db.query(Matter).filter(Matter.is_deleted == False, Matter.status == "ready").all()
    if not matters:
        return {"results": [], "total": 0}

    # Embed the query
    try:
        query_embedding = embed_query_fn(query)
    except Exception as e:
        logger.error(f"Embedding failed for precedent search: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed")

    # Search across all matter collections
    all_results = []
    for matter in matters:
        try:
            results = search_vectors(
                collection_name=str(matter.id),
                query_vector=query_embedding,
                limit=5,
            )
            for result in results:
                payload = result.payload or {}
                all_results.append({
                    "matter_id": str(matter.id),
                    "matter_name": matter.name,
                    "document_name": payload.get("document_name", "Unknown"),
                    "page_num": payload.get("page_num", "?"),
                    "section_name": payload.get("section_name", ""),
                    "content": payload.get("content", ""),
                    "relevance_score": round(result.score, 3) if result.score else 0,
                })
        except Exception as e:
            logger.warning(f"Search failed for matter {matter.id}: {e}")
            continue

    # Sort by relevance and take top 20
    all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_results = all_results[:20]

    return {"results": top_results, "total": len(top_results)}


@app.post("/precedents/save", response_model=dict)
async def save_precedent(
    title: str = Body(..., embed=True),
    query: str = Body(..., embed=True),
    document_name: str = Body(None, embed=True),
    matter_id: str = Body(None, embed=True),
    chunk_content: str = Body(None, embed=True),
    page_num: int = Body(None, embed=True),
    section_name: str = Body(None, embed=True),
    relevance_score: str = Body(None, embed=True),
    tags: list = Body([], embed=True),
    notes: str = Body(None, embed=True),
    db: Session = Depends(get_db)
):
    """Save a search result as a precedent."""
    matter_uuid = None
    if matter_id:
        try:
            matter_uuid = UUID(matter_id)
        except ValueError:
            pass

    precedent = SavedPrecedent(
        id=uuid.uuid4(),
        title=title,
        query=query,
        document_name=document_name,
        matter_id=matter_uuid,
        chunk_content=chunk_content,
        page_num=page_num,
        section_name=section_name,
        relevance_score=relevance_score,
        tags=tags,
        notes=notes,
    )
    db.add(precedent)
    db.commit()

    return {
        "id": str(precedent.id),
        "title": precedent.title,
        "created_at": precedent.created_at.isoformat(),
    }


@app.get("/precedents", response_model=list)
async def list_precedents(
    db: Session = Depends(get_db)
):
    """List all saved precedents."""
    precedents = db.query(SavedPrecedent).order_by(SavedPrecedent.created_at.desc()).all()

    return [
        {
            "id": str(p.id),
            "title": p.title,
            "query": p.query,
            "document_name": p.document_name,
            "matter_id": str(p.matter_id) if p.matter_id else None,
            "chunk_content": p.chunk_content,
            "page_num": p.page_num,
            "section_name": p.section_name,
            "relevance_score": p.relevance_score,
            "tags": p.tags or [],
            "notes": p.notes,
            "created_at": p.created_at.isoformat(),
        }
        for p in precedents
    ]


@app.delete("/precedents/{precedent_id}", response_model=dict)
async def delete_precedent(
    precedent_id: str,
    db: Session = Depends(get_db)
):
    """Delete a saved precedent."""
    try:
        p_uuid = UUID(precedent_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format")

    precedent = db.query(SavedPrecedent).filter(SavedPrecedent.id == p_uuid).first()
    if not precedent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Precedent not found")

    db.delete(precedent)
    db.commit()
    return {"id": str(p_uuid), "deleted": True}
```

**Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat: add contract review, draft, audit log, and precedents endpoints"
```

---

### Task 7: Add frontend API services and types

**Files:**
- Modify: `frontend/lib/api-services.ts`
- Modify: `frontend/lib/types.ts`

**Step 1: Add new types to `frontend/lib/types.ts`**

Append to the end of the file:

```typescript
export interface ContractRisk {
  clause: string
  risk_level: "high" | "medium" | "low"
  explanation: string
  remedy: string
}

export interface ContractReviewResponse {
  exists: boolean
  id?: string
  matter_id?: string
  document_id?: string
  risks?: ContractRisk[]
  summary?: {
    total_clauses: number
    high_risk: number
    medium_risk: number
    low_risk: number
  }
  missing_clauses?: string[]
  overall_score?: number
  created_at?: string
}

export interface DraftResponse {
  id: string
  document_type: string
  instructions: string
  content: string
  sources: { document_name: string; page_num: string; section_name: string; excerpt: string }[]
  created_at: string
}

export interface AuditLogEntry {
  id: string
  action: string
  user: string
  details: string | null
  sources: string | null
  created_at: string
}

export interface PrecedentSearchResult {
  matter_id: string
  matter_name: string
  document_name: string
  page_num: string
  section_name: string
  content: string
  relevance_score: number
}

export interface SavedPrecedent {
  id: string
  title: string
  query: string
  document_name: string | null
  matter_id: string | null
  chunk_content: string | null
  page_num: number | null
  section_name: string | null
  relevance_score: string | null
  tags: string[]
  notes: string | null
  created_at: string
}
```

**Step 2: Add new API functions to `frontend/lib/api-services.ts`**

Append to the end of the file (before the SSE section or at the very end):

```typescript
// ============================================
// Contract Review API Functions
// ============================================

export async function getContractReview(
  matterId: string,
  documentId?: string
): Promise<ContractReviewResponse> {
  const params = documentId ? { document_id: documentId } : {}
  const { data } = await api.get<ContractReviewResponse>(
    `/matters/${matterId}/contract-review`,
    { params }
  )
  return data
}

export async function runContractReview(
  matterId: string,
  documentId?: string
): Promise<ContractReviewResponse> {
  const { data } = await api.post<ContractReviewResponse>(
    `/matters/${matterId}/contract-review`,
    { document_id: documentId || null }
  )
  return data
}

// ============================================
// Draft Assistant API Functions
// ============================================

export async function createDraft(
  matterId: string,
  documentType: string,
  instructions: string
): Promise<DraftResponse> {
  const { data } = await api.post<DraftResponse>(
    `/matters/${matterId}/drafts`,
    { document_type: documentType, instructions }
  )
  return data
}

export async function listDrafts(matterId: string): Promise<DraftResponse[]> {
  const { data } = await api.get<DraftResponse[]>(`/matters/${matterId}/drafts`)
  return data
}

export async function getDraft(matterId: string, draftId: string): Promise<DraftResponse> {
  const { data } = await api.get<DraftResponse>(`/matters/${matterId}/drafts/${draftId}`)
  return data
}

// ============================================
// Audit Log API Functions
// ============================================

export async function getAuditLog(matterId: string, limit = 100): Promise<AuditLogEntry[]> {
  const { data } = await api.get<AuditLogEntry[]>(
    `/matters/${matterId}/audit-log`,
    { params: { limit } }
  )
  return data
}

// ============================================
// Precedents API Functions
// ============================================

export async function searchPrecedents(query: string): Promise<{ results: PrecedentSearchResult[]; total: number }> {
  const { data } = await api.post<{ results: PrecedentSearchResult[]; total: number }>(
    "/precedents/search",
    { query }
  )
  return data
}

export async function savePrecedent(params: {
  title: string
  query: string
  document_name?: string
  matter_id?: string
  chunk_content?: string
  page_num?: number
  section_name?: string
  relevance_score?: string
  tags?: string[]
  notes?: string
}): Promise<{ id: string; title: string; created_at: string }> {
  const { data } = await api.post("/precedents/save", params)
  return data
}

export async function listSavedPrecedents(): Promise<SavedPrecedent[]> {
  const { data } = await api.get<SavedPrecedent[]>("/precedents")
  return data
}

export async function deletePrecedent(id: string): Promise<{ id: string; deleted: boolean }> {
  const { data } = await api.delete(`/precedents/${id}`)
  return data
}
```

Add the necessary import at the top of api-services.ts:

```typescript
import type { ChunkResponse, ContractReviewResponse, DraftResponse, AuditLogEntry, PrecedentSearchResult, SavedPrecedent } from "./types"
```

**Step 3: Commit**

```bash
git add frontend/lib/api-services.ts frontend/lib/types.ts
git commit -m "feat: add frontend API services for contract review, drafts, audit log, precedents"
```

---

### Task 8: Add frontend React Query hooks

**Files:**
- Modify: `frontend/hooks/use-matters.ts`

**Step 1: Add new hooks at the end of `use-matters.ts`**

Add the new imports at the top of the file:

```typescript
import {
  // ... existing imports ...
  getContractReview,
  runContractReview,
  createDraft,
  listDrafts,
  getAuditLog,
  searchPrecedents,
  savePrecedent,
  listSavedPrecedents,
  deletePrecedent,
  type ContractReviewResponse,
  type DraftResponse,
  type AuditLogEntry,
  type PrecedentSearchResult,
  type SavedPrecedent,
} from "@/lib/api-services"
// Also import from types
import type { ChunkResponse, ContractReviewResponse as CRType } from "@/lib/types"
```

Add at the end of the file:

```typescript
// ============================================
// Contract Review Hooks
// ============================================

export function useContractReview(matterId: string, documentId?: string) {
  return useQuery<ContractReviewResponse>({
    queryKey: ["matters", matterId, "contract-review", documentId ?? "default"],
    queryFn: () => getContractReview(matterId, documentId),
    enabled: !!matterId,
  })
}

export function useRunContractReview(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (documentId?: string) => runContractReview(matterId, documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "contract-review"] })
    },
  })
}

// ============================================
// Draft Hooks
// ============================================

export function useDrafts(matterId: string) {
  return useQuery<DraftResponse[]>({
    queryKey: ["matters", matterId, "drafts"],
    queryFn: () => listDrafts(matterId),
    enabled: !!matterId,
  })
}

export function useCreateDraft(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ documentType, instructions }: { documentType: string; instructions: string }) =>
      createDraft(matterId, documentType, instructions),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "drafts"] })
    },
  })
}

// ============================================
// Audit Log Hook
// ============================================

export function useAuditLog(matterId: string) {
  return useQuery<AuditLogEntry[]>({
    queryKey: ["matters", matterId, "audit-log"],
    queryFn: () => getAuditLog(matterId),
    enabled: !!matterId,
  })
}

// ============================================
// Precedents Hooks
// ============================================

export function useSearchPrecedents() {
  return useMutation({
    mutationFn: (query: string) => searchPrecedents(query),
  })
}

export function useSavedPrecedents() {
  return useQuery<SavedPrecedent[]>({
    queryKey: ["precedents"],
    queryFn: listSavedPrecedents,
  })
}

export function useSavePrecedent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: savePrecedent,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["precedents"] })
    },
  })
}

export function useDeletePrecedent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => deletePrecedent(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["precedents"] })
    },
  })
}
```

**Step 2: Commit**

```bash
git add frontend/hooks/use-matters.ts
git commit -m "feat: add React Query hooks for contract review, drafts, audit log, precedents"
```

---

### Task 9: Update Matter Workspace page — wire Contract Review tab

**Files:**
- Modify: `frontend/app/matters/[id]/page.tsx`

**Step 1: Replace the mock data and contract review tab content**

Remove `contractRisks` and `mockAuditLog` constants. Replace the Contract Review `<TabsContent>` with real data fetched via hooks. The tab should:
- Call `useContractReview(matterId)` to check for cached review
- Show "Run Analysis" button if no review exists
- Show `useRunContractReview(matterId)` mutation for triggering analysis
- Display real risks, summary, missing clauses, overall_score from the API response
- Show loading spinner while analysis runs
- Keep the same visual layout (risk cards left, summary right)

**Step 2: Commit**

```bash
git add frontend/app/matters/[id]/page.tsx
git commit -m "feat: wire Contract Review tab to real backend API"
```

---

### Task 10: Update Matter Workspace page — wire Draft Assistant tab

**Files:**
- Modify: `frontend/app/matters/[id]/page.tsx`

**Step 1: Replace the Draft Assistant tab content**

Wire the Draft Assistant tab to use `useCreateDraft(matterId)` and `useDrafts(matterId)`. The tab should:
- Left panel: form (document type dropdown, instructions textarea, Generate button) + list of past drafts
- Right panel: selected draft content display
- Show loading state while draft generates
- Past drafts clickable to view content
- Copy button and download as .txt for generated content

**Step 2: Commit**

```bash
git add frontend/app/matters/[id]/page.tsx
git commit -m "feat: wire Draft Assistant tab to real backend API"
```

---

### Task 11: Update Matter Workspace page — wire Audit Log tab

**Files:**
- Modify: `frontend/app/matters/[id]/page.tsx`

**Step 1: Replace audit log mock data with real API call**

Wire the Audit Log tab to use `useAuditLog(matterId)`. Replace `mockAuditLog` references with real data. Keep the same DataTable layout. Map action types to badge variants. Show "No activity yet" when empty.

**Step 2: Commit**

```bash
git add frontend/app/matters/[id]/page.tsx
git commit -m "feat: wire Audit Log tab to real backend API"
```

---

### Task 12: Rewrite Precedents page with real search + save

**Files:**
- Modify: `frontend/app/precedents/page.tsx`

**Step 1: Rewrite the precedents page**

Replace all mock data with real API calls. The page should have two tabs:
- **Search tab:** search input + "Search" button → `useSearchPrecedents()` mutation → results grouped by matter. Each result shows document name, section, relevance score, content excerpt, and a "Save" button that opens a dialog with title/tags/notes fields.
- **Saved tab:** `useSavedPrecedents()` query → list of saved precedents with tags, notes, delete button. Filter by tags.

Remove the `mockPrecedents` array entirely.

**Step 2: Commit**

```bash
git add frontend/app/precedents/page.tsx
git commit -m "feat: rewrite Precedents page with real cross-matter search and save"
```

---

### Task 13: Test the full flow

**Step 1: Start services**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && docker-compose up -d`

**Step 2: Run migration**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && alembic upgrade head`

**Step 3: Test Contract Review**

1. Open the matter workspace in the browser
2. Click "Contract Review" tab
3. Click "Run Analysis"
4. Verify risks, summary, missing clauses, and score appear
5. Refresh page — verify cached result loads without re-analyzing

**Step 4: Test Draft Assistant**

1. Click "Draft Assistant" tab
2. Select "Legal Memo" as document type
3. Enter instructions: "Summarize the key patent infringement claims"
4. Click "Generate Draft"
5. Verify draft content appears with source references
6. Verify draft appears in the past drafts list

**Step 5: Test Audit Log**

1. Click "Audit Log" tab
2. Verify real activity entries appear (matter_created, document_uploaded, query_asked from earlier testing)
3. Ask a new question in Ask AI tab, then check audit log again

**Step 6: Test Precedents**

1. Go to Precedents page in sidebar
2. Search for "patent infringement"
3. Verify results from your uploaded matter appear
4. Click "Save" on a result, add tags and notes
5. Switch to Saved tab, verify it appears
6. Delete it, verify it's removed

**Step 7: Final commit**

```bash
git add -A
git commit -m "feat: all functional tabs and precedents working end-to-end"
```
