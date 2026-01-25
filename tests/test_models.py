"""Test ORM models"""
import pytest
from uuid import uuid4
from datetime import datetime
from sqlalchemy.orm import Session
from backend.models import User, Case, Chunk, Query, CaseStatus, Base


def test_user_model_creation(db: Session):
    """User model can be created"""
    user = User(
        email="lawyer@example.com",
        password_hash="hashed_password"
    )
    db.add(user)
    db.commit()

    retrieved = db.query(User).filter(User.email == "lawyer@example.com").first()
    assert retrieved is not None
    assert retrieved.email == "lawyer@example.com"
    assert retrieved.is_deleted == False


def test_case_model_with_user(db: Session):
    """Case model linked to User"""
    user = User(email="lawyer@example.com", password_hash="hash")
    db.add(user)
    db.commit()

    case = Case(
        user_id=user.id,
        name="Smith v. Jones",
        blob_storage_path="cases/abc123.pdf",
        status="processing"
    )
    db.add(case)
    db.commit()

    retrieved = db.query(Case).filter(Case.id == case.id).first()
    assert retrieved.user_id == user.id
    assert retrieved.status == "processing"
    assert retrieved.is_deleted == False


def test_chunk_model_with_embedding(db: Session):
    """Chunk model stores embedding metadata"""
    user = User(email="lawyer@example.com", password_hash="hash")
    db.add(user)
    db.commit()

    case = Case(
        user_id=user.id,
        name="Case 1",
        blob_storage_path="path.pdf",
        status="processing"
    )
    db.add(case)
    db.commit()

    chunk = Chunk(
        case_id=case.id,
        page_num="5",
        section_name="Arguments",
        content="The plaintiff argued that...",
        embedding_hash="sha256_hash_123",
        chunk_sequence=1
    )
    db.add(chunk)
    db.commit()

    retrieved = db.query(Chunk).filter(Chunk.id == chunk.id).first()
    assert retrieved.embedding_hash == "sha256_hash_123"
    assert retrieved.chunk_sequence == 1


def test_query_model_with_citations(db: Session):
    """Query model stores Q&A with citations"""
    user = User(email="lawyer@example.com", password_hash="hash")
    db.add(user)
    db.commit()

    case = Case(
        user_id=user.id,
        name="Case 1",
        blob_storage_path="path.pdf",
        status="ready"
    )
    db.add(case)
    db.commit()

    citations = [
        {"page": "5", "section": "Arguments", "content_snippet": "..."},
        {"page": "6", "section": "Judgment", "content_snippet": "..."}
    ]

    query = Query(
        case_id=case.id,
        user_id=user.id,
        question="What are the key arguments?",
        answer="The plaintiff argued that...",
        citations=citations
    )
    db.add(query)
    db.commit()

    retrieved = db.query(Query).filter(Query.id == query.id).first()
    assert len(retrieved.citations) == 2
    assert retrieved.citations[0]["page"] == "5"


def test_case_soft_delete(db: Session):
    """Case soft delete works"""
    user = User(email="lawyer@example.com", password_hash="hash")
    db.add(user)
    db.commit()

    case = Case(
        user_id=user.id,
        name="Case 1",
        blob_storage_path="path.pdf",
        status="ready"
    )
    db.add(case)
    db.commit()

    # Soft delete
    case.is_deleted = True
    db.commit()

    # Should not appear in active queries (when filtering)
    active_cases = db.query(Case).filter(Case.is_deleted == False).all()
    assert len(active_cases) == 0


def test_user_relationships(db: Session):
    """User can access related cases"""
    user = User(email="lawyer@example.com", password_hash="hash")
    db.add(user)
    db.commit()

    case1 = Case(user_id=user.id, name="Case 1", blob_storage_path="p1.pdf", status="ready")
    case2 = Case(user_id=user.id, name="Case 2", blob_storage_path="p2.pdf", status="processing")
    db.add_all([case1, case2])
    db.commit()

    retrieved_user = db.query(User).filter(User.id == user.id).first()
    assert len(retrieved_user.cases) == 2


def test_case_relationships(db: Session):
    """Case can access related chunks and queries"""
    user = User(email="lawyer@example.com", password_hash="hash")
    db.add(user)
    db.commit()

    case = Case(user_id=user.id, name="Case 1", blob_storage_path="p.pdf", status="ready")
    db.add(case)
    db.commit()

    chunk = Chunk(case_id=case.id, page_num="5", section_name="Args", content="text", embedding_hash="h1", chunk_sequence=1)
    query = Query(case_id=case.id, user_id=user.id, question="Q?", answer="A.", citations=[])
    db.add_all([chunk, query])
    db.commit()

    retrieved_case = db.query(Case).filter(Case.id == case.id).first()
    assert len(retrieved_case.chunks) == 1
    assert len(retrieved_case.queries) == 1
