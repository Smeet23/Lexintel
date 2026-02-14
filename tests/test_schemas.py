"""Test Pydantic schemas"""
import pytest
from datetime import datetime
from uuid import uuid4
from backend.schemas import (
    CaseCreate, CaseResponse,
    ChunkResponse,
    QueryCreate, QueryResponse
)


def test_case_create_validation():
    """CaseCreate validates name"""
    # Valid
    case = CaseCreate(name="Smith v. Jones")
    assert case.name == "Smith v. Jones"

    # Empty name
    with pytest.raises(ValueError):
        CaseCreate(name="")


def test_query_create_validation():
    """QueryCreate validates question"""
    # Valid
    query = QueryCreate(question="What is the judgment?")
    assert query.question == "What is the judgment?"

    # Empty question
    with pytest.raises(ValueError):
        QueryCreate(question="")


def test_case_response_includes_timestamps():
    """CaseResponse includes created_at and updated_at"""
    from backend.models import Case

    now = datetime.utcnow()
    case = Case(
        id=uuid4(),
        name="Case 1",
        blob_storage_path="path.pdf",
        status="ready",
        created_at=now,
        updated_at=now
    )

    response = CaseResponse.model_validate(case)
    assert response.name == "Case 1"
    assert response.status == "ready"
    assert response.created_at == now
