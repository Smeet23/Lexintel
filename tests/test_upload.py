"""Tests for file upload endpoints"""
import pytest
import os
from io import BytesIO
from unittest.mock import patch
from uuid import uuid4
from datetime import datetime, timezone
from tempfile import SpooledTemporaryFile

# Set required env vars before imports
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/test_lexintel_upload.db')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
os.environ.setdefault('DEBUG', 'True')

from backend.main import app
from backend.database import get_db
from backend.models import Base, Case
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def create_test_pdf() -> bytes:
    """Create minimal valid PDF for testing"""
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
214
%%EOF"""
    return pdf_content


def create_upload_file(filename: str, content: bytes, content_type: str = "application/pdf"):
    """Create a SpooledTemporaryFile based UploadFile"""
    from fastapi import UploadFile
    from starlette.datastructures import Headers

    temp_file = SpooledTemporaryFile(max_size=2621440)
    temp_file.write(content)
    temp_file.seek(0)

    headers = Headers({"content-type": content_type})
    return UploadFile(file=temp_file, filename=filename, headers=headers)


@pytest.fixture
def test_db_session():
    """Create test database session"""
    engine = create_engine(
        "sqlite:////tmp/test_lexintel_upload.db",
        connect_args={"check_same_thread": False}
    )

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    yield db

    db.close()
    Base.metadata.drop_all(bind=engine)


class TestUploadEndpoint:
    """Test case upload functionality"""

    def test_upload_pdf_success(self, test_db_session):
        """Test successful PDF upload"""
        pdf_content = create_test_pdf()

        # Mock dependencies
        def override_get_db():
            yield test_db_session

        app.dependency_overrides[get_db] = override_get_db

        # Mock blob storage upload at the main module level
        with patch('backend.main.upload_document_to_blob') as mock_upload:
            mock_upload.return_value = "uuid/test.pdf"

            import asyncio
            from backend.main import upload_case

            file = create_upload_file("test.pdf", pdf_content, "application/pdf")

            # Test the endpoint
            result = asyncio.run(upload_case(
                name="Test Case",
                file=file,
                db=test_db_session
            ))

            assert "id" in result
            assert result["name"] == "Test Case"
            assert result["status"] == "processing"
            assert "blob_storage_path" in result

        app.dependency_overrides.clear()

    def test_upload_non_pdf_rejected(self, test_db_session):
        """Test that unsupported files are rejected"""

        def override_get_db():
            yield test_db_session

        app.dependency_overrides[get_db] = override_get_db

        from backend.main import upload_case
        from fastapi.exceptions import HTTPException
        import asyncio

        file = create_upload_file("test.exe", b"not a document", "application/octet-stream")

        # Should raise HTTPException for unsupported type
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(upload_case(
                name="Test Case",
                file=file,
                db=test_db_session
            ))

        assert exc_info.value.status_code == 400

        app.dependency_overrides.clear()

    def test_upload_invalid_name(self, test_db_session):
        """Test upload with invalid case name"""
        pdf_content = create_test_pdf()

        def override_get_db():
            yield test_db_session

        app.dependency_overrides[get_db] = override_get_db

        from backend.main import upload_case
        from fastapi.exceptions import HTTPException
        import asyncio

        file = create_upload_file("test.pdf", pdf_content, "application/pdf")

        # Test with empty name
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(upload_case(
                name="",
                file=file,
                db=test_db_session
            ))

        assert exc_info.value.status_code == 400

        app.dependency_overrides.clear()

    def test_case_record_created(self, test_db_session):
        """Test that case record is created in database"""
        pdf_content = create_test_pdf()

        def override_get_db():
            yield test_db_session

        app.dependency_overrides[get_db] = override_get_db

        with patch('backend.main.upload_document_to_blob') as mock_upload:
            mock_upload.return_value = "test-uuid/test.pdf"

            import asyncio
            from backend.main import upload_case

            file = create_upload_file("test.pdf", pdf_content, "application/pdf")

            result = asyncio.run(upload_case(
                name="Database Test Case",
                file=file,
                db=test_db_session
            ))

            # Verify case exists in database
            from uuid import UUID
            case_id = UUID(result["id"])
            case = test_db_session.query(Case).filter(Case.id == case_id).first()
            assert case is not None
            assert case.name == "Database Test Case"
            assert case.status == "processing"

        app.dependency_overrides.clear()

    def test_upload_storage_error_handling(self, test_db_session):
        """Test that storage errors are properly handled"""
        pdf_content = create_test_pdf()

        def override_get_db():
            yield test_db_session

        app.dependency_overrides[get_db] = override_get_db

        with patch('backend.main.upload_document_to_blob') as mock_upload:
            # Simulate storage error
            mock_upload.side_effect = Exception("Storage connection failed")

            import asyncio
            from backend.main import upload_case
            from fastapi.exceptions import HTTPException

            file = create_upload_file("test.pdf", pdf_content, "application/pdf")

            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(upload_case(
                    name="Test Case",
                    file=file,
                    db=test_db_session
                ))

            assert exc_info.value.status_code == 500

        app.dependency_overrides.clear()
