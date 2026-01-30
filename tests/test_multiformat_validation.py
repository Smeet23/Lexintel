"""Tests for multi-format validation (validators and storage)"""
import pytest
import os
from fastapi import HTTPException

# Set required env vars before imports
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/test_lexintel.db')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-long-enough')

from backend.validators import validate_filename, validate_file_type
from backend.services.storage import validate_file_format


class TestValidateFilename:
    """Test filename validation for multiple formats"""

    def test_validate_filename_pdf(self):
        """Test that PDF filenames are accepted"""
        result = validate_filename("document.pdf")
        assert result == "document.pdf"

    def test_validate_filename_docx(self):
        """Test that DOCX filenames are accepted"""
        result = validate_filename("document.docx")
        assert result == "document.docx"

    def test_validate_filename_txt(self):
        """Test that TXT filenames are accepted"""
        result = validate_filename("document.txt")
        assert result == "document.txt"

    def test_validate_filename_case_insensitive(self):
        """Test that file extension matching is case-insensitive"""
        assert validate_filename("document.PDF")
        assert validate_filename("document.DOCX")
        assert validate_filename("document.TXT")

    def test_validate_filename_mixed_case(self):
        """Test mixed case extensions"""
        assert validate_filename("document.Pdf")
        assert validate_filename("document.DocX")

    def test_validate_filename_invalid_extension(self):
        """Test that invalid extensions are rejected"""
        with pytest.raises(HTTPException) as exc:
            validate_filename("document.docm")
        assert "Only" in str(exc.value.detail)

    def test_validate_filename_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked"""
        with pytest.raises(HTTPException):
            validate_filename("../../../etc/passwd.pdf")

    def test_validate_filename_tilde_blocked(self):
        """Test that tilde paths are blocked"""
        with pytest.raises(HTTPException):
            validate_filename("~/document.pdf")

    def test_validate_filename_absolute_path_blocked(self):
        """Test that absolute paths are blocked"""
        with pytest.raises(HTTPException):
            validate_filename("/etc/passwd.pdf")

    def test_validate_filename_too_long(self):
        """Test that filenames longer than 255 chars are rejected"""
        long_name = "a" * 256 + ".pdf"
        with pytest.raises(HTTPException) as exc:
            validate_filename(long_name)
        assert "too long" in str(exc.value.detail).lower()

    def test_validate_filename_empty(self):
        """Test that empty filename is rejected"""
        with pytest.raises(HTTPException):
            validate_filename("")


class TestValidateFileType:
    """Test file type detection"""

    def test_validate_file_type_pdf(self):
        """Test PDF detection"""
        result = validate_file_type("application/pdf", "document.pdf")
        assert result == "pdf"

    def test_validate_file_type_docx(self):
        """Test DOCX detection"""
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        result = validate_file_type(mime_type, "document.docx")
        assert result == "docx"

    def test_validate_file_type_txt(self):
        """Test TXT detection"""
        result = validate_file_type("text/plain", "document.txt")
        assert result == "txt"

    def test_validate_file_type_case_insensitive(self):
        """Test that extension matching is case-insensitive"""
        assert validate_file_type("application/pdf", "document.PDF") == "pdf"
        assert validate_file_type("application/pdf", "DOCUMENT.pdf") == "pdf"

    def test_validate_file_type_invalid(self):
        """Test that invalid file types raise ValueError"""
        with pytest.raises(ValueError):
            validate_file_type("application/pdf", "document.exe")

    def test_validate_file_type_ignores_mime_type(self):
        """Test that detection uses filename, not MIME type"""
        # Even if MIME type is wrong, filename extension is used
        result = validate_file_type("application/octet-stream", "document.docx")
        assert result == "docx"


class TestValidateFileFormat:
    """Test file content validation by magic bytes"""

    def test_validate_pdf_magic_bytes(self):
        """Test PDF magic bytes validation"""
        pdf_content = b"%PDF-1.4\nValid PDF content"
        assert validate_file_format(pdf_content, "pdf") is True

    def test_validate_pdf_invalid_magic_bytes(self):
        """Test PDF with invalid magic bytes"""
        invalid_pdf = b"Not a PDF\nThis is invalid"
        assert validate_file_format(invalid_pdf, "pdf") is False

    def test_validate_docx_magic_bytes(self):
        """Test DOCX (ZIP) magic bytes validation"""
        # DOCX is a ZIP file, starts with PK\x03\x04
        docx_content = b"PK\x03\x04\x14\x00\x00\x00Valid DOCX content"
        assert validate_file_format(docx_content, "docx") is True

    def test_validate_docx_invalid_magic_bytes(self):
        """Test DOCX with invalid magic bytes"""
        invalid_docx = b"Not a DOCX file\nThis is invalid"
        assert validate_file_format(invalid_docx, "docx") is False

    def test_validate_txt_utf8(self):
        """Test TXT UTF-8 validation"""
        txt_content = "This is valid UTF-8 text".encode('utf-8')
        assert validate_file_format(txt_content, "txt") is True

    def test_validate_txt_invalid_encoding(self):
        """Test TXT with invalid UTF-8"""
        invalid_txt = b"\x80\x81\x82\x83Invalid UTF-8"
        assert validate_file_format(invalid_txt, "txt") is False

    def test_validate_unsupported_format(self):
        """Test unsupported file format"""
        content = b"Some content"
        # Unsupported format should return False
        result = validate_file_format(content, "exe")
        assert result is False


class TestValidationIntegration:
    """Integration tests for validation workflow"""

    def test_full_validation_pdf(self):
        """Test full PDF validation flow"""
        filename = "legal_document.pdf"
        mime_type = "application/pdf"
        content = b"%PDF-1.4\nValid PDF"

        # Validate filename
        validated_name = validate_filename(filename)
        assert validated_name == filename

        # Detect file type
        file_type = validate_file_type(mime_type, filename)
        assert file_type == "pdf"

        # Validate format
        is_valid = validate_file_format(content, file_type)
        assert is_valid is True

    def test_full_validation_docx(self):
        """Test full DOCX validation flow"""
        filename = "contract.docx"
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        content = b"PK\x03\x04Valid DOCX"

        validated_name = validate_filename(filename)
        assert validated_name == filename

        file_type = validate_file_type(mime_type, filename)
        assert file_type == "docx"

        is_valid = validate_file_format(content, file_type)
        assert is_valid is True

    def test_full_validation_txt(self):
        """Test full TXT validation flow"""
        filename = "legal_text.txt"
        mime_type = "text/plain"
        content = "This is plain text document".encode('utf-8')

        validated_name = validate_filename(filename)
        assert validated_name == filename

        file_type = validate_file_type(mime_type, filename)
        assert file_type == "txt"

        is_valid = validate_file_format(content, file_type)
        assert is_valid is True

    def test_validation_rejects_spoofed_format(self):
        """Test that validation catches format spoofing"""
        # Someone tries to upload .exe as .pdf
        filename = "malicious.pdf"
        mime_type = "application/pdf"
        content = b"MZ\x90\x00Executable content"  # EXE magic bytes

        validated_name = validate_filename(filename)
        assert validated_name == filename

        file_type = validate_file_type(mime_type, filename)
        assert file_type == "pdf"

        # But content validation should catch it
        is_valid = validate_file_format(content, file_type)
        assert is_valid is False
