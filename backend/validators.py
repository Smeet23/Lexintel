"""Input validation utilities"""
from fastapi import HTTPException, status
import re


def validate_filename(filename: str) -> str:
    """
    Validate filename for security and constraints

    - Blocks path traversal attempts (../, ~/, etc.)
    - Enforces max length of 255 characters
    - Only allows PDF files

    Args:
        filename: The filename to validate

    Returns:
        The validated filename

    Raises:
        HTTPException: If validation fails
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    # Check for path traversal attacks
    if ".." in filename or filename.startswith("~") or filename.startswith("/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path traversal not allowed"
        )

    # Check length
    if len(filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename too long (max 255 characters)"
        )

    # Check extension
    allowed_extensions = [".pdf", ".docx", ".txt"]
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only {', '.join(allowed_extensions)} files are allowed"
        )

    return filename


def validate_question(question: str) -> str:
    """
    Validate question for constraints

    - Min 3 characters
    - Max 5000 characters

    Args:
        question: The question to validate

    Returns:
        The validated question

    Raises:
        HTTPException: If validation fails
    """
    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required"
        )

    if len(question) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must be at least 3 characters long"
        )

    if len(question) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not exceed 5000 characters"
        )

    return question.strip()


def validate_file_type(content_type: str, filename: str) -> str:
    """
    Validate and return file type based on filename extension.

    Args:
        content_type: MIME type from request (not used for this implementation)
        filename: Original filename from upload

    Returns:
        File type string: 'pdf', 'docx', or 'txt'

    Raises:
        ValueError: If file type is unsupported
    """
    filename_lower = filename.lower()
    if filename_lower.endswith('.pdf'):
        return 'pdf'
    elif filename_lower.endswith('.docx'):
        return 'docx'
    elif filename_lower.endswith('.txt'):
        return 'txt'
    else:
        raise ValueError(f"Unsupported file type for file: {filename}")


def validate_matter_name(name: str) -> str:
    """
    Validate matter name for constraints

    - Min 1 character
    - Max 255 characters

    Args:
        name: The matter name to validate

    Returns:
        The validated matter name

    Raises:
        HTTPException: If validation fails
    """
    if not name or not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Matter name is required"
        )

    if len(name) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Matter name must not exceed 255 characters"
        )

    return name.strip()
