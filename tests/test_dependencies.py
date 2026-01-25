"""Test that all required dependencies can be imported"""
import pytest

def test_fastapi_importable():
    """FastAPI can be imported"""
    import fastapi
    assert hasattr(fastapi, 'FastAPI')

def test_sqlalchemy_importable():
    """SQLAlchemy can be imported"""
    import sqlalchemy
    assert sqlalchemy.__version__

def test_pydantic_importable():
    """Pydantic can be imported"""
    import pydantic
    assert hasattr(pydantic, 'BaseModel')

def test_openai_importable():
    """OpenAI client can be imported"""
    from openai import OpenAI
    assert OpenAI

def test_langchain_importable():
    """LangChain can be imported"""
    import langchain
    assert langchain.__version__

def test_qdrant_importable():
    """Qdrant client can be imported"""
    from qdrant_client import QdrantClient
    assert QdrantClient
