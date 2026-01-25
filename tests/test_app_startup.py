"""Test that FastAPI app can start"""
import pytest
import os

def test_app_can_instantiate():
    """FastAPI app can be instantiated"""
    # Set required env vars
    os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost/test')
    os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
    os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
    os.environ.setdefault('SECRET_KEY', 'test-secret')
    os.environ.setdefault('DEBUG', 'True')

    from backend.main import app
    assert app is not None

def test_health_endpoint_exists():
    """Health check endpoint exists"""
    os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost/test')
    os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
    os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
    os.environ.setdefault('SECRET_KEY', 'test-secret')
    os.environ.setdefault('DEBUG', 'True')

    from backend.main import app

    # Verify the health endpoint is registered
    routes = [route.path for route in app.routes]
    assert "/health" in routes
