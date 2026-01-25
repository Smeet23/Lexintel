"""Test configuration loading from environment variables"""
import os
import pytest

def test_settings_load():
    """Settings can be loaded"""
    # Set minimal env vars
    os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost/test'
    os.environ['OPENAI_API_KEY'] = 'sk-test'
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = 'UseDevelopmentStorage=true'
    os.environ['SECRET_KEY'] = 'test-secret'

    from backend.config import get_settings
    settings = get_settings()

    assert settings.database_url == 'postgresql://test:test@localhost/test'
    assert settings.openai_api_key == 'sk-test'

def test_env_example_exists():
    """env.example file exists"""
    import os
    env_example_path = 'backend/.env.example'
    assert os.path.exists(env_example_path), f"{env_example_path} not found"
