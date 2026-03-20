"""Tests for CourtListener configuration"""
import pytest


def test_settings_has_courtlistener_token():
    """Settings class should have courtlistener_api_token field"""
    from backend.config import Settings

    fields = Settings.model_fields
    assert "courtlistener_api_token" in fields


def test_settings_courtlistener_token_default():
    """CourtListener token should default to empty string (optional)"""
    from backend.config import Settings

    fields = Settings.model_fields
    assert fields["courtlistener_api_token"].default == ""
