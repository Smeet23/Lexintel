"""Integration tests for firm theme API endpoints."""

import pytest


def test_create_firm(client):
    resp = client.post("/api/firms", json={"name": "Test Law Firm"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Test Law Firm"
    assert data["slug"] == "test-law-firm"
    assert data["theme_config"] is not None
    assert "light" in data["theme_config"]
    assert "dark" in data["theme_config"]


def test_get_firm_theme(client):
    client.post("/api/firms", json={"name": "Theme Firm"})
    resp = client.get("/api/firms/theme-firm/theme")
    assert resp.status_code == 200
    data = resp.json()
    assert data["firm_name"] == "Theme Firm"
    assert data["firm_slug"] == "theme-firm"
    assert data["theme"]["light"]["background"] == "#FAFAF8"


def test_update_firm_theme_partial(client):
    client.post("/api/firms", json={"name": "Update Firm"})
    resp = client.put(
        "/api/firms/update-firm/theme",
        json={
            "light": {"primary": "#FF0000", "background": "#FFFFFF"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # Updated values applied
    assert data["theme"]["light"]["primary"] == "#FF0000"
    assert data["theme"]["light"]["background"] == "#FFFFFF"
    # Other values preserved from defaults
    assert data["theme"]["light"]["foreground"] == "#111111"
    # Dark mode untouched
    assert data["theme"]["dark"]["background"] == "#0A0A0A"


def test_reset_firm_theme(client):
    client.post("/api/firms", json={"name": "Reset Firm"})
    # Modify theme
    client.put(
        "/api/firms/reset-firm/theme",
        json={"light": {"primary": "#FF0000"}},
    )
    # Reset
    resp = client.post("/api/firms/reset-firm/theme/reset")
    assert resp.status_code == 200
    data = resp.json()
    # Should be back to default
    assert data["theme"]["light"]["primary"] == "#111111"


def test_get_firm_not_found(client):
    resp = client.get("/api/firms/nonexistent/theme")
    assert resp.status_code == 404


def test_theme_css_variables_match_tokens(client):
    """Verify all theme tokens follow the --app-* naming convention expected by CSS."""
    client.post("/api/firms", json={"name": "CSS Firm"})
    resp = client.get("/api/firms/css-firm/theme")
    data = resp.json()

    light = data["theme"]["light"]
    dark = data["theme"]["dark"]

    # All color values should be valid hex
    for mode_name, colors in [("light", light), ("dark", dark)]:
        for key, val in colors.items():
            assert val.startswith("#"), f"{mode_name}.{key} = {val} is not a hex color"
            assert len(val) == 7, f"{mode_name}.{key} = {val} is not #RRGGBB format"
