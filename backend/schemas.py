"""Pydantic v2 schemas for request/response validation"""
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
from typing import Optional
from uuid import UUID
import re


# ============================================
# MATTER SCHEMAS
# ============================================

class MatterCreate(BaseModel):
    """Create matter request"""
    name: str = Field(..., min_length=1, max_length=255)


class MatterResponse(BaseModel):
    """Matter response (output)"""
    id: UUID
    name: str
    status: str
    file_type: str
    blob_storage_path: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================
# CHUNK SCHEMAS
# ============================================

class ChunkResponse(BaseModel):
    """Chunk response (output)"""
    id: UUID
    matter_id: UUID
    page_num: Optional[str] = None
    section_name: Optional[str] = None
    content: str
    embedding_hash: Optional[str] = None
    chunk_sequence: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================
# QUERY SCHEMAS
# ============================================

class QueryCreate(BaseModel):
    """Query request (ask question)"""
    question: str = Field(..., min_length=1, max_length=1000)


class CitationData(BaseModel):
    """Citation metadata"""
    page: str
    section: Optional[str] = None
    content_snippet: str
    score: Optional[float] = Field(None, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Query response (output)"""
    id: UUID
    question: str
    answer: str
    citations: list[CitationData] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================
# FIRM & THEME SCHEMAS
# ============================================

class FirmCreate(BaseModel):
    """Create firm request"""
    name: str = Field(..., min_length=1, max_length=255)
    slug: Optional[str] = Field(None, max_length=255, pattern=r'^[a-z0-9][a-z0-9-]*[a-z0-9]$')


class FirmResponse(BaseModel):
    """Firm response (output)"""
    id: UUID
    name: str
    slug: str
    logo_url: Optional[str] = None
    theme_config: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ThemeColorTokens(BaseModel):
    """Color tokens for a single mode (light or dark).
    All values must be valid #RRGGBB hex colors."""
    background: Optional[str] = None
    foreground: Optional[str] = None
    primary: Optional[str] = None
    primary_foreground: Optional[str] = Field(None, alias="primary-foreground")
    secondary: Optional[str] = None
    secondary_foreground: Optional[str] = Field(None, alias="secondary-foreground")
    accent: Optional[str] = None
    accent_foreground: Optional[str] = Field(None, alias="accent-foreground")
    muted: Optional[str] = None
    muted_foreground: Optional[str] = Field(None, alias="muted-foreground")
    destructive: Optional[str] = None
    destructive_foreground: Optional[str] = Field(None, alias="destructive-foreground")
    card: Optional[str] = None
    card_foreground: Optional[str] = Field(None, alias="card-foreground")
    popover: Optional[str] = None
    popover_foreground: Optional[str] = Field(None, alias="popover-foreground")
    border: Optional[str] = None
    input: Optional[str] = None
    ring: Optional[str] = None
    surface: Optional[str] = None
    surface_hover: Optional[str] = Field(None, alias="surface-hover")
    border_strong: Optional[str] = Field(None, alias="border-strong")
    primary_light: Optional[str] = Field(None, alias="primary-light")
    accent_hover: Optional[str] = Field(None, alias="accent-hover")
    accent_muted: Optional[str] = Field(None, alias="accent-muted")
    success: Optional[str] = None
    success_light: Optional[str] = Field(None, alias="success-light")
    warning: Optional[str] = None
    warning_light: Optional[str] = Field(None, alias="warning-light")
    sidebar_background: Optional[str] = Field(None, alias="sidebar-background")
    sidebar_foreground: Optional[str] = Field(None, alias="sidebar-foreground")
    sidebar_accent: Optional[str] = Field(None, alias="sidebar-accent")
    sidebar_border: Optional[str] = Field(None, alias="sidebar-border")
    chart_1: Optional[str] = Field(None, alias="chart-1")
    chart_2: Optional[str] = Field(None, alias="chart-2")
    chart_3: Optional[str] = Field(None, alias="chart-3")
    chart_4: Optional[str] = Field(None, alias="chart-4")
    chart_5: Optional[str] = Field(None, alias="chart-5")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator('*', mode='before')
    @classmethod
    def validate_hex_color(cls, v):
        if v is not None and isinstance(v, str):
            if not re.match(r'^#[0-9a-fA-F]{6}$', v):
                raise ValueError(f'Invalid hex color: {v}. Must be #RRGGBB format.')
        return v


class ThemeTypography(BaseModel):
    font_sans: Optional[str] = Field(None, alias="font-sans")
    font_display: Optional[str] = Field(None, alias="font-display")
    font_mono: Optional[str] = Field(None, alias="font-mono")
    model_config = ConfigDict(populate_by_name=True)


class ThemeLayout(BaseModel):
    radius_sm: Optional[str] = Field(None, alias="radius-sm")
    radius_md: Optional[str] = Field(None, alias="radius-md")
    radius_lg: Optional[str] = Field(None, alias="radius-lg")
    radius_xl: Optional[str] = Field(None, alias="radius-xl")
    model_config = ConfigDict(populate_by_name=True)


class ThemeShadows(BaseModel):
    color: Optional[str] = None
    opacity: Optional[str] = None
    blur: Optional[str] = None
    spread: Optional[str] = None
    offset_x: Optional[str] = Field(None, alias="offset-x")
    offset_y: Optional[str] = Field(None, alias="offset-y")
    model_config = ConfigDict(populate_by_name=True)


class ThemeConfigUpdate(BaseModel):
    """Full theme config update. light/dark are TYPED (not raw dict)
    so Pydantic validates hex colors on all incoming values."""
    light: Optional[ThemeColorTokens] = None
    dark: Optional[ThemeColorTokens] = None
    typography: Optional[ThemeTypography] = None
    layout: Optional[ThemeLayout] = None
    shadows: Optional[ThemeShadows] = None


class MemberInvite(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    name: Optional[str] = Field(None, max_length=255)
    role: str = Field(default="associate", pattern=r'^(admin|partner|associate|paralegal)$')


class MemberRoleUpdate(BaseModel):
    role: str = Field(..., pattern=r'^(admin|partner|associate|paralegal)$')
