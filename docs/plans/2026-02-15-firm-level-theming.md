# Firm-Level Theming Implementation Plan (v3 — Codebase-Validated)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable each law firm to fully customize their Lexintel UI — colors, typography, logo, shadows, border radius — through a comprehensive visual theme editor.

**Architecture:** Refactor `globals.css` to a two-layer CSS variable system (`@layer theme` + `@theme inline`) so Tailwind v4 utility classes reference `var()` functions that can be overridden at runtime. Add a `Firm` model with JSONB `theme_config`. Build a `FirmThemeProvider` that applies CSS variables via `setProperty()`. Extract shared page views into components and create firm-scoped routes under `/firm/:slug/`. Build a full theme editor at `/firm/:slug/settings/theme`.

**Tech Stack:** Next.js 14 (synchronous params), shadcn/ui (new-york style), Tailwind CSS v4 (`@theme inline` + `@layer theme`), FastAPI, SQLAlchemy 2.0 with JSONB, PostgreSQL, Alembic, React Query, Framer Motion

---

## Issues Found in v2 & Fixed in v3

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | **Users table already exists** — initial migration `f794e7d74f24` creates `users(id, email, password_hash, is_deleted, created_at, updated_at)`. Task 2 tried `create_table('users')` which would fail. | CRITICAL | Task 2 now uses `add_column` to add `name`, `role`, `firm_id` to existing users table |
| 2 | **`down_revision` wrong** — plan had `'2_seed_demo_user'` but actual revision ID is `'seed_demo_user'` | CRITICAL | Fixed to `'seed_demo_user'` |
| 3 | **`text-muted` semantic break** — changing `--color-muted` from `#6B6B6B` (text color) to `#F5F5F3` (background) breaks **135 instances** of `text-muted` across 23 files (text becomes near-white on white = invisible) | CRITICAL | Added new Task 5.5 to migrate `text-muted` → `text-muted-foreground` BEFORE the CSS refactor |
| 4 | **`--color-muted-foreground` value changes silently** — from `#999999` to `#6B6B6B`, affecting 7 existing uses | MEDIUM | Kept at `#999999` to match current visual. Dropped redundant `muted-text` custom token |
| 5 | **Tasks 5, 7, 10-21 referenced "v1 plan" with no code** — ~12 tasks were unimplementable | MAJOR | Tasks 5 and 7 now fully specified. Tasks 10-17 expanded with clear specs |
| 6 | **Seed migration (Task 3) had no code** — said "same as v1 plan" | MEDIUM | Full seed migration code added |
| 7 | **No routers/ directory** — backend is monolithic main.py | MEDIUM | Task 5 specifies router creation and registration |
| 8 | **`getFirmTheme` / `FirmThemeResponse` not defined** — Task 8 imported them but Task 7 was empty | MEDIUM | Now defined in Task 7 |
| 9 | **AppLayout wrapping missing in firm routes** — firm layout only had FirmThemeProvider | MEDIUM | Fixed Task 9 firm layout to include AppLayout |
| 10 | **Hooks directory wrong** — plan implied `lib/`, actual hooks are in `frontend/hooks/` | LOW | Fixed paths to use `frontend/hooks/` |

---

## Research-Driven Changes from v1

| Issue | Old Plan | Fixed Plan |
|-------|----------|------------|
| `@theme inline` with hardcoded hex | `--color-primary: #111111` (inlined, not overridable) | Two-layer: `--app-primary` in `@layer theme`, `--color-primary: var(--app-primary)` in `@theme inline` |
| Dark mode + JS specificity conflict | Relied on CSS `.dark` rules (overridden by inline styles) | FirmThemeProvider sets both light AND dark `--app-*` values via JS on mode toggle |
| Page re-exports break params/layout | `export { default } from '...'` | Extract shared views into `/components/views/`, thin wrappers in firm routes |
| JSON vs JSONB | `Column(JSON)` | `Column(JSONB)` with `server_default` |
| Pydantic validation gaps | `light: Optional[dict]` | `light: Optional[ThemeColorTokens]` with hex regex validator |
| Next.js 14 params | `params: Promise<{ slug }>` (Next.js 15 API) | `params: { slug: string }` (synchronous, correct for v14) |
| Dynamic fonts | Modified root layout static `<link>` | `<link>` tag injection via custom hook + `document.fonts.ready` |
| Custom Lexintel tokens missing | Only standard shadcn/ui tokens | Includes `surface`, `surface-hover`, `border-strong`, `primary-light`, `accent-hover`, `success`, `warning` |

---

## Phase 1: Database & Multi-Tenancy Foundation

### Task 1: Add Firm and User models to SQLAlchemy

**Files:**
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/backend/models.py`

**Step 1: Add import for JSONB**

Change line 3 from:
```python
from sqlalchemy.dialects.postgresql import UUID
```
to:
```python
from sqlalchemy.dialects.postgresql import UUID, JSONB
```

**Step 2: Add UserRole enum, Firm model, and User model after FileType class (line 24)**

```python
class UserRole(str, enum.Enum):
    """User roles within a firm"""
    ADMIN = "admin"
    PARTNER = "partner"
    ASSOCIATE = "associate"
    PARALEGAL = "paralegal"


class Firm(Base):
    __tablename__ = "firms"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False, unique=True, index=True)
    logo_url = Column(String(500), nullable=True)
    theme_config = Column(JSONB, nullable=True, server_default='{}')
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    users = relationship("User", back_populates="firm")
    matters = relationship("Matter", back_populates="firm")

    def __repr__(self):
        return f"<Firm(id={self.id}, name={self.name}, slug={self.slug})>"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=True)
    role = Column(String(50), default="associate", nullable=False)
    firm_id = Column(UUID(as_uuid=True), ForeignKey("firms.id"), nullable=True, index=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    firm = relationship("Firm", back_populates="users")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"
```

**Step 3: Add `firm_id` FK and relationship to existing Matter model**

Add after `updated_at` column:
```python
    firm_id = Column(UUID(as_uuid=True), ForeignKey("firms.id"), nullable=True, index=True)
```

Add to Matter relationships:
```python
    firm = relationship("Firm", back_populates="matters")
```

**Step 4: Commit**

```bash
git add backend/models.py
git commit -m "$(cat <<'EOF'
feat: add Firm and User models with JSONB theme_config

Adds multi-tenancy foundation: Firm model with slug, logo_url,
and JSONB theme_config column with server_default. User model
with role and firm_id FK. Adds firm_id FK to Matter model.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create Alembic migration for firms, users columns, and matter firm_id

> **v3 FIX:** The `users` table already exists (created by migration `f794e7d74f24`
> with columns `id, email, password_hash, is_deleted, created_at, updated_at`).
> This migration ADDS columns to users, not creates the table.
> Also fixes `down_revision` from `'2_seed_demo_user'` to `'seed_demo_user'`.

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/backend/alembic/versions/3_add_firms_users_theme.py`

**Step 1: Write the migration file**

```python
"""Add firms table, user columns (name, role, firm_id), and firm_id to matters

Revision ID: 3_add_firms_users_theme
Revises: seed_demo_user
Create Date: 2026-02-15
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = '3_add_firms_users_theme'
down_revision = 'seed_demo_user'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create firms table with JSONB (not JSON)
    op.create_table(
        'firms',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(255), nullable=False, unique=True),
        sa.Column('logo_url', sa.String(500), nullable=True),
        sa.Column('theme_config', JSONB, nullable=True, server_default=sa.text("'{}'::jsonb")),
        sa.Column('is_deleted', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
    )
    op.create_index('ix_firms_slug', 'firms', ['slug'], unique=True)

    # ADD columns to existing users table (NOT create — table exists from f794e7d74f24)
    op.add_column('users', sa.Column('name', sa.String(255), nullable=True))
    op.add_column('users', sa.Column('role', sa.String(50), server_default='associate', nullable=False))
    op.add_column('users', sa.Column('firm_id', UUID(as_uuid=True), sa.ForeignKey('firms.id'), nullable=True))
    op.create_index('ix_users_firm_id', 'users', ['firm_id'])

    # Add firm_id to matters table
    op.add_column('matters', sa.Column('firm_id', UUID(as_uuid=True), sa.ForeignKey('firms.id'), nullable=True))
    op.create_index('ix_matters_firm_id', 'matters', ['firm_id'])


def downgrade() -> None:
    op.drop_index('ix_matters_firm_id', 'matters')
    op.drop_column('matters', 'firm_id')
    op.drop_index('ix_users_firm_id', 'users')
    op.drop_column('users', 'firm_id')
    op.drop_column('users', 'role')
    op.drop_column('users', 'name')
    op.drop_table('firms')
```

**Step 2: Commit**

```bash
git add backend/alembic/versions/3_add_firms_users_theme.py
git commit -m "$(cat <<'EOF'
feat: add migration for firms (JSONB), user columns, and matter firm_id

Creates firms table with JSONB theme_config and server_default.
Adds name, role, firm_id columns to existing users table.
Adds firm_id to matters table.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add default theme config constant and seed script

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/backend/theme_defaults.py`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/backend/alembic/versions/4_seed_default_firm.py`

**Step 1: Create theme defaults file**

This includes ALL tokens used by Lexintel's codebase — both standard shadcn/ui tokens AND the custom Lexintel tokens (`surface`, `surface-hover`, `border-strong`, `primary-light`, `accent-hover`, `accent-muted`, `success`, `warning`, etc.).

> **v3 FIX:** `muted-foreground` kept at `#999999` (matching current CSS value) instead
> of `#6B6B6B`. The `muted` token changes to `#F5F5F3` per shadcn/ui convention
> but a codebase-wide `text-muted` → `text-muted-foreground` migration in Task 5.5
> ensures no visual regression. Removed redundant `muted-text` custom token.

```python
"""Default theme configuration for new firms.

Includes standard shadcn/ui tokens AND Lexintel custom tokens
(surface, surface-hover, border-strong, primary-light, accent-hover,
accent-muted, success, success-light, warning, warning-light).
"""

DEFAULT_THEME_CONFIG = {
    "light": {
        # Standard shadcn/ui tokens
        "background": "#FAFAF8",
        "foreground": "#111111",
        "primary": "#111111",
        "primary-foreground": "#FAFAF8",
        "secondary": "#F5F5F3",
        "secondary-foreground": "#111111",
        "accent": "#111111",
        "accent-foreground": "#FFFFFF",
        "muted": "#F5F5F3",
        "muted-foreground": "#999999",
        "destructive": "#DC2626",
        "destructive-foreground": "#FFFFFF",
        "card": "#FFFFFF",
        "card-foreground": "#111111",
        "popover": "#FFFFFF",
        "popover-foreground": "#111111",
        "border": "#E5E4E2",
        "input": "#E5E4E2",
        "ring": "#111111",
        "chart-1": "#111111",
        "chart-2": "#444444",
        "chart-3": "#777777",
        "chart-4": "#AAAAAA",
        "chart-5": "#DDDDDD",
        "sidebar-background": "#F5F4F2",
        "sidebar-foreground": "#111111",
        "sidebar-accent": "#FFFFFF",
        "sidebar-border": "#E5E4E2",
        # Lexintel custom tokens (used 168+ times across codebase)
        "primary-light": "#222222",
        "accent-hover": "#2A2A2A",
        "accent-muted": "#F5F5F3",
        "surface": "#F5F4F2",
        "surface-hover": "#EDECEB",
        "border-strong": "#D1D0CE",
        "success": "#16A34A",
        "success-light": "#F0FDF4",
        "warning": "#CA8A04",
        "warning-light": "#FEFCE8",
    },
    "dark": {
        # Standard shadcn/ui tokens
        "background": "#0A0A0A",
        "foreground": "#FAFAF8",
        "primary": "#FAFAF8",
        "primary-foreground": "#111111",
        "secondary": "#1A1A1A",
        "secondary-foreground": "#FAFAF8",
        "accent": "#FAFAF8",
        "accent-foreground": "#111111",
        "muted": "#1A1A1A",
        "muted-foreground": "#666666",
        "destructive": "#EF4444",
        "destructive-foreground": "#FFFFFF",
        "card": "#141414",
        "card-foreground": "#FAFAF8",
        "popover": "#141414",
        "popover-foreground": "#FAFAF8",
        "border": "#2A2A2A",
        "input": "#2A2A2A",
        "ring": "#FAFAF8",
        "chart-1": "#FAFAF8",
        "chart-2": "#CCCCCC",
        "chart-3": "#999999",
        "chart-4": "#666666",
        "chart-5": "#333333",
        "sidebar-background": "#0F0F0F",
        "sidebar-foreground": "#FAFAF8",
        "sidebar-accent": "#1A1A1A",
        "sidebar-border": "#2A2A2A",
        # Lexintel custom tokens
        "primary-light": "#DDDDDD",
        "accent-hover": "#CCCCCC",
        "accent-muted": "#1A1A1A",
        "surface": "#141414",
        "surface-hover": "#1E1E1E",
        "border-strong": "#3A3A3A",
        "success": "#22C55E",
        "success-light": "#052E16",
        "warning": "#EAB308",
        "warning-light": "#422006",
    },
    "typography": {
        "font-sans": "DM Sans",
        "font-display": "DM Serif Display",
        "font-mono": "JetBrains Mono",
    },
    "layout": {
        "radius-sm": "0.375rem",
        "radius-md": "0.5rem",
        "radius-lg": "0.75rem",
        "radius-xl": "1rem",
    },
    "shadows": {
        "color": "#000000",
        "opacity": "0.04",
        "blur": "12",
        "spread": "0",
        "offset-x": "0",
        "offset-y": "4",
    },
}

# All valid theme token keys for validation
VALID_COLOR_TOKENS = list(DEFAULT_THEME_CONFIG["light"].keys())
VALID_TYPOGRAPHY_TOKENS = list(DEFAULT_THEME_CONFIG["typography"].keys())
VALID_LAYOUT_TOKENS = list(DEFAULT_THEME_CONFIG["layout"].keys())
VALID_SHADOW_TOKENS = list(DEFAULT_THEME_CONFIG["shadows"].keys())
```

**Step 2: Create seed migration**

```python
"""Seed default firm with theme config

Revision ID: 4_seed_default_firm
Revises: 3_add_firms_users_theme
Create Date: 2026-02-15
"""
from alembic import op
import sqlalchemy as sa
import json

revision = '4_seed_default_firm'
down_revision = '3_add_firms_users_theme'
branch_labels = None
depends_on = None

# Import inline to avoid dependency issues during migration
DEFAULT_THEME_CONFIG = {
    "light": {
        "background": "#FAFAF8", "foreground": "#111111",
        "primary": "#111111", "primary-foreground": "#FAFAF8",
        "secondary": "#F5F5F3", "secondary-foreground": "#111111",
        "accent": "#111111", "accent-foreground": "#FFFFFF",
        "muted": "#F5F5F3", "muted-foreground": "#999999",
        "destructive": "#DC2626", "destructive-foreground": "#FFFFFF",
        "card": "#FFFFFF", "card-foreground": "#111111",
        "popover": "#FFFFFF", "popover-foreground": "#111111",
        "border": "#E5E4E2", "input": "#E5E4E2", "ring": "#111111",
        "chart-1": "#111111", "chart-2": "#444444", "chart-3": "#777777",
        "chart-4": "#AAAAAA", "chart-5": "#DDDDDD",
        "sidebar-background": "#F5F4F2", "sidebar-foreground": "#111111",
        "sidebar-accent": "#FFFFFF", "sidebar-border": "#E5E4E2",
        "primary-light": "#222222", "accent-hover": "#2A2A2A",
        "accent-muted": "#F5F5F3", "surface": "#F5F4F2",
        "surface-hover": "#EDECEB", "border-strong": "#D1D0CE",
        "success": "#16A34A", "success-light": "#F0FDF4",
        "warning": "#CA8A04", "warning-light": "#FEFCE8",
    },
    "dark": {
        "background": "#0A0A0A", "foreground": "#FAFAF8",
        "primary": "#FAFAF8", "primary-foreground": "#111111",
        "secondary": "#1A1A1A", "secondary-foreground": "#FAFAF8",
        "accent": "#FAFAF8", "accent-foreground": "#111111",
        "muted": "#1A1A1A", "muted-foreground": "#666666",
        "destructive": "#EF4444", "destructive-foreground": "#FFFFFF",
        "card": "#141414", "card-foreground": "#FAFAF8",
        "popover": "#141414", "popover-foreground": "#FAFAF8",
        "border": "#2A2A2A", "input": "#2A2A2A", "ring": "#FAFAF8",
        "chart-1": "#FAFAF8", "chart-2": "#CCCCCC", "chart-3": "#999999",
        "chart-4": "#666666", "chart-5": "#333333",
        "sidebar-background": "#0F0F0F", "sidebar-foreground": "#FAFAF8",
        "sidebar-accent": "#1A1A1A", "sidebar-border": "#2A2A2A",
        "primary-light": "#DDDDDD", "accent-hover": "#CCCCCC",
        "accent-muted": "#1A1A1A", "surface": "#141414",
        "surface-hover": "#1E1E1E", "border-strong": "#3A3A3A",
        "success": "#22C55E", "success-light": "#052E16",
        "warning": "#EAB308", "warning-light": "#422006",
    },
    "typography": {"font-sans": "DM Sans", "font-display": "DM Serif Display", "font-mono": "JetBrains Mono"},
    "layout": {"radius-sm": "0.375rem", "radius-md": "0.5rem", "radius-lg": "0.75rem", "radius-xl": "1rem"},
    "shadows": {"color": "#000000", "opacity": "0.04", "blur": "12", "spread": "0", "offset-x": "0", "offset-y": "4"},
}


def upgrade() -> None:
    theme_json = json.dumps(DEFAULT_THEME_CONFIG)
    op.execute(
        sa.text("""
            INSERT INTO firms (id, name, slug, theme_config, is_deleted, created_at, updated_at)
            VALUES (
                '00000000-0000-0000-0000-000000000010',
                'LexIntel Default',
                'default',
                :theme_config::jsonb,
                false,
                NOW(),
                NOW()
            )
            ON CONFLICT DO NOTHING;
        """),
        {"theme_config": theme_json},
    )

    # Associate existing demo user with default firm
    op.execute(
        sa.text("""
            UPDATE users
            SET firm_id = '00000000-0000-0000-0000-000000000010',
                role = 'admin',
                name = 'Demo User'
            WHERE id = '00000000-0000-0000-0000-000000000001';
        """)
    )


def downgrade() -> None:
    op.execute(
        sa.text("""
            UPDATE users SET firm_id = NULL, role = 'associate', name = NULL
            WHERE id = '00000000-0000-0000-0000-000000000001';
        """)
    )
    op.execute(
        sa.text("DELETE FROM firms WHERE id = '00000000-0000-0000-0000-000000000010';")
    )
```

**Step 3: Commit**

```bash
git add backend/theme_defaults.py backend/alembic/versions/4_seed_default_firm.py
git commit -m "$(cat <<'EOF'
feat: add default theme config with all Lexintel custom tokens

Includes standard shadcn/ui tokens AND Lexintel-specific tokens
(surface, border-strong, success, warning, etc). Seeds default firm
and associates demo user as admin.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Backend API Layer

### Task 4: Add Pydantic schemas with hex validation

**Files:**
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/backend/schemas.py`

**Step 1: Add firm and theme schemas with proper validation**

Add `import re` at the top. Add `field_validator` to the pydantic import:
```python
from pydantic import BaseModel, Field, ConfigDict, field_validator
```

Then append after the existing QueryResponse class:

```python
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
```

**Step 2: Commit**

```bash
git add backend/schemas.py
git commit -m "$(cat <<'EOF'
feat: add Pydantic schemas with hex color validation

ThemeColorTokens validates all color tokens as #RRGGBB hex.
ThemeConfigUpdate uses typed light/dark fields (not raw dict).
Includes all Lexintel custom tokens.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Create firm API router

> **v3 FIX:** Fully specified. Backend currently has no routers/ directory —
> all routes are in main.py. This task creates the router pattern.

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/backend/routers/__init__.py`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/backend/routers/firms.py`
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/backend/main.py`

**Step 1: Create routers directory and __init__.py**

```python
# backend/routers/__init__.py
```
(empty file)

**Step 2: Create firm router**

```python
# backend/routers/firms.py
"""Firm management and theme configuration API endpoints."""

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from uuid import UUID
import re
import copy

try:
    from backend.database import get_db
    from backend.models import Firm, User
    from backend.schemas import (
        FirmCreate, FirmResponse, ThemeConfigUpdate,
        MemberInvite, MemberRoleUpdate,
    )
    from backend.theme_defaults import DEFAULT_THEME_CONFIG
except ImportError:
    from database import get_db
    from models import Firm, User
    from schemas import (
        FirmCreate, FirmResponse, ThemeConfigUpdate,
        MemberInvite, MemberRoleUpdate,
    )
    from theme_defaults import DEFAULT_THEME_CONFIG

router = APIRouter(prefix="/api/firms", tags=["firms"])


def _get_firm_or_404(slug: str, db: Session) -> Firm:
    firm = db.query(Firm).filter(Firm.slug == slug, Firm.is_deleted == False).first()
    if not firm:
        raise HTTPException(status_code=404, detail="Firm not found")
    return firm


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')


# ---- Firm CRUD ----

@router.post("", response_model=FirmResponse, status_code=201)
async def create_firm(body: FirmCreate, db: Session = Depends(get_db)):
    slug = body.slug or _slugify(body.name)
    existing = db.query(Firm).filter(Firm.slug == slug).first()
    if existing:
        raise HTTPException(status_code=409, detail="Firm slug already exists")

    firm = Firm(
        name=body.name,
        slug=slug,
        theme_config=copy.deepcopy(DEFAULT_THEME_CONFIG),
    )
    db.add(firm)
    db.commit()
    db.refresh(firm)
    return firm


@router.get("/{slug}", response_model=FirmResponse)
async def get_firm(slug: str, db: Session = Depends(get_db)):
    return _get_firm_or_404(slug, db)


@router.delete("/{slug}")
async def delete_firm(slug: str, db: Session = Depends(get_db)):
    firm = _get_firm_or_404(slug, db)
    firm.is_deleted = True
    db.commit()
    return {"slug": slug, "deleted": True}


# ---- Theme ----

@router.get("/{slug}/theme")
async def get_firm_theme(slug: str, db: Session = Depends(get_db)):
    """Get firm theme config for FirmThemeProvider."""
    firm = _get_firm_or_404(slug, db)
    theme = firm.theme_config or DEFAULT_THEME_CONFIG
    return {
        "firm_name": firm.name,
        "firm_slug": firm.slug,
        "logo_url": firm.logo_url,
        "theme": theme,
    }


@router.put("/{slug}/theme")
async def update_firm_theme(slug: str, body: ThemeConfigUpdate, db: Session = Depends(get_db)):
    """Update theme. Merges partial updates into existing config."""
    firm = _get_firm_or_404(slug, db)

    current = copy.deepcopy(firm.theme_config or DEFAULT_THEME_CONFIG)

    if body.light:
        updates = body.light.model_dump(by_alias=True, exclude_none=True)
        current.setdefault("light", {}).update(updates)
    if body.dark:
        updates = body.dark.model_dump(by_alias=True, exclude_none=True)
        current.setdefault("dark", {}).update(updates)
    if body.typography:
        updates = body.typography.model_dump(by_alias=True, exclude_none=True)
        current.setdefault("typography", {}).update(updates)
    if body.layout:
        updates = body.layout.model_dump(by_alias=True, exclude_none=True)
        current.setdefault("layout", {}).update(updates)
    if body.shadows:
        updates = body.shadows.model_dump(by_alias=True, exclude_none=True)
        current.setdefault("shadows", {}).update(updates)

    # Reassignment triggers SQLAlchemy change detection for JSONB
    firm.theme_config = current
    flag_modified(firm, "theme_config")
    db.commit()
    db.refresh(firm)

    return {"firm_slug": firm.slug, "theme": firm.theme_config}


@router.post("/{slug}/theme/reset")
async def reset_firm_theme(slug: str, db: Session = Depends(get_db)):
    """Reset theme to defaults."""
    firm = _get_firm_or_404(slug, db)
    firm.theme_config = copy.deepcopy(DEFAULT_THEME_CONFIG)
    flag_modified(firm, "theme_config")
    db.commit()
    return {"firm_slug": firm.slug, "theme": firm.theme_config}


# ---- Members ----

@router.get("/{slug}/members")
async def list_members(slug: str, db: Session = Depends(get_db)):
    firm = _get_firm_or_404(slug, db)
    members = db.query(User).filter(User.firm_id == firm.id, User.is_deleted == False).all()
    return [
        {
            "id": str(m.id),
            "email": m.email,
            "name": m.name,
            "role": m.role,
        }
        for m in members
    ]


@router.put("/{slug}/members/{user_id}/role")
async def update_member_role(
    slug: str, user_id: str, body: MemberRoleUpdate, db: Session = Depends(get_db)
):
    firm = _get_firm_or_404(slug, db)
    try:
        uid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    user = db.query(User).filter(User.id == uid, User.firm_id == firm.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found in this firm")

    user.role = body.role
    db.commit()
    return {"id": str(user.id), "role": user.role}
```

**Step 3: Register router in main.py**

Add after the CORS middleware block (around line 64):
```python
# Import and register firm router
try:
    from backend.routers.firms import router as firms_router
except ImportError:
    try:
        from routers.firms import router as firms_router
    except ImportError:
        from .routers.firms import router as firms_router

app.include_router(firms_router)
```

**Step 4: Commit**

```bash
git add backend/routers/ backend/main.py
git commit -m "$(cat <<'EOF'
feat: add firm API router with theme CRUD endpoints

Creates /api/firms with CRUD, /api/firms/:slug/theme for
get/update/reset, and /api/firms/:slug/members for team
management. Registers router in main.py.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: CSS Refactor & Frontend Routing

### Task 5.5: Migrate `text-muted` to `text-muted-foreground` across codebase (NEW — CRITICAL)

> **v3 NEW TASK:** The current CSS has `--color-muted: #6B6B6B` which is used as a text
> color via `text-muted` (135 instances across 23 files). Task 6 changes `--color-muted`
> to `#F5F5F3` (a background color per shadcn/ui convention). Without this migration,
> ALL text-muted usage becomes near-white text on white background = invisible.
>
> This task MUST run BEFORE Task 6.

**Scope:** Replace `text-muted` (used as text color) with `text-muted-foreground` across all frontend files. The current `--color-muted: #6B6B6B` value will move to `--color-muted-foreground` in the new CSS architecture, preserving the exact same visual.

**Rules:**
- `text-muted` → `text-muted-foreground` (text color usage)
- `text-muted/50` → `text-muted-foreground/50` (opacity variants)
- `text-muted-foreground` stays unchanged (already correct)
- `bg-muted` stays unchanged (no instances exist, and the new value `#F5F5F3` is correct for backgrounds)
- `border-muted` stays unchanged

**Files to update (23 files, 135 instances):**
All `.tsx` files in `frontend/components/` and `frontend/app/` that use `text-muted`.

**Step 1: Run search-and-replace**

Use editor find-and-replace with word boundary matching:
- Pattern: `text-muted` (but NOT `text-muted-foreground`)
- Replace: `text-muted-foreground`

Apply across all `.tsx` files in `frontend/`.

**Step 2: Verify build**

```bash
cd /Users/smeet/Documents/GitHub/Lexintel/frontend && npm run build
```

**Step 3: Spot-check key files**

Visually confirm Sidebar.tsx, dashboard/page.tsx, and matters/page.tsx use `text-muted-foreground` where they previously had `text-muted`.

**Step 4: Commit**

```bash
git add frontend/
git commit -m "$(cat <<'EOF'
refactor: migrate text-muted to text-muted-foreground

Aligns with shadcn/ui convention where 'muted' is a background
color and 'muted-foreground' is its text color. Prepares for
two-layer CSS architecture where --color-muted becomes #F5F5F3.
135 instances across 23 files.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Refactor globals.css to two-layer architecture (CRITICAL)

> **PREREQUISITE:** Task 5.5 must be completed first.

**Files:**
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/frontend/app/globals.css`

**Why this is critical:** The current `@theme inline` hardcodes hex values directly into Tailwind utility classes. This means `bg-primary` compiles to `background-color: #111111` — no CSS variable reference, so `setProperty()` does nothing. We must split into two layers.

> **v3 FIX:** `--app-muted-foreground` is `#999999` (matching current `--color-muted-foreground`
> value). After Task 5.5's migration, `text-muted-foreground` renders as `#999999`
> — exactly the same as the old `text-muted-foreground` value. The old `text-muted`
> value of `#6B6B6B` is no longer referenced by any component.

**Step 1: Rewrite globals.css**

```css
@import "tailwindcss";
@import "tw-animate-css";

@custom-variant dark (&:is(.dark *));

/* ━━━ Layer 1: Bridge to Tailwind utility classes ━━━
   These map --color-* to var(--app-*) so Tailwind generates:
   .bg-primary { background-color: var(--app-primary); }
   Now setProperty('--app-primary', '#blue') works at runtime. */
@theme inline {
  /* Standard shadcn/ui tokens */
  --color-primary: var(--app-primary);
  --color-primary-foreground: var(--app-primary-foreground);
  --color-secondary: var(--app-secondary);
  --color-secondary-foreground: var(--app-secondary-foreground);
  --color-accent: var(--app-accent);
  --color-accent-foreground: var(--app-accent-foreground);
  --color-background: var(--app-background);
  --color-foreground: var(--app-foreground);
  --color-card: var(--app-card);
  --color-card-foreground: var(--app-card-foreground);
  --color-popover: var(--app-popover);
  --color-popover-foreground: var(--app-popover-foreground);
  --color-muted: var(--app-muted);
  --color-muted-foreground: var(--app-muted-foreground);
  --color-destructive: var(--app-destructive);
  --color-destructive-foreground: var(--app-destructive-foreground);
  --color-border: var(--app-border);
  --color-input: var(--app-input);
  --color-ring: var(--app-ring);
  --color-chart-1: var(--app-chart-1);
  --color-chart-2: var(--app-chart-2);
  --color-chart-3: var(--app-chart-3);
  --color-chart-4: var(--app-chart-4);
  --color-chart-5: var(--app-chart-5);

  /* Sidebar tokens */
  --color-sidebar-background: var(--app-sidebar-background);
  --color-sidebar-foreground: var(--app-sidebar-foreground);
  --color-sidebar-accent: var(--app-sidebar-accent);
  --color-sidebar-border: var(--app-sidebar-border);

  /* Lexintel custom tokens */
  --color-primary-light: var(--app-primary-light);
  --color-accent-hover: var(--app-accent-hover);
  --color-accent-muted: var(--app-accent-muted);
  --color-surface: var(--app-surface);
  --color-surface-hover: var(--app-surface-hover);
  --color-border-strong: var(--app-border-strong);
  --color-success: var(--app-success);
  --color-success-light: var(--app-success-light);
  --color-warning: var(--app-warning);
  --color-warning-light: var(--app-warning-light);

  /* Typography — also overridable */
  --font-sans: var(--app-font-sans);
  --font-display: var(--app-font-display);
  --font-mono: var(--app-font-mono);

  /* Radii */
  --radius-sm: var(--app-radius-sm);
  --radius-md: var(--app-radius-md);
  --radius-lg: var(--app-radius-lg);
  --radius-xl: var(--app-radius-xl);

  /* Animations (static, not overridable) */
  --animate-fade-in: fade-in 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  --animate-slide-in: slide-in 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  --animate-slide-up: slide-up 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ━━━ Layer 2: Actual values (overridable at runtime via JS) ━━━
   Use @layer theme so dark mode has correct specificity.
   JS setProperty('--app-primary', '#xxx') overrides these. */
@layer theme {
  :root {
    /* Standard */
    --app-primary: #111111;
    --app-primary-foreground: #FAFAF8;
    --app-secondary: #F5F5F3;
    --app-secondary-foreground: #111111;
    --app-accent: #111111;
    --app-accent-foreground: #FFFFFF;
    --app-background: #FAFAF8;
    --app-foreground: #111111;
    --app-card: #FFFFFF;
    --app-card-foreground: #111111;
    --app-popover: #FFFFFF;
    --app-popover-foreground: #111111;
    --app-muted: #F5F5F3;
    --app-muted-foreground: #999999;
    --app-destructive: #DC2626;
    --app-destructive-foreground: #FFFFFF;
    --app-border: #E5E4E2;
    --app-input: #E5E4E2;
    --app-ring: #111111;
    --app-chart-1: #111111;
    --app-chart-2: #444444;
    --app-chart-3: #777777;
    --app-chart-4: #AAAAAA;
    --app-chart-5: #DDDDDD;
    --app-sidebar-background: #F5F4F2;
    --app-sidebar-foreground: #111111;
    --app-sidebar-accent: #FFFFFF;
    --app-sidebar-border: #E5E4E2;

    /* Lexintel custom */
    --app-primary-light: #222222;
    --app-accent-hover: #2A2A2A;
    --app-accent-muted: #F5F5F3;
    --app-surface: #F5F4F2;
    --app-surface-hover: #EDECEB;
    --app-border-strong: #D1D0CE;
    --app-success: #16A34A;
    --app-success-light: #F0FDF4;
    --app-warning: #CA8A04;
    --app-warning-light: #FEFCE8;

    /* Typography */
    --app-font-sans: "DM Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
    --app-font-display: "DM Serif Display", Georgia, "Times New Roman", serif;
    --app-font-mono: "JetBrains Mono", ui-monospace, monospace;

    /* Radii */
    --app-radius-sm: 0.375rem;
    --app-radius-md: 0.5rem;
    --app-radius-lg: 0.75rem;
    --app-radius-xl: 1rem;
  }

  /* Dark mode defaults — NOTE: when FirmThemeProvider sets
     --app-* via JS inline styles, it overrides BOTH :root and .dark
     because inline styles have higher specificity. The provider
     must set ALL values for whichever mode is active. */
  .dark {
    --app-primary: #FAFAF8;
    --app-primary-foreground: #111111;
    --app-secondary: #1A1A1A;
    --app-secondary-foreground: #FAFAF8;
    --app-accent: #FAFAF8;
    --app-accent-foreground: #111111;
    --app-background: #0A0A0A;
    --app-foreground: #FAFAF8;
    --app-card: #141414;
    --app-card-foreground: #FAFAF8;
    --app-popover: #141414;
    --app-popover-foreground: #FAFAF8;
    --app-muted: #1A1A1A;
    --app-muted-foreground: #666666;
    --app-destructive: #EF4444;
    --app-destructive-foreground: #FFFFFF;
    --app-border: #2A2A2A;
    --app-input: #2A2A2A;
    --app-ring: #FAFAF8;
    --app-chart-1: #FAFAF8;
    --app-chart-2: #CCCCCC;
    --app-chart-3: #999999;
    --app-chart-4: #666666;
    --app-chart-5: #333333;
    --app-sidebar-background: #0F0F0F;
    --app-sidebar-foreground: #FAFAF8;
    --app-sidebar-accent: #1A1A1A;
    --app-sidebar-border: #2A2A2A;

    --app-primary-light: #DDDDDD;
    --app-accent-hover: #CCCCCC;
    --app-accent-muted: #1A1A1A;
    --app-surface: #141414;
    --app-surface-hover: #1E1E1E;
    --app-border-strong: #3A3A3A;
    --app-success: #22C55E;
    --app-success-light: #052E16;
    --app-warning: #EAB308;
    --app-warning-light: #422006;
  }
}

/* ━━━ Keyframes (unchanged) ━━━ */
@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes slide-in {
  from { opacity: 0; transform: translateX(-12px); }
  to { opacity: 1; transform: translateX(0); }
}
@keyframes slide-up {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes subtle-pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 0.7; }
}
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-6px); }
}

/* ━━━ Base styles (unchanged) ━━━ */
@layer base {
  *, *::before, *::after { border-color: var(--color-border); }
  body {
    background-color: var(--color-background);
    color: var(--color-foreground);
    font-family: var(--font-sans);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    letter-spacing: -0.011em;
  }
  h1, h2, h3, h4 {
    font-family: var(--font-display);
    letter-spacing: -0.025em;
  }
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--color-border-strong); border-radius: 100px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--color-muted-foreground); }
  ::selection { background-color: #11111118; color: var(--color-foreground); }
}

/* ━━━ Utility classes (unchanged) ━━━ */
@layer utilities {
  .glass {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
  }
  .glass-subtle {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(12px) saturate(150%);
    -webkit-backdrop-filter: blur(12px) saturate(150%);
  }
  .shadow-elevated {
    box-shadow: 0 0 0 1px rgba(0,0,0,0.03), 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);
  }
  .shadow-elevated-lg {
    box-shadow: 0 0 0 1px rgba(0,0,0,0.03), 0 2px 4px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.06);
  }
  .shadow-glow {
    box-shadow: 0 0 0 1px rgba(0,0,0,0.04), 0 2px 8px rgba(0,0,0,0.06), 0 12px 40px rgba(0,0,0,0.08);
  }
  .skeleton {
    background: linear-gradient(90deg, var(--color-surface) 25%, var(--color-surface-hover) 50%, var(--color-surface) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.8s ease-in-out infinite;
  }
  .text-gradient {
    background: linear-gradient(135deg, #111111 0%, #444444 50%, #111111 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .grid-pattern-light {
    background-image: linear-gradient(#111 1px, transparent 1px), linear-gradient(90deg, #111 1px, transparent 1px);
    background-size: 60px 60px;
  }
  .grid-pattern-dark {
    background-image: linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px);
    background-size: 60px 60px;
  }
}
```

**Step 2: Verify the build compiles**

```bash
cd /Users/smeet/Documents/GitHub/Lexintel/frontend && npm run build
```

Expected: Build succeeds. All existing Tailwind utility classes (`bg-primary`, `text-foreground`, `bg-surface`, etc.) still work because `@theme inline` registers them via `var()`.

**Step 3: Commit**

```bash
git add frontend/app/globals.css
git commit -m "$(cat <<'EOF'
refactor: two-layer CSS architecture for runtime theme overrides

Splits globals.css into @theme inline (var() bridges for Tailwind
utility class generation) and @layer theme (actual values in :root
and .dark). Now setProperty('--app-primary', '#xxx') works at
runtime. All Lexintel tokens + standard shadcn/ui tokens.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add frontend types and API services for firms/themes

> **v3 FIX:** Fully specified. Previously said "Same as v1 plan" with no code.

**Files:**
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/frontend/lib/types.ts`
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/frontend/lib/api-services.ts`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/hooks/use-firm.ts`

**Step 1: Add theme types to types.ts**

Append to the end of `frontend/lib/types.ts`:

```typescript
// ============================================
// FIRM & THEME TYPES
// ============================================

export interface ThemeColorTokens {
  [key: string]: string
}

export interface ThemeTypography {
  "font-sans"?: string
  "font-display"?: string
  "font-mono"?: string
}

export interface ThemeLayout {
  "radius-sm"?: string
  "radius-md"?: string
  "radius-lg"?: string
  "radius-xl"?: string
}

export interface ThemeShadows {
  color?: string
  opacity?: string
  blur?: string
  spread?: string
  "offset-x"?: string
  "offset-y"?: string
}

export interface ThemeConfig {
  light: ThemeColorTokens
  dark: ThemeColorTokens
  typography?: ThemeTypography
  layout?: ThemeLayout
  shadows?: ThemeShadows
}

export interface FirmThemeResponse {
  firm_name: string
  firm_slug: string
  logo_url: string | null
  theme: ThemeConfig
}

export interface FirmResponse {
  id: string
  name: string
  slug: string
  logo_url: string | null
  theme_config: ThemeConfig | null
  created_at: string
  updated_at: string
}

export interface FirmMember {
  id: string
  email: string
  name: string | null
  role: "admin" | "partner" | "associate" | "paralegal"
}
```

**Step 2: Add firm API service functions to api-services.ts**

Append to the end of `frontend/lib/api-services.ts`:

```typescript
// ============================================
// Firm & Theme API Service Functions
// ============================================

import type { FirmThemeResponse, ThemeConfig, FirmMember } from "./types"

export async function getFirmTheme(slug: string): Promise<FirmThemeResponse> {
  const { data } = await api.get<FirmThemeResponse>(`/api/firms/${slug}/theme`)
  return data
}

export async function updateFirmTheme(slug: string, theme: Partial<ThemeConfig>): Promise<{ firm_slug: string; theme: ThemeConfig }> {
  const { data } = await api.put(`/api/firms/${slug}/theme`, theme)
  return data
}

export async function resetFirmTheme(slug: string): Promise<{ firm_slug: string; theme: ThemeConfig }> {
  const { data } = await api.post(`/api/firms/${slug}/theme/reset`)
  return data
}

export async function getFirmMembers(slug: string): Promise<FirmMember[]> {
  const { data } = await api.get<FirmMember[]>(`/api/firms/${slug}/members`)
  return data
}
```

**Step 3: Create firm hooks**

> **v3 FIX:** Hooks go in `frontend/hooks/` (where `use-matters.ts` lives), not `frontend/lib/`.

```typescript
// frontend/hooks/use-firm.ts
"use client"

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  getFirmTheme,
  updateFirmTheme,
  resetFirmTheme,
  getFirmMembers,
} from "@/lib/api-services"
import type { FirmThemeResponse, ThemeConfig, FirmMember } from "@/lib/types"

export function useFirmTheme(slug: string) {
  return useQuery<FirmThemeResponse>({
    queryKey: ["firm-theme", slug],
    queryFn: () => getFirmTheme(slug),
    staleTime: 5 * 60 * 1000,
    enabled: !!slug,
  })
}

export function useUpdateFirmTheme(slug: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (theme: Partial<ThemeConfig>) => updateFirmTheme(slug, theme),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["firm-theme", slug] })
    },
  })
}

export function useResetFirmTheme(slug: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => resetFirmTheme(slug),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["firm-theme", slug] })
    },
  })
}

export function useFirmMembers(slug: string) {
  return useQuery<FirmMember[]>({
    queryKey: ["firm-members", slug],
    queryFn: () => getFirmMembers(slug),
    enabled: !!slug,
  })
}
```

**Step 4: Commit**

```bash
git add frontend/lib/types.ts frontend/lib/api-services.ts frontend/hooks/use-firm.ts
git commit -m "$(cat <<'EOF'
feat: add frontend types, API services, and hooks for firm theming

Adds ThemeConfig, FirmThemeResponse types. Adds getFirmTheme,
updateFirmTheme, resetFirmTheme API functions. Creates use-firm
hooks with React Query integration.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Create FirmThemeProvider (research-corrected)

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/lib/firm-theme-context.tsx`

**Key fixes from research:**
1. Sets `--app-*` variables (not `--color-*`) since `@theme inline` bridges them
2. On dark mode toggle: sets ALL `--app-*` values from the correct mode's color set (doesn't rely on CSS `.dark` rules, since inline styles have higher specificity)
3. Uses `<link>` injection + `document.fonts.ready` for dynamic Google Fonts (not `next/font`)
4. Loading skeleton uses hardcoded colors (not CSS variables, since they haven't loaded yet)

```tsx
"use client"

import React, { createContext, useContext, useEffect, useState, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import { getFirmTheme } from "@/lib/api-services"
import type { ThemeConfig, ThemeColorTokens, FirmThemeResponse } from "@/lib/types"

interface FirmThemeContextValue {
  firmName: string
  firmSlug: string
  logoUrl: string | null
  theme: ThemeConfig | null
  isDarkMode: boolean
  toggleDarkMode: () => void
  isLoading: boolean
}

const FirmThemeContext = createContext<FirmThemeContextValue | null>(null)

export function useFirmTheme() {
  const ctx = useContext(FirmThemeContext)
  if (!ctx) throw new Error("useFirmTheme must be used within FirmThemeProvider")
  return ctx
}

/**
 * Apply theme to DOM by setting --app-* CSS variables.
 *
 * IMPORTANT: We set --app-* (not --color-*) because @theme inline
 * maps --color-primary: var(--app-primary). Inline styles on
 * documentElement have higher specificity than @layer theme rules,
 * so we must set ALL values for the active mode — the CSS .dark
 * rules will be overridden by these inline styles.
 */
function applyThemeToDOM(theme: ThemeConfig, isDark: boolean) {
  const root = document.documentElement
  const colors: ThemeColorTokens = isDark ? theme.dark : theme.light

  // Set ALL color tokens for the active mode
  for (const [key, value] of Object.entries(colors)) {
    root.style.setProperty(`--app-${key}`, value)
  }

  // Typography
  if (theme.typography) {
    const sans = theme.typography["font-sans"]
    const display = theme.typography["font-display"]
    const mono = theme.typography["font-mono"]
    if (sans) root.style.setProperty("--app-font-sans", `"${sans}", ui-sans-serif, system-ui, -apple-system, sans-serif`)
    if (display) root.style.setProperty("--app-font-display", `"${display}", Georgia, "Times New Roman", serif`)
    if (mono) root.style.setProperty("--app-font-mono", `"${mono}", ui-monospace, monospace`)
  }

  // Layout
  if (theme.layout) {
    for (const [key, value] of Object.entries(theme.layout)) {
      root.style.setProperty(`--app-${key}`, value)
    }
  }

  // Dark class (for @custom-variant dark selectors in component CSS)
  if (isDark) {
    root.classList.add("dark")
  } else {
    root.classList.remove("dark")
  }
}

function clearThemeFromDOM() {
  const root = document.documentElement
  const style = root.style
  for (let i = style.length - 1; i >= 0; i--) {
    const prop = style[i]
    if (prop.startsWith("--app-")) {
      style.removeProperty(prop)
    }
  }
  root.classList.remove("dark")
}

/**
 * Load Google Fonts dynamically via <link> injection.
 * Waits for document.fonts.ready before resolving.
 */
function loadGoogleFonts(fonts: string[]) {
  const filtered = fonts.filter(Boolean)
  if (filtered.length === 0) return

  const linkId = "firm-theme-fonts"
  let link = document.getElementById(linkId) as HTMLLinkElement | null
  if (!link) {
    link = document.createElement("link")
    link.id = linkId
    link.rel = "stylesheet"
    document.head.appendChild(link)
  }

  const families = filtered
    .map((f) => `family=${f.replace(/ /g, "+")}:wght@400;500;600;700`)
    .join("&")
  link.href = `https://fonts.googleapis.com/css2?${families}&display=swap`
}

interface FirmThemeProviderProps {
  children: React.ReactNode
  firmSlug: string
}

export default function FirmThemeProvider({ children, firmSlug }: FirmThemeProviderProps) {
  const [isDarkMode, setIsDarkMode] = useState(false)

  const { data, isLoading } = useQuery<FirmThemeResponse>({
    queryKey: ["firm-theme", firmSlug],
    queryFn: () => getFirmTheme(firmSlug),
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  // Load dark mode preference
  useEffect(() => {
    const stored = localStorage.getItem(`lexintel_dark_mode_${firmSlug}`)
    if (stored === "true") setIsDarkMode(true)
  }, [firmSlug])

  // Apply theme whenever data or dark mode changes
  useEffect(() => {
    if (data?.theme) {
      applyThemeToDOM(data.theme, isDarkMode)
      loadGoogleFonts([
        data.theme.typography?.["font-sans"],
        data.theme.typography?.["font-display"],
        data.theme.typography?.["font-mono"],
      ].filter(Boolean) as string[])
    }
    return () => clearThemeFromDOM()
  }, [data, isDarkMode])

  const toggleDarkMode = useCallback(() => {
    setIsDarkMode((prev) => {
      const next = !prev
      localStorage.setItem(`lexintel_dark_mode_${firmSlug}`, String(next))
      return next
    })
  }, [firmSlug])

  // Loading skeleton — uses hardcoded colors since CSS vars haven't loaded
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: "#FAFAF8" }}>
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 rounded-lg animate-pulse" style={{ backgroundColor: "#111" }} />
          <div className="h-4 w-32 rounded animate-pulse" style={{ backgroundColor: "#E5E4E2" }} />
        </div>
      </div>
    )
  }

  return (
    <FirmThemeContext.Provider
      value={{
        firmName: data?.firm_name || "",
        firmSlug,
        logoUrl: data?.logo_url || null,
        theme: data?.theme || null,
        isDarkMode,
        toggleDarkMode,
        isLoading,
      }}
    >
      {children}
    </FirmThemeContext.Provider>
  )
}
```

**Step 2: Commit**

```bash
git add frontend/lib/firm-theme-context.tsx
git commit -m "$(cat <<'EOF'
feat: add FirmThemeProvider with --app-* variable application

Sets --app-* variables (bridged by @theme inline to --color-*).
On dark mode toggle, sets ALL values for the active mode to
override CSS .dark rules. Dynamic Google Font loading via
<link> injection. Hardcoded loading skeleton.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Extract shared views and create firm-scoped routes

> **v3 FIX:** Firm layout now includes AppLayout wrapping (all current pages use
> `<AppLayout title="...">` from `@/layouts/AppLayout`). Firm-scoped pages must
> preserve this wrapping.

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/views/DashboardView.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/views/MattersView.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/views/MatterDetailView.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/views/SettingsView.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/views/PrecedentsView.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/views/TeamView.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/views/BillingView.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/app/firm/[slug]/layout.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/app/firm/[slug]/dashboard/page.tsx`
- (+ 6 more firm-scoped pages)

**Why not re-export:** Research confirmed re-exports break params shapes, layout context, and metadata. Extracting shared views is cleaner.

**Step 1: For each existing page, extract the UI into a View component**

Pattern (using dashboard as example):

```tsx
// components/views/DashboardView.tsx
"use client"
// ... move ALL the existing dashboard JSX/logic here (everything inside the
// AppLayout wrapper). Do NOT include the AppLayout wrapper itself.
// Accept optional firmSlug prop
export default function DashboardView({ firmSlug }: { firmSlug?: string }) {
  // ... existing dashboard code, using firmSlug for API calls if provided
  // PageHeader, stats, table, activity feed — all moved here
}
```

```tsx
// app/dashboard/page.tsx (original — now thin wrapper)
"use client"
import AppLayout from "@/layouts/AppLayout"
import DashboardView from "@/components/views/DashboardView"
export default function DashboardPage() {
  return (
    <AppLayout title="Dashboard">
      <DashboardView />
    </AppLayout>
  )
}
```

```tsx
// app/firm/[slug]/dashboard/page.tsx (firm-scoped — thin wrapper)
"use client"
import { useParams } from "next/navigation"
import AppLayout from "@/layouts/AppLayout"
import DashboardView from "@/components/views/DashboardView"
export default function FirmDashboardPage() {
  const { slug } = useParams<{ slug: string }>()
  return (
    <AppLayout title="Dashboard">
      <DashboardView firmSlug={slug} />
    </AppLayout>
  )
}
```

**Step 2: Create firm layout with FirmThemeProvider**

> **v3 FIX:** This layout wraps with FirmThemeProvider only (not AppLayout).
> AppLayout is applied per-page in the thin wrappers above, which matches
> the existing pattern where each page includes `<AppLayout title="...">`.

```tsx
// app/firm/[slug]/layout.tsx
"use client"

import FirmThemeProvider from "@/lib/firm-theme-context"
import { useParams } from "next/navigation"

export default function FirmLayout({ children }: { children: React.ReactNode }) {
  const { slug } = useParams<{ slug: string }>()

  return (
    <FirmThemeProvider firmSlug={slug}>
      {children}
    </FirmThemeProvider>
  )
}
```

> **Note:** Using `useParams()` instead of the `params` prop is the most future-proof approach per research (works in Next.js 14, 15, and 16 without changes).

**Step 3: Repeat the view extraction for all 7 pages**

Apply the same pattern to: Matters, MatterDetail, Settings, Precedents, Team, Billing.

**Step 4: Commit**

```bash
git add frontend/components/views/ frontend/app/firm/ frontend/app/dashboard/page.tsx frontend/app/matters/ frontend/app/settings/page.tsx frontend/app/precedents/page.tsx frontend/app/team/page.tsx frontend/app/billing/page.tsx
git commit -m "$(cat <<'EOF'
feat: extract shared views and create firm-scoped routes

Moves page UI into /components/views/ components. Original
pages become thin wrappers. Firm-scoped pages at /firm/[slug]/
pass firmSlug. Uses useParams() for future-proof param access.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Update Sidebar and AppLayout to be firm-slug-aware

**Files:**
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/Sidebar.tsx`
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/frontend/layouts/AppLayout.tsx`

**Step 1: Make Sidebar firm-aware**

Add optional `firmSlug` prop. When provided, prefix all nav `href` values with `/firm/${firmSlug}`. Show firm logo instead of default if `logoUrl` is available via `useFirmTheme()` context (use try/catch since context may not exist outside firm routes).

Key changes to `Sidebar.tsx`:
- Try to read `useFirmTheme()` — if available, use `firmName`, `logoUrl`, `firmSlug`
- Map navigation hrefs: `item.href` → `/firm/${firmSlug}${item.href}` when inside firm context
- Replace the Scale icon logo with an `<img>` when `logoUrl` is available
- Replace "LexIntel" text with `firmName` when available

**Step 2: Commit**

```bash
git add frontend/components/Sidebar.tsx frontend/layouts/AppLayout.tsx
git commit -m "$(cat <<'EOF'
feat: make Sidebar and AppLayout firm-slug-aware

Sidebar detects firm context via useFirmTheme hook. Prefixes
nav links with /firm/:slug when inside firm routes. Shows
custom firm logo and name when available.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Theme Editor

### Task 11: Create theme presets

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/lib/theme-presets.ts`

8 curated presets as static JSON, each containing full `ThemeConfig` objects (light + dark + typography + layout + shadows). See design doc section 6 for preset names and color palettes:

1. **Lexintel Default** — Current monochrome. DM Sans/DM Serif Display.
2. **Corporate Navy** — Navy #1a2744, gold accent #c9a84c. Playfair Display.
3. **Modern Slate** — Slate #334155, teal accent #0d9488. Inter.
4. **Warm Counsel** — Brown #44403c, amber accent #d97706, cream bg. Lora.
5. **Pacific Blue** — Blue #1e40af, cool grays. Open Sans.
6. **Emerald & Ivory** — Green #166534, ivory bg, gold ring. Merriweather.
7. **Minimal Charcoal** — Near-black #18181b, pure white. Space Grotesk.
8. **Burgundy Classic** — Wine #7f1d1d, stone bg, copper. Crimson Text.

Each preset exports: `{ name, description, preview: { primary, accent, background }, config: ThemeConfig }`.

---

### Task 12: Build color picker component

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/theme-editor/ColorPicker.tsx`

A single color token editor: swatch preview + hex text input + color picker popover. Uses native `<input type="color">` inside a styled popover. Validates hex on blur. Calls `onChange(tokenKey, hexValue)`.

---

### Task 13: Build theme control panel

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/theme-editor/ThemeControlPanel.tsx`

Tabbed panel (using shadcn Tabs) with 4 tabs per design doc section 4:

- **Colors tab:** Groups tokens by category (Brand, Base, Surface, Subtle, Danger, Sidebar, Charts). Each group renders ColorPicker components. Light/dark toggle switches which color set is being edited.
- **Typography tab:** Font family dropdowns (Google Fonts), letter-spacing slider, font preview text.
- **Layout & Shadows tab:** Border radius sliders, shadow controls (color, opacity, blur, spread, offset).
- **Branding tab:** Logo upload (drag & drop), logo preview at sidebar + navbar sizes, firm display name.

Action bar at top: Light/Dark toggle, Preset selector dropdown, Reset button, Save button, unsaved changes indicator.

---

### Task 14: Build live preview panel

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/theme-editor/ThemePreview.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/theme-editor/ComponentShowcase.tsx`
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/theme-editor/MiniDashboard.tsx`

Two-tab preview per design doc section 5:

- **Component Showcase:** Scrollable panel with real shadcn/ui components (buttons, cards, forms, table, dialog, badges, charts, typography samples using all 3 fonts).
- **Mini Dashboard:** Scaled-down (70%) rendering of actual Lexintel layout with sidebar, stats cards, matters table with mock data.

Both tabs respect current light/dark toggle.

---

### Task 15: Build theme editor page

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/app/firm/[slug]/settings/theme/page.tsx`

Resizable two-panel layout: left (40%) ThemeControlPanel, right (60%) ThemePreview.

Key behavior:
- On color change: immediately apply to DOM via `--app-*` variables for live preview
- On save: call `updateFirmTheme()` API
- On reset: call `resetFirmTheme()` API and reapply defaults to DOM
- Unsaved changes detection: compare current state to last saved state
- Uses `useUpdateFirmTheme()` and `useResetFirmTheme()` hooks from Task 7

---

### Task 16: Add theme settings link

**Files:**
- Modify: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/views/SettingsView.tsx`

Add a "Theme & Branding" card/section to the existing settings page that links to `/firm/:slug/settings/theme`. Only visible when `firmSlug` prop is provided.

---

### Task 17: Add admin route guard

**Files:**
- Create: `/Users/smeet/Documents/GitHub/Lexintel/frontend/components/AdminGuard.tsx`

Simple wrapper component that checks user role from context/API. If not admin, shows "Access Denied" message. Wrap the theme editor page with this guard.

---

## Phase 5: Polish & Edge Cases

### Task 18: Fallback theme on API failure

When `getFirmTheme()` fails, FirmThemeProvider should fall back to the default theme (imported from a static constant matching `DEFAULT_THEME_CONFIG` from the backend). Must use `--app-*` variable names in its DOM application logic.

### Task 19: Contrast validation

Add a utility function that validates WCAG 2.0 AA contrast ratios for key token pairs (foreground/background, primary/primary-foreground, etc.). Show warnings in the theme editor when contrast is insufficient.

### Task 20: Responsive theme editor

Make the theme editor work on mobile/tablet: stack control panel above preview panel on small screens. Use CSS Grid with breakpoints.

### Task 21: Integration test

Create an end-to-end test that:
1. Creates a firm via API
2. Updates its theme
3. Navigates to `/firm/:slug/dashboard`
4. Verifies CSS variables are applied to DOM

---

## Summary

| Phase | Tasks | Key Changes from v2 |
|-------|-------|---------------------|
| 1. Database | Tasks 1-3 | **FIX:** ALTER users table (not CREATE), fix down_revision, full seed migration code |
| 2. Backend API | Tasks 4-5 | **FIX:** Full router code with file creation, registration in main.py |
| 3. CSS & Routing | Tasks 5.5-10 | **NEW:** Task 5.5 migrates text-muted → text-muted-foreground (135 instances). FIX: muted-foreground kept at #999999. Full types/services/hooks code in Task 7. AppLayout wrapping fixed in Task 9 |
| 4. Theme Editor | Tasks 11-17 | Expanded from "same as v1" to clear specs per design doc |
| 5. Polish | Tasks 18-21 | Unchanged |
| **Total** | **22 tasks** | **~35 files** |
