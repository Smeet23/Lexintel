# Firm-Level Theming — Design Document

**Date:** 2026-02-15
**Status:** Approved

## Summary

Each law firm on Lexintel gets a fully customizable UI theme — colors, typography, logo, shadows, border radius — applied across the entire application. Firm admins configure this through a comprehensive visual theme editor with live preview.

## Decisions

| Decision | Choice |
|----------|--------|
| Multi-tenancy model | URL path per firm (`/firm/:slug/...`) |
| Theme editor access | Firm admin only |
| Theme scope | Comprehensive (25+ tokens, near tweakcn-level) |
| Live preview | Tabbed: component showcase + mini dashboard |
| Dark mode | Independent light + dark configs |
| Presets | 5-8 curated law firm presets |

---

## 1. Database Schema

### New table: `firms`

| Column | Type | Notes |
|--------|------|-------|
| id | UUID, PK | |
| name | String | Firm display name |
| slug | String, unique | URL-safe identifier |
| logo_url | String, nullable | Azure Blob URL |
| theme_config | JSONB | Full light+dark theme |
| created_at | DateTime | |
| updated_at | DateTime | |
| is_deleted | Boolean | Soft delete |

### Modify `users`

- Add `firm_id` (UUID, FK → firms, nullable)
- Add `role` enum: `admin` | `partner` | `associate` | `paralegal`

### Modify `matters`

- Add `firm_id` (UUID, FK → firms)

### `theme_config` JSONB structure

```json
{
  "light": {
    "background": "#fafaf8",
    "foreground": "#111111",
    "primary": "#111111",
    "primary-foreground": "#fafaf8",
    "secondary": "#f5f5f3",
    "secondary-foreground": "#111111",
    "accent": "#111111",
    "accent-foreground": "#fafaf8",
    "muted": "#f5f5f3",
    "muted-foreground": "#6b6b6b",
    "destructive": "#dc2626",
    "destructive-foreground": "#ffffff",
    "card": "#ffffff",
    "card-foreground": "#111111",
    "popover": "#ffffff",
    "popover-foreground": "#111111",
    "border": "#e5e4e2",
    "input": "#e5e4e2",
    "ring": "#111111",
    "sidebar-background": "#fafaf8",
    "sidebar-foreground": "#111111",
    "sidebar-accent": "#f5f5f3",
    "sidebar-border": "#e5e4e2",
    "chart-1": "#111111",
    "chart-2": "#444444",
    "chart-3": "#777777",
    "chart-4": "#aaaaaa",
    "chart-5": "#dddddd"
  },
  "dark": {
    "background": "#111111",
    "foreground": "#fafaf8",
    "...": "..."
  },
  "typography": {
    "font-sans": "DM Sans",
    "font-serif": "DM Serif Display",
    "font-mono": "JetBrains Mono"
  },
  "layout": {
    "radius": "0.5rem",
    "spacing": "0.25rem"
  },
  "shadows": {
    "color": "#000000",
    "opacity": "0.08",
    "blur": "8",
    "spread": "0",
    "offset-x": "0",
    "offset-y": "2"
  }
}
```

---

## 2. API Endpoints

### Firm Management
- `POST /api/firms` — Create firm
- `GET /api/firms/:slug` — Get firm details + theme
- `PUT /api/firms/:slug` — Update firm (admin only)
- `DELETE /api/firms/:slug` — Soft delete (admin only)

### Theme
- `GET /api/firms/:slug/theme` — Get theme config (for page load)
- `PUT /api/firms/:slug/theme` — Update full theme (admin only, Pydantic validated)
- `POST /api/firms/:slug/theme/logo` — Upload logo to Azure Blob (admin only)
- `POST /api/firms/:slug/theme/reset` — Reset to default (admin only)

### Members
- `POST /api/firms/:slug/members` — Invite user (admin only)
- `PUT /api/firms/:slug/members/:user_id/role` — Change role (admin only)
- `GET /api/firms/:slug/members` — List members

### Auth Changes
- Login response includes `firm_slug` and `role`
- JWT payload gets `firm_id` and `role` claims
- Middleware validates user belongs to firm in URL path

---

## 3. Frontend Theme Application

### Flow
1. User navigates to `/firm/:slug/...`
2. Next.js middleware resolves firm slug
3. Layout fetches `GET /firms/:slug/theme` (cached via React Query)
4. `FirmThemeProvider` sets CSS variables on `document.documentElement`
5. All shadcn/ui components re-theme automatically (no component changes)

### FirmThemeProvider responsibilities
- Apply all color tokens as `--color-*` CSS variables to `:root`
- Manage light/dark mode toggle (swap color set, toggle `.dark` class)
- Dynamically load Google Fonts via `<link>` injection
- Expose `firm.logoUrl` via context for sidebar/navbar
- Store dark mode preference in `localStorage`

### Fallback strategy
- If API fails: use current default theme from `globals.css`
- Loading skeleton while theme fetches to prevent FOUC

---

## 4. Theme Editor UI

**Route:** `/firm/:slug/settings/theme` (admin only)

### Layout
Resizable two-panel: left (40%) controls, right (60%) preview.

### Control Panel Tabs

**Tab 1 — Colors** (by category):
- Brand: primary, primary-fg, accent, accent-fg
- Base: background, foreground, secondary, secondary-fg
- Surface: card, card-fg, popover, popover-fg
- Subtle: muted, muted-fg, border, input, ring
- Danger: destructive, destructive-fg
- Sidebar: sidebar-bg, sidebar-fg, sidebar-accent, sidebar-border
- Charts: chart-1 through chart-5

Each: swatch + hex input + color picker popover.

**Tab 2 — Typography:**
- Font dropdowns (sans, serif, mono) from Google Fonts
- Letter-spacing slider
- Font preview text

**Tab 3 — Layout & Shadows:**
- Border radius slider (0 → 1.5rem)
- Spacing slider (0.125rem → 0.5rem)
- Shadow controls (color, opacity, blur, spread, offset-x, offset-y)
- Shadow level preview (2xs → 2xl)

**Tab 4 — Branding:**
- Logo upload (drag & drop, max 2MB, PNG/SVG/JPEG)
- Logo preview at sidebar + navbar sizes
- Firm display name

### Action Bar
- Light/Dark mode toggle (switches editing mode)
- Preset selector dropdown
- Reset to default
- Save button
- Unsaved changes indicator

---

## 5. Live Preview Panel

### Tab 1 — Component Showcase
Scrollable panel with real shadcn/ui components:
- Buttons (primary, secondary, outline, ghost, destructive)
- Cards (header, content, footer, badges)
- Forms (input, textarea, select, switch, checkbox)
- Data display (table, progress bar, badges)
- Dialog trigger
- Mini sidebar, tabs, breadcrumbs
- Chart with 5 color tokens
- Typography (h1-h4, body, muted, links, all 3 fonts)

### Tab 2 — Mini Dashboard
Scaled-down (70%) rendering of actual Lexintel layout:
- Sidebar with logo + nav
- Stats cards
- Matters table with status badges
- Mock data, real components

Both tabs respect current light/dark toggle.

---

## 6. Theme Presets

8 curated presets stored as static JSON in frontend:

1. **Lexintel Default** — Current monochrome. DM Sans/DM Serif Display.
2. **Corporate Navy** — Navy #1a2744, gold accent #c9a84c. Playfair Display.
3. **Modern Slate** — Slate #334155, teal accent #0d9488. Inter.
4. **Warm Counsel** — Brown #44403c, amber accent #d97706, cream bg. Lora.
5. **Pacific Blue** — Blue #1e40af, cool grays. Open Sans.
6. **Emerald & Ivory** — Green #166534, ivory bg, gold ring. Merriweather.
7. **Minimal Charcoal** — Near-black #18181b, pure white. Space Grotesk.
8. **Burgundy Classic** — Wine #7f1d1d, stone bg, copper. Crimson Text.

---

## 7. Implementation Phases

### Phase 1 — Database & Multi-tenancy Foundation
- Alembic migration: firms table, firm_id + role on users, firm_id on matters
- SQLAlchemy Firm model
- Seed script for default firm
- Update existing queries to filter by firm_id

### Phase 2 — API Layer
- Firm CRUD + Pydantic theme validation schemas
- Logo upload (Azure Blob)
- Auth middleware: firm resolution + role checking
- JWT claims: firm_id + role

### Phase 3 — Frontend Routing & Theme Provider
- Route restructure: `/firm/:slug/*`
- FirmThemeProvider context
- Dynamic Google Fonts loader
- Sidebar/navbar logo integration
- Post-login redirect to `/firm/:slug/dashboard`

### Phase 4 — Theme Editor
- Settings page at `/firm/:slug/settings/theme`
- Control panel (colors, typography, layout, branding tabs)
- Preset system
- Live preview: component showcase + mini dashboard
- Save/reset/unsaved-changes logic
- Admin route guard

### Phase 5 — Polish & Edge Cases
- Fallback theme on API failure
- Loading skeleton during theme fetch
- Contrast validation (prevent white-on-white)
- Mobile-responsive editor
