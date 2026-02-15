"""Firm management and theme configuration API endpoints."""

from fastapi import APIRouter, HTTPException, status, Depends
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
    try:
        from database import get_db
        from models import Firm, User
        from schemas import (
            FirmCreate, FirmResponse, ThemeConfigUpdate,
            MemberInvite, MemberRoleUpdate,
        )
        from theme_defaults import DEFAULT_THEME_CONFIG
    except ImportError:
        from .database import get_db
        from .models import Firm, User
        from .schemas import (
            FirmCreate, FirmResponse, ThemeConfigUpdate,
            MemberInvite, MemberRoleUpdate,
        )
        from .theme_defaults import DEFAULT_THEME_CONFIG

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
