from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_
from app.database import get_db
from shared.models import Case
from app.schemas.case import CaseCreate, CaseResponse, CaseUpdate
from typing import List
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cases", tags=["cases"])

@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    case_create: CaseCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new case"""
    try:
        new_case = Case(id=str(uuid4()), **case_create.dict())
        db.add(new_case)
        await db.commit()
        await db.refresh(new_case)

        logger.info(f"[cases] Created case: {new_case.id}")
        return new_case
    except Exception as e:
        logger.error(f"[cases] Create case error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create case",
        )

@router.get("", response_model=List[CaseResponse])
async def list_cases(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
):
    """List all cases"""
    try:
        stmt = select(Case).offset(skip).limit(limit)
        result = await db.execute(stmt)
        cases = result.scalars().all()
        return cases
    except Exception as e:
        logger.error(f"[cases] List cases error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list cases",
        )

@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get case details"""
    try:
        stmt = select(Case).where(Case.id == case_id)
        result = await db.execute(stmt)
        case = result.scalar_one_or_none()

        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found",
            )

        return case
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[cases] Get case error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get case",
        )

@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    case_update: CaseUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update case"""
    try:
        stmt = select(Case).where(Case.id == case_id)
        result = await db.execute(stmt)
        case = result.scalar_one_or_none()

        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found",
            )

        update_data = case_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(case, field, value)

        await db.commit()
        await db.refresh(case)

        logger.info(f"[cases] Updated case: {case.id}")
        return case
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[cases] Update case error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update case",
        )

@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete case"""
    try:
        stmt = select(Case).where(Case.id == case_id)
        result = await db.execute(stmt)
        case = result.scalar_one_or_none()

        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found",
            )

        await db.delete(case)
        await db.commit()

        logger.info(f"[cases] Deleted case: {case.id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[cases] Delete case error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete case",
        )
