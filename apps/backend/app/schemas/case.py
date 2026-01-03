from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from shared.models import CaseStatus

class CaseBase(BaseModel):
    name: str
    case_number: str
    practice_area: str
    status: CaseStatus = CaseStatus.ACTIVE
    description: Optional[str] = None

class CaseCreate(CaseBase):
    pass

class CaseUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[CaseStatus] = None
    description: Optional[str] = None

class CaseResponse(CaseBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
