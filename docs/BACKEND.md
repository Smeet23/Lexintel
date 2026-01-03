# Backend API Architecture & Patterns

> FastAPI backend implementation guide for LexIntel

---

## 📐 API Architecture

### Directory Structure
```
backend/app/
├── api/                      # API routers
│   ├── __init__.py
│   ├── cases.py              # ✅ Cases CRUD
│   ├── documents.py          # ✅ Document upload/management
│   ├── search.py             # TODO: Full-text + semantic search
│   └── chat.py               # TODO: Chat/RAG endpoints
│
├── schemas/                  # Pydantic request/response models
│   ├── case.py
│   ├── document.py
│   └── chat.py
│
├── services/                 # Business logic layer
│   ├── storage.py            # ✅ File storage
│   ├── search.py             # TODO: Search logic
│   └── embeddings.py         # TODO: Embeddings service
│
├── models/                   # SQLAlchemy ORM
├── database.py               # Session factory
├── config.py                 # Settings
└── main.py                   # FastAPI app
```

---

## 🏗️ Standard Endpoint Pattern

### Template
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database import get_db
from app.models import Model
from app.schemas import ResponseSchema
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/resources", tags=["resources"])

@router.post("", response_model=ResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_resource(
    create_data: CreateSchema,
    db: AsyncSession = Depends(get_db),
):
    """Create a new resource"""
    try:
        new_obj = Model(id=str(uuid4()), **create_data.dict())
        db.add(new_obj)
        await db.commit()
        await db.refresh(new_obj)

        logger.info(f"[resource] Created: {new_obj.id}")
        return new_obj
    except Exception as e:
        logger.error(f"[resource] Create error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create resource",
        )

@router.get("", response_model=List[ResponseSchema])
async def list_resources(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
):
    """List resources"""
    try:
        stmt = select(Model).offset(skip).limit(limit)
        result = await db.execute(stmt)
        resources = result.scalars().all()
        return resources
    except Exception as e:
        logger.error(f"[resource] List error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list resources",
        )

@router.get("/{resource_id}", response_model=ResponseSchema)
async def get_resource(
    resource_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get resource by ID"""
    try:
        stmt = select(Model).where(Model.id == resource_id)
        result = await db.execute(stmt)
        resource = result.scalar_one_or_none()

        if not resource:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found",
            )

        return resource
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[resource] Get error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get resource",
        )

@router.patch("/{resource_id}", response_model=ResponseSchema)
async def update_resource(
    resource_id: str,
    update_data: UpdateSchema,
    db: AsyncSession = Depends(get_db),
):
    """Update resource"""
    try:
        stmt = select(Model).where(Model.id == resource_id)
        result = await db.execute(stmt)
        resource = result.scalar_one_or_none()

        if not resource:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found",
            )

        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(resource, field, value)

        await db.commit()
        await db.refresh(resource)

        logger.info(f"[resource] Updated: {resource.id}")
        return resource
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[resource] Update error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update resource",
        )

@router.delete("/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_resource(
    resource_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete resource"""
    try:
        stmt = select(Model).where(Model.id == resource_id)
        result = await db.execute(stmt)
        resource = result.scalar_one_or_none()

        if not resource:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found",
            )

        await db.delete(resource)
        await db.commit()

        logger.info(f"[resource] Deleted: {resource_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[resource] Delete error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete resource",
        )
```

---

## 📋 Pydantic Schema Pattern

### Template
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ResourceBase(BaseModel):
    """Shared fields for creation and updates"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

class ResourceCreate(ResourceBase):
    """Schema for creation requests"""
    pass

class ResourceUpdate(BaseModel):
    """Schema for update requests - all optional"""
    name: Optional[str] = None
    description: Optional[str] = None

class ResourceResponse(ResourceBase):
    """Schema for API responses"""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # For ORM mapping
```

---

## 🔗 Integrating Routers in main.py

```python
from app.api import cases, documents, search, chat

# Include routers
app.include_router(cases.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
```

---

## 🚀 Adding a New Endpoint

### Step 1: Create Schema
```python
# app/schemas/resource.py
class ResourceCreate(BaseModel):
    field: str
```

### Step 2: Create API Router
```python
# app/api/resource.py
router = APIRouter(prefix="/resources", tags=["resources"])

@router.post("", response_model=ResourceResponse)
async def create(data: ResourceCreate, db: AsyncSession = Depends(get_db)):
    # Implementation following template above
    pass
```

### Step 3: Include Router
```python
# app/main.py
from app.api import resource
app.include_router(resource.router)
```

### Step 4: Test
```bash
curl -X POST http://localhost:8000/resources \
  -H "Content-Type: application/json" \
  -d '{"field": "value"}'
```

---

## 🔍 Database Query Patterns

### Select Single
```python
stmt = select(Model).where(Model.id == resource_id)
result = await db.execute(stmt)
obj = result.scalar_one_or_none()  # None if not found
```

### Select Multiple
```python
stmt = select(Model).offset(skip).limit(limit)
result = await db.execute(stmt)
objects = result.scalars().all()
```

### Select with Filter
```python
stmt = select(Model).where(
    and_(
        Model.status == "active",
        Model.created_at > datetime.now() - timedelta(days=7)
    )
)
result = await db.execute(stmt)
objects = result.scalars().all()
```

### Create
```python
obj = Model(id=str(uuid4()), **data.dict())
db.add(obj)
await db.commit()
await db.refresh(obj)
```

### Update
```python
obj.field = new_value
await db.commit()
await db.refresh(obj)
```

### Delete
```python
await db.delete(obj)
await db.commit()
```

---

## 📝 Error Handling

### Standard HTTP Exceptions
```python
from fastapi import HTTPException, status

# 400 Bad Request
raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input")

# 404 Not Found
raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")

# 409 Conflict
raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Resource already exists")

# 500 Internal Server Error
raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process")
```

### Always Rollback on Error
```python
try:
    db.add(obj)
    await db.commit()
except Exception as e:
    await db.rollback()  # Important!
    raise HTTPException(status_code=500, detail="Error")
```

---

## 🎯 Logging Standards

Prefix all log messages for clarity:
```python
logger.info(f"[resource] Created: {obj.id}")
logger.error(f"[resource] Create error: {e}")
logger.warning(f"[resource] Invalid input detected")
```

---

## ✅ Checklist for New Endpoints

- [ ] Schema created in `app/schemas/`
- [ ] Router created in `app/api/`
- [ ] Router included in `app/main.py`
- [ ] Logging added with appropriate prefix
- [ ] Error handling with rollback on DB errors
- [ ] HTTPException used instead of generic errors
- [ ] Response model specified on endpoint
- [ ] Status codes correct (201 create, 204 delete, etc.)
- [ ] Tested with curl or similar tool

---

## 📚 References

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **SQLAlchemy Async**: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- **Pydantic**: https://docs.pydantic.dev/
