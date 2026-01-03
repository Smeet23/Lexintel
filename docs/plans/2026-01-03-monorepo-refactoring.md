# Monorepo Structure Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor monorepo to have clean, separation-of-concerns architecture where backend contains only API/business logic, workers contains all Celery task definitions, and shared contains database models and utilities.

**Architecture:**
- Remove duplicate model definitions and worker tasks from backend
- Backend imports directly from shared for models and schemas
- Workers service (`/apps/workers/app/`) owns all Celery task definitions
- Consistent `app/` package structure across both backend and workers services
- Clear module organization: API routes → Services → Database models (all in shared)

**Tech Stack:** FastAPI, Celery, SQLAlchemy, Pydantic, PostgreSQL, Redis

---

## Phase 1: Analysis & Preparation

### Task 1: Document Current Import Dependencies

**Files to analyze:**
- `/apps/backend/app/` - All Python files
- `/apps/backend/tests/` - All test files
- `/apps/workers/src/` - All Python files

**Step 1: Run grep to identify all import patterns**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
grep -r "from app\." apps/backend/ --include="*.py" | wc -l
grep -r "from app.models" apps/backend/ --include="*.py" | wc -l
grep -r "from app.workers" apps/backend/ --include="*.py" | wc -l
grep -r "from shared" apps/ --include="*.py" | wc -l
```

Expected output: ~50+ imports from app.*, ~20+ from app.models, ~3-5 from app.workers

**Step 2: Create a mapping document (for reference, not a file)**

Map out:
- Which files import `from app.models` (need to change to `from shared`)
- Which files import `from app.workers.tasks` (need to remove)
- Which files import `from app.database` (keep as-is)

**Step 3: Commit checkpoint**

```bash
git status  # Should be clean
```

---

## Phase 2: Backend Cleanup & Refactoring

### Task 2: Update Backend Models Import Wrapper

**Goal:** Make `app/models/__init__.py` import from shared to maintain backwards compatibility during transition

**Files:**
- Modify: `/apps/backend/app/models/__init__.py`

**Step 1: Read current file**

```bash
cat /apps/backend/app/models/__init__.py
```

**Step 2: Update to import from shared**

Replace the file with:

```python
"""Re-export models from shared package for backwards compatibility."""

from shared.models import (
    Case,
    CaseStatus,
    Document,
    DocumentChunk,
    DocumentType,
    ProcessingStatus,
    ChatConversation,
    ChatMessage,
    Base,
    TimestampMixin,
)

__all__ = [
    "Case",
    "CaseStatus",
    "Document",
    "DocumentChunk",
    "DocumentType",
    "ProcessingStatus",
    "ChatConversation",
    "ChatMessage",
    "Base",
    "TimestampMixin",
]
```

**Step 3: Verify imports work**

```bash
cd /apps/backend
python -c "from app.models import Case, Document; print('✓ Imports work')"
```

Expected: `✓ Imports work`

**Step 4: Commit**

```bash
git add app/models/__init__.py
git commit -m "refactor: update models to import from shared package"
```

---

### Task 3: Remove Backend Workers Module (Delete Old Task Definitions)

**Goal:** Remove `/apps/backend/app/workers/` folder since workers service owns all tasks

**Files:**
- Delete: `/apps/backend/app/workers/` (entire folder)

**Step 1: Verify what's in the workers folder**

```bash
ls -la /apps/backend/app/workers/
```

**Step 2: Check what imports this module**

```bash
grep -r "from app.workers" /apps/backend/ --include="*.py"
grep -r "import app.workers" /apps/backend/ --include="*.py"
```

Expected files to update:
- `/apps/backend/app/api/documents.py` - imports `process_document_pipeline`
- `/apps/backend/tests/test_extraction_integration.py` - imports `_extract_and_chunk_document`

**Step 3: Remove the folder**

```bash
rm -rf /apps/backend/app/workers/
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove duplicate worker task definitions from backend"
```

---

### Task 4: Update Backend API to Queue Tasks via Celery (Not Direct Import)

**Goal:** Change `documents.py` to queue tasks via Celery instead of importing from app.workers

**Files:**
- Modify: `/apps/backend/app/api/documents.py` (around line 16)

**Step 1: Read the current imports section**

```bash
head -30 /apps/backend/app/api/documents.py
```

**Step 2: Find the problematic import**

Look for: `from app.workers.tasks import process_document_pipeline`

**Step 3: Replace with Celery queue call**

Change from:
```python
from app.workers.tasks import process_document_pipeline
# ... later in code:
process_document_pipeline.delay(document_id)
```

To:
```python
# Import celery_app instead of tasks
from app.celery_app import celery_app
# ... later in code:
celery_app.send_task('workers.document_extraction.process_document_pipeline',
                     args=[document_id])
```

**Step 4: Verify the celery_app configuration**

```bash
grep -A5 "task_routes\|autodiscover" /apps/backend/app/celery_app.py
```

**Step 5: Update the documents.py file**

Read and modify the file to use celery_app.send_task instead of direct import

**Step 6: Test the API still works**

```bash
cd /apps/backend
python -m pytest tests/test_progress_sse.py -v
```

Expected: Tests pass or show clear error about task routing

**Step 7: Commit**

```bash
git add app/api/documents.py
git commit -m "refactor: queue document processing tasks via celery instead of direct import"
```

---

### Task 5: Update Backend Tests to Remove Worker Imports

**Goal:** Remove test imports from `app.workers.tasks` since module is deleted

**Files:**
- Modify: `/apps/backend/tests/test_extraction_integration.py`

**Step 1: Read the test file**

```bash
head -40 /apps/backend/tests/test_extraction_integration.py
```

**Step 2: Find worker imports**

Look for: `from app.workers.tasks import _extract_and_chunk_document`

**Step 3: Update the test**

Either:
- Remove the test if it directly tests worker internals (worker tests should be in workers/tests/)
- Or mock the task call if testing API behavior

For now, comment out or remove worker-specific tests. The actual worker logic is tested in `/apps/workers/tests/`

**Step 4: Run tests**

```bash
cd /apps/backend
python -m pytest tests/ -v --tb=short
```

Expected: Tests pass or show only expected failures for TODO items

**Step 5: Commit**

```bash
git add tests/test_extraction_integration.py
git commit -m "refactor: remove worker task imports from backend tests"
```

---

### Task 6: Verify Backend Models Can Be Removed (Pure Re-exports)

**Goal:** Confirm that `/apps/backend/app/models/` can be fully deleted once imports updated

**Step 1: Check if any backend code directly imports from app.models submodules**

```bash
grep -r "from app\.models\.[a-z]" /apps/backend/ --include="*.py"
grep -r "from app\.models import" /apps/backend/ --include="*.py"
```

Expected: All should be caught by our Task 2 changes (importing from shared via app.models wrapper)

**Step 2: Verify the wrapper works**

```bash
cd /apps/backend
python -c "from app.models import Case, Document, DocumentChunk; print('✓ Re-exports work')"
```

Expected: `✓ Re-exports work`

**Step 3: Keep the folder for now**

Don't delete yet - keep the re-export wrapper for backward compatibility. This allows a gradual migration where:
- New code imports directly from shared
- Old code imports from app.models (which proxies to shared)

**Step 4: Commit (status checkpoint)**

```bash
git status
```

Expected: Clean working tree

---

## Phase 3: Workers Service Refactoring

### Task 7: Rename `/apps/workers/src/` to `/apps/workers/app/`

**Goal:** Make workers service have consistent structure with backend

**Files:**
- Rename: `/apps/workers/src/` → `/apps/workers/app/`

**Step 1: Verify src/ structure**

```bash
tree /apps/workers/src/ -L 2
```

**Step 2: Rename the directory**

```bash
cd /apps/workers
mv src app
```

**Step 3: Verify rename worked**

```bash
ls -la /apps/workers/
```

Expected: `app/` folder exists, `src/` does not

**Step 4: Update __main__.py if needed**

The `/apps/workers/app/__main__.py` may have import adjustments needed. Check:

```bash
grep -n "sys.path\|from src\|import src" /apps/workers/app/__main__.py
```

If any `sys.path.insert(0, '../src')` or similar, update to match new location.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: rename workers/src to workers/app for consistency"
```

---

### Task 8: Update Workers Imports After Rename

**Goal:** Fix all import paths in workers that referenced the old `src` module

**Files:**
- Modify: `/apps/workers/app/__main__.py`
- Modify: `/apps/workers/app/workers/document_extraction.py`
- Modify: `/apps/workers/app/lib/*.py`
- Check: `/apps/workers/pyproject.toml`

**Step 1: Find imports referencing sys.path manipulation**

```bash
grep -r "sys.path" /apps/workers/app/ --include="*.py"
grep -r "../src" /apps/workers/ --include="*.py"
```

**Step 2: Update __main__.py**

The entry point file likely has:
```python
import sys
sys.path.insert(0, 'src')
```

Should become:
```python
# No need for sys.path manipulation if using proper imports
```

Or if it's relative path finding:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Step 3: Check worker task imports**

Read `/apps/workers/app/workers/document_extraction.py`:

```bash
head -15 /apps/workers/app/workers/document_extraction.py
```

Look for:
- `sys.path.insert(0, '../../packages/shared/src')` - this might be fine
- `from src.` imports - change to `from app.`
- `from lib` imports - might need updating

**Step 4: Test imports work**

```bash
cd /apps/workers
python -c "from app import celery_app; print('✓ Worker imports work')"
```

**Step 5: Commit**

```bash
git add app/
git commit -m "refactor: update worker imports after rename to app/"
```

---

### Task 9: Update Workers Celery Task Registration

**Goal:** Ensure Celery can find tasks in the new `app/workers/` location

**Files:**
- Modify: `/apps/workers/app/celery_app.py`

**Step 1: Read the Celery app configuration**

```bash
cat /apps/workers/app/celery_app.py
```

**Step 2: Check task autodiscovery configuration**

Look for `autodiscover_tasks()` call. It should reference `'app.workers'`:

```python
app.autodiscover_tasks(['app.workers'])  # Should find app/workers/document_extraction.py
```

**Step 3: If using old path, update to use 'app'**

Change from:
```python
app.autodiscover_tasks(['src.workers'])  # Old
```

To:
```python
app.autodiscover_tasks(['app.workers'])  # New
```

**Step 4: Verify Celery can find tasks**

```bash
cd /apps/workers
python -c "from app.celery_app import celery_app; print(list(celery_app.tasks.keys()))"
```

Expected: Should show task names like `'app.workers.document_extraction.extract_text_from_document'`

**Step 5: Commit**

```bash
git add app/celery_app.py
git commit -m "refactor: update celery task autodiscovery for new app/ structure"
```

---

## Phase 4: Database & Configuration

### Task 10: Verify Shared Database Imports in Backend

**Goal:** Ensure backend uses shared database configuration correctly

**Files:**
- Check: `/apps/backend/app/database.py`

**Step 1: Read database.py**

```bash
cat /apps/backend/app/database.py
```

**Step 2: Verify it imports from shared**

Should look like:
```python
from shared.database import async_session, engine, Base, init_db

__all__ = ['async_session', 'engine', 'Base', 'init_db', 'get_db']

async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            pass
```

**Step 3: If not importing from shared, update it**

Replace the entire file with:

```python
"""Database configuration and session management.

Re-exports from shared and provides FastAPI dependency.
"""

from shared.database import async_session, engine, Base, init_db

__all__ = ['async_session', 'engine', 'Base', 'init_db', 'get_db']


async def get_db():
    """FastAPI dependency for database sessions."""
    async with async_session() as session:
        try:
            yield session
        finally:
            pass
```

**Step 4: Test imports**

```bash
cd /apps/backend
python -c "from app.database import async_session, get_db, Base; print('✓ Database imports work')"
```

Expected: `✓ Database imports work`

**Step 5: Commit**

```bash
git add app/database.py
git commit -m "refactor: verify database uses shared configuration"
```

---

## Phase 5: Docker & Entry Points

### Task 11: Update Docker Entry Points

**Goal:** Ensure docker-compose and Dockerfile use correct entry points after refactoring

**Files:**
- Check: `/docker-compose.yml`
- Check: `/apps/backend/Dockerfile`
- Check: `/apps/workers/Dockerfile`

**Step 1: Read docker-compose.yml**

```bash
grep -A10 "backend:\|celery-worker:\|workers:" /docker-compose.yml | head -40
```

**Step 2: Check backend service entry point**

Should be something like:
```yaml
backend:
  build: ./apps/backend
  command: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If using old `src` reference, update.

**Step 3: Check workers service entry point**

Should be:
```yaml
workers:
  build: ./apps/workers
  command: python -m app --config prod
```

Or:
```yaml
command: python -m app.__main__
```

Update if still using `src/`.

**Step 4: Check Dockerfiles**

```bash
grep "WORKDIR\|ENTRYPOINT\|CMD" /apps/backend/Dockerfile
grep "WORKDIR\|ENTRYPOINT\|CMD" /apps/workers/Dockerfile
```

Both should reference correct paths.

**Step 5: Update docker-compose.yml if needed**

If workers entry point was `python -m workers`, it should now be:
```yaml
command: python -m app
```

**Step 6: Commit**

```bash
git add docker-compose.yml apps/backend/Dockerfile apps/workers/Dockerfile
git commit -m "refactor: update docker entry points for new structure"
```

---

## Phase 6: Testing & Verification

### Task 12: Run Backend Tests

**Goal:** Verify backend still works after refactoring

**Files:**
- Test: `/apps/backend/tests/`

**Step 1: Install dependencies**

```bash
cd /apps/backend
pip install -e . --quiet
```

**Step 2: Run all backend tests**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | head -100
```

Expected: Tests pass (or fail only on incomplete Phase 4 features like embeddings)

**Step 3: Check for import errors**

```bash
python -m pytest tests/ --collect-only 2>&1 | grep -E "ERROR|ImportError"
```

Expected: No import errors

**Step 4: Commit (checkpoint)**

```bash
git add -A  # Add any generated files from tests
git commit -m "test: verify backend tests pass after refactoring" --allow-empty
```

---

### Task 13: Run Workers Tests

**Goal:** Verify workers service works after refactoring

**Files:**
- Test: `/apps/workers/tests/`

**Step 1: Install dependencies**

```bash
cd /apps/workers
pip install -e . --quiet
```

**Step 2: Run worker tests**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | head -100
```

Expected: Tests pass or show celery-related warnings (acceptable)

**Step 3: Verify Celery tasks can be discovered**

```bash
cd /apps/workers
python -c "from app.celery_app import celery_app; tasks = celery_app.tasks; print(f'Found {len(tasks)} tasks'); [print(f'  - {t}') for t in sorted(tasks.keys()) if 'app.workers' in t]"
```

Expected: Shows registered task names

**Step 4: Commit**

```bash
git add -A
git commit -m "test: verify workers tests pass after refactoring" --allow-empty
```

---

### Task 14: Integration Test - Docker Compose

**Goal:** Verify services can run together with docker-compose

**Step 1: Build images**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
docker-compose build --no-cache 2>&1 | tail -20
```

Expected: All images build successfully

**Step 2: Start services (short test)**

```bash
docker-compose up -d postgres redis 2>&1
sleep 3
docker-compose logs postgres redis | tail -10
```

Expected: Services start without errors

**Step 3: Start backend**

```bash
docker-compose up -d backend 2>&1
sleep 5
docker-compose logs backend | tail -20
```

Expected: Backend starts, connects to database

**Step 4: Test health endpoint**

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Expected: JSON response with health status

**Step 5: Clean up**

```bash
docker-compose down
```

**Step 6: Commit**

```bash
git add -A
git commit -m "test: verify docker-compose services work after refactoring" --allow-empty
```

---

## Phase 7: Documentation Updates

### Task 15: Update Project Documentation

**Goal:** Update CLAUDE.md and BACKEND.md to reflect new structure

**Files:**
- Modify: `/claude.md`
- Modify: `/docs/BACKEND.md`
- Modify: `/docs/WORKERS.md`

**Step 1: Update Architecture Section in CLAUDE.md**

Update the directory structure diagram to show:
```
apps/backend/
  ├── app/
  │   ├── main.py
  │   ├── config.py
  │   ├── schemas/
  │   ├── api/
  │   └── services/

apps/workers/
  ├── app/
  │   ├── __main__.py
  │   ├── workers/
  │   └── lib/
```

**Step 2: Update BACKEND.md import patterns**

Change documentation examples from:
```python
from app.models import Case
from app.workers.tasks import extract_text
```

To:
```python
from shared.models import Case
# Tasks are in workers service, queued via Celery
celery_app.send_task('app.workers.document_extraction.extract_text_from_document')
```

**Step 3: Update WORKERS.md**

Update references to `src/` → `app/` in workers service documentation.

**Step 4: Commit**

```bash
git add claude.md docs/BACKEND.md docs/WORKERS.md
git commit -m "docs: update architecture documentation after monorepo refactoring"
```

---

## Summary

This refactoring achieves:

✅ **Clean Separation of Concerns**
- Backend: API + business logic only
- Workers: All Celery task definitions
- Shared: Models, schemas, utilities

✅ **Consistent Structure**
- Both backend and workers have `app/` package
- Clear module hierarchy
- No duplicate code

✅ **Proper Dependencies**
- Backend imports from shared and workers (via Celery)
- Workers import from shared
- Shared has no dependencies on backend/workers

✅ **Scalability**
- Easy to add new services: `apps/scheduler/`, `apps/api/`, etc.
- All follow same pattern

**Total commits expected:** ~12-15 focused commits
**Total estimated changes:** ~20-30 files touched (imports + deletions)
**Risk level:** Low (all tests pass before/after, isolated changes)

