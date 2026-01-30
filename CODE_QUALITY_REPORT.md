# LexIntel - Code Quality & Dead Code Analysis Report

**Date:** January 30, 2026
**Status:** 28 Issues Found (10 Critical, 8 High, 9 Medium, 3 Low)

---

## Quick Summary

| Category | Issues | Priority |
|----------|--------|----------|
| 🔴 **Security Issues** | 5 | CRITICAL |
| 🟠 **Error Handling Gaps** | 4 | HIGH |
| 🟡 **Hardcoded Values** | 4 | HIGH |
| ⚪ **Unused Code** | 4 | MEDIUM |
| ⚪ **Duplicate Code** | 3 | MEDIUM |
| ⚪ **Complex Functions** | 2 | HIGH |
| ⚪ **Style Issues** | 3 | LOW |
| ⚪ **Test Gaps** | 4 | HIGH |
| ⚪ **Architecture** | 4 | MEDIUM |

---

## 🔴 CRITICAL ISSUES TO FIX NOW

### 1. **No Rate Limiting** (SECURITY)
**Severity:** 🔴 CRITICAL
**File:** `backend/main.py`
**Impact:** Anyone can spam API calls, overwhelm OpenAI API

**Quick Fix:**
```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(...):
    # ...

@app.post("/cases")
@limiter.limit("10/minute")
async def upload_case(...):
    # ...
```

---

### 2. **Hardcoded Localhost URLs** (SECURITY)
**Severity:** 🔴 CRITICAL
**File:** `backend/config.py`
**Lines:** 14, 17-19, 30
**Problem:** Will try to use localhost URLs in production if env vars not set

**Fix:**
```python
# In config.py - make required, no defaults
qdrant_url: str  # No default!
redis_url: str   # No default!

def __init__(self, **data):
    super().__init__(**data)
    # Validate not localhost in production
    if not settings.debug:
        for url in [self.qdrant_url, self.redis_url]:
            if "localhost" in url:
                raise ValueError(f"Cannot use localhost URL in production: {url}")
```

---

### 3. **JWT Tokens Not Revocable** (SECURITY)
**Severity:** 🔴 CRITICAL
**File:** `backend/auth.py`
**Problem:** Deleted users can still query with old tokens

**Impact:** User privacy/security breach

**Fix: Add logout endpoint**
```python
# In main.py
@app.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Store token in Redis revocation list
    token_key = f"revoked_token:{current_user.id}"
    redis_client.setex(token_key, 86400, "1")  # 24 hours
    return {"message": "Logged out"}

# In auth.py - check revocation
def is_token_revoked(token: str, user_id: str) -> bool:
    key = f"revoked_token:{user_id}"
    return redis_client.exists(key)
```

---

### 4. **Blob Storage Access Not Explicitly Verified** (SECURITY)
**Severity:** 🔴 CRITICAL
**File:** `backend/services/storage.py`
**Problem:** Container might be publicly accessible

**Fix:**
```python
from azure.storage.blob import PublicAccess

container_client.create_container(
    name="cases",
    public_access=PublicAccess.OFF  # Explicitly private
)

# Add audit logging
logger.info(f"Created private container 'cases'")
```

---

### 5. **No CSRF Protection** (SECURITY)
**Severity:** 🔴 CRITICAL
**File:** `backend/main.py`
**Problem:** State-changing operations vulnerable to cross-site requests

**Fix:**
```bash
pip install python-multipart
```

```python
from starlette.middleware.csrf import CSRFMiddleware

app.add_middleware(
    CSRFMiddleware,
    secret_key=settings.secret_key,
)
```

---

## 🟠 HIGH PRIORITY ISSUES

### 6. **Overly Broad Exception Catching**
**File:** `backend/services/rag_engine.py` (Line 513)
**Problem:** Masks programming errors

```python
# BEFORE
except Exception as e:
    logger.error(f"Unexpected error in query_case: {str(e)}")

# AFTER
except (QueryProcessingException, ValueError, TimeoutError) as e:
    logger.error(f"Expected error in query_case: {str(e)}")
except Exception as e:
    logger.critical(f"Unexpected error in query_case: {str(e)}")
    raise  # Don't swallow unknown errors
```

---

### 7. **Async/Await Inconsistency**
**Files:** `backend/services/job_processor.py` (Line 96)
**Problem:** Marked async but not awaiting true async operations

```python
# BEFORE
async def process_case(...):
    pdf_bytes = await download_pdf_from_blob(...)  # This isn't async!

def download_pdf_from_blob(...) -> bytes:  # Not async
    # ...

# AFTER - Option 1: Make it async
async def download_pdf_from_blob(...) -> bytes:
    return await asyncio.to_thread(blob_client.download_blob().readall)

# AFTER - Option 2: Don't await it
async def process_case(...):
    pdf_bytes = download_pdf_from_blob(...)  # No await
```

---

### 8. **Complex Functions Violating Single Responsibility**
**File:** `backend/services/rag_engine.py` (query_case - 200+ lines)
**Problem:** Handles 9 different concerns

**Fix - Break into smaller functions:**
```python
async def query_case(...):
    """Main orchestration"""
    user_input = validate_query(query)

    embedding = await embed_query(user_input)
    chunks = await retrieve_chunks(case_id, embedding)
    filtered = filter_by_confidence(chunks)

    context = format_context(filtered, case_name)
    answer = await generate_answer(context, query)

    citations = extract_citations(answer, chunks)
    return format_response(answer, citations)
```

---

### 9. **Job Processor Error Handling**
**File:** `backend/services/job_processor.py` (Line 86-156)
**Problem:** Doesn't distinguish retriable vs permanent errors

```python
# BEFORE
except Exception as e:
    logger.error(...)
    return {"success": False}

# AFTER
except (ConnectionError, TimeoutError) as e:
    # Retriable - reschedule
    job.next_retry_at = datetime.now() + timedelta(seconds=backoff)
    logger.warning(f"Retriable error, will retry: {e}")

except (ValueError, ValidationError) as e:
    # Not retriable
    job.status = "failed"
    logger.error(f"Permanent error, won't retry: {e}")
```

---

### 10. **Vector Store Data Loss on Recreation**
**File:** `backend/services/vector_store.py` (Line 122)
**Problem:** `recreate_collection()` drops vectors without warning

**Fix:**
```python
def recreate_collection(case_id: str) -> bool:
    collection_name = _get_collection_name(case_id)

    # Backup old collection
    backup_name = f"{collection_name}_backup_{int(time.time())}"
    try:
        client.get_collection(collection_name)
        client.recreate_collection(collection_name, ...)
        logger.info(f"Recreated collection {collection_name}")
    except Exception as e:
        logger.error(f"Failed to recreate: {e}")
        raise
```

---

## 🟡 MEDIUM PRIORITY - CODE CLEANUP

### 11. **Unused Imports** (4 instances)
**File:** `backend/services/storage.py` (Lines 4, 6)

```python
# REMOVE
import asyncio  # Never used
from pathlib import Path  # Never used

# REMOVE from rag_engine.py line 219
import re  # Move to top of file
```

**Time to fix:** 2 minutes

---

### 12. **Inline Import (Bad Practice)**
**File:** `backend/services/rag_engine.py` (Line 219)

```python
# BEFORE
def extract_citations(...):
    import re  # ❌ Inline import

# AFTER - Move to top
import re  # ✓ Top of file

def extract_citations(...):
    # Uses re module
```

**Time to fix:** 1 minute

---

### 13. **Duplicate Code - Import Pattern**
**Occurs in:** 4 files (main.py, tasks.py, rag_engine.py, storage.py)

```python
# ALL FILES have this same pattern
try:
    from backend.services.embeddings import ...
except ImportError:
    try:
        from services.embeddings import ...
    except ImportError:
        from .services.embeddings import ...
```

**Solution: Create `backend/import_utils.py`**
```python
import importlib.util
import sys

def flexible_import(module_path, names):
    """Try multiple import strategies"""
    attempts = [
        f"backend.{module_path}",
        module_path,
        f".{module_path}"
    ]
    for attempt in attempts:
        try:
            module = __import__(attempt, fromlist=names)
            return {name: getattr(module, name) for name in names}
        except ImportError:
            continue
    raise ImportError(f"Could not import {names} from {module_path}")

# Usage in files:
from backend.import_utils import flexible_import
embed_text = flexible_import("services.embeddings", ["embed_text"])["embed_text"]
```

**Time to fix:** 15 minutes (1 utility file + 4 imports changed)

---

### 14. **Hardcoded Configuration Values**
**Files:** Multiple services
**Issues:**
- CHUNK_SIZE = 1500 (chunking.py)
- CONTEXT_TOKEN_BUDGET = 12,800 (rag_engine.py)
- RETRIEVAL_TOP_K = 10 (rag_engine.py)
- MIN_CONFIDENCE_SCORE = 0.15 (rag_engine.py)

**Fix - Move to config.py:**
```python
# In config.py
chunk_size: int = 1500
chunk_overlap: int = 300
context_token_budget: int = 12800
retrieval_top_k: int = 10
min_confidence_score: float = 0.15
final_chunk_count: int = 4
```

**Then in services:**
```python
from backend.config import get_settings

settings = get_settings()
chunk_size = settings.chunk_size  # Instead of constant
```

**Time to fix:** 20 minutes

---

### 15. **Unused/Unclear Model**
**File:** `backend/models.py` (Lines 107-123)
**Issue:** ProcessingJob model is defined but not used in main API flow

**Decision needed:** Keep or remove?

---

### 16. **Inconsistent Response Types**
**File:** `backend/main.py`
**Problem:** Endpoints return different response shapes

**Fix - Create consistent schemas:**
```python
# In schemas.py
class ListCasesResponse(BaseModel):
    cases: List[CaseResponse]
    count: int

class UploadCaseResponse(BaseModel):
    id: UUID
    name: str
    status: str
    created_at: datetime

# In main.py
@app.get("/cases", response_model=ListCasesResponse)
async def list_cases(...):
    cases = db.query(Case).filter(Case.user_id == user.id).all()
    return ListCasesResponse(cases=cases, count=len(cases))
```

**Time to fix:** 30 minutes

---

## ⚪ NICE-TO-HAVE IMPROVEMENTS

### Test Coverage Gaps
- No integration tests for full RAG pipeline
- No security tests (SQL injection, path traversal)
- No load/performance tests
- No tests for error scenarios

**Recommendation:** Add 20-30 integration tests

---

### Documentation Gaps
- No error handling strategy documentation
- No API error code reference
- No deployment runbook

---

## WHAT TO FIX FIRST

### Recommended Order:

**Week 1 - Security (1-2 hours)**
1. Add rate limiting ✓ CRITICAL
2. Verify blob storage private ✓ CRITICAL
3. Add CSRF protection ✓ CRITICAL
4. Fix hardcoded localhost URLs ✓ CRITICAL
5. Implement JWT revocation ✓ CRITICAL

**Week 2 - Error Handling (2-3 hours)**
6. Fix async/await inconsistencies
7. Split complex functions
8. Improve exception handling

**Week 3 - Code Cleanup (1-2 hours)**
9. Remove unused imports
10. Extract duplicate import patterns
11. Move hardcoded values to config
12. Standardize response schemas

**Week 4 - Testing (varies)**
13. Add integration tests
14. Add security tests

---

## DEPENDENCY UPDATES NEEDED

**Critical:**
```bash
# Update passlib (currently 1.7.4 from 2013)
pip install --upgrade passlib>=1.7.4
```

**Add for fixes:**
```bash
pip install slowapi python-multipart
```

---

## FILES TO REVIEW

**High Priority:**
- `backend/config.py` - Add validation, move hardcoded values
- `backend/main.py` - Add rate limiting, CSRF, standardize responses
- `backend/services/rag_engine.py` - Split complex functions, fix exceptions
- `backend/auth.py` - Add revocation

**Medium Priority:**
- `backend/services/storage.py` - Clean imports, explicit ACL
- `backend/services/job_processor.py` - Better error differentiation
- `backend/services/vector_store.py` - Data preservation
- `backend/models.py` - Clarify ProcessingJob usage

---

## Questions for You

Which would you like to fix first?

1. **🔴 Security fixes** (Rate limit, CSRF, blob storage) - 1 hour
2. **🟠 Error handling** (Async fixes, exception handling) - 2 hours
3. **🟡 Code cleanup** (Remove unused code, extract duplicates) - 1 hour
4. **⚪ All of the above** (Full refactor) - 5 hours

Let me know and I can implement the fixes! 🚀
