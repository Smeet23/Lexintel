# LexIntel File Reference Guide

Complete reference documentation for all LexIntel project files with function signatures, parameters, return values, and usage examples.

## Table of Contents

### Backend Files
- [backend/main.py](#backendmainpy) - FastAPI application entry point
- [backend/config.py](#backendconfigpy) - Configuration management
- [backend/database.py](#backenddatabasepy) - Database session management
- [backend/models.py](#backendmodelspy) - SQLAlchemy ORM models
- [backend/schemas.py](#backendschemaspy) - Pydantic request/response schemas
- [backend/auth.py](#backendauthpy) - JWT authentication utilities
- [backend/validators.py](#backendvalidatorspy) - Input validation functions
- [backend/exceptions.py](#backendexceptionspy) - Custom exception classes
- [backend/services/storage.py](#backendservicesstoragepy) - Azure Blob Storage operations
- [backend/services/chunking.py](#backendserviceschunkingpy) - Document chunking service
- [backend/services/text_extraction.py](#backendservicestext_extractionpy) - Multi-format text extraction
- [backend/services/embeddings.py](#backendservicesembeddingspy) - OpenAI embeddings service
- [backend/services/vector_store.py](#backendservicesvector_storepy) - Qdrant vector database
- [backend/services/rag_engine.py](#backendservicesrag_enginepy) - RAG pipeline orchestration
- [backend/services/job_processor.py](#backendservicesjob_processorpy) - Background job processing

### Frontend Files
- [frontend/app/layout.tsx](#frontendapplayouttsx) - Root layout with providers
- [frontend/app/page.tsx](#frontendapppagetsx) - Home/landing page
- [frontend/app/auth/login/page.tsx](#frontendappauthloginpagetsx) - Login page
- [frontend/app/auth/register/page.tsx](#frontendappauthregisterpagetsx) - Registration page
- [frontend/app/cases/[id]/page.tsx](#frontendappcasesidpagetsx) - Case detail page
- [frontend/app/dashboard/page.tsx](#frontendappdashboardpagetsx) - Dashboard with upload
- [frontend/components/file-upload-zone.tsx](#frontendcomponentsfile-upload-zonetsx) - File upload component
- [frontend/components/navbar.tsx](#frontendcomponentsnavbartsx) - Navigation bar
- [frontend/lib/auth-context.tsx](#frontendlibauth-contexttsx) - Authentication context
- [frontend/lib/query-provider.tsx](#frontendlibquery-providertsx) - React Query provider

---

## Backend Files

### backend/main.py

FastAPI application entry point defining all REST API endpoints for authentication, case management, and RAG queries.

#### Constants

```python
# CORS configuration and settings
settings = get_settings()
cors_origins = get_cors_origins()
```

#### Functions

##### `get_cors_origins() -> list`

**Purpose:** Retrieve and validate allowed CORS origins from environment configuration.

**Parameters:** None

**Return Value:** `list[str]` - List of allowed origin URLs

**Error Handling:** Raises `ValueError` if placeholder domains are found in CORS config

**Usage Example:**
```python
origins = get_cors_origins()  # Returns ['http://localhost:3000', 'https://example.com']
```

---

##### `startup_validation()`

**Purpose:** Validate critical configuration on application startup (async event handler).

**Parameters:** None (async function)

**Return Value:** None

**Error Handling:** Logs warnings for insecure defaults, raises for CORS config errors

**Usage Example:**
```python
# Automatically called on FastAPI startup
# Validates SECRET_KEY and CORS configuration
```

---

##### `health_check()`

**Purpose:** Simple health check endpoint for monitoring.

**Parameters:** None

**Return Value:** `dict` with `{"status": "ok"}`

**Usage Example:**
```python
# GET /health → {"status": "ok"}
```

---

##### `get_current_user(authorization: str = Header(None), db: Session = Depends(get_db)) -> UUID`

**Purpose:** Extract and validate JWT token from Authorization header (dependency).

**Parameters:**
- `authorization` (str, optional): Authorization header value (format: "Bearer {token}")
- `db` (Session): Database session dependency

**Return Value:** `UUID` - User ID from token 'sub' claim

**Error Handling:**
- Raises `HTTPException(401)` if header missing or malformed
- Raises `HTTPException(401)` if token invalid/expired
- Raises `HTTPException(401)` if user not found/deleted

**Usage Example:**
```python
@app.get("/protected")
def protected_endpoint(current_user_id: UUID = Depends(get_current_user)):
    return {"user_id": str(current_user_id)}
```

---

##### `verify_case_ownership(case, user_id: UUID) -> None`

**Purpose:** Verify that a user owns a specific case (authorization check).

**Parameters:**
- `case` (Case): Case model instance
- `user_id` (UUID): User ID to verify

**Return Value:** None

**Error Handling:** Raises `HTTPException(403)` if user doesn't own case

**Usage Example:**
```python
case = db.query(Case).filter(Case.id == case_id).first()
verify_case_ownership(case, current_user_id)  # Raises 403 if unauthorized
```

---

#### Endpoints

##### `POST /auth/register`

**Purpose:** Register new user account.

**Request Body:** `UserCreate` schema
- `email` (EmailStr): Valid email address
- `password` (str): 8-128 characters, must contain uppercase and digit

**Response:** `UserResponse` schema with user details

**Error Handling:**
- Returns 400 if email already registered
- Returns 400 if password validation fails

**Usage Example:**
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123"}'
# Response: {"id": "uuid", "email": "user@example.com", "created_at": "2024-01-01T..."}
```

---

##### `POST /auth/login`

**Purpose:** Authenticate user and return JWT token.

**Request Body:** `UserCreate` schema
- `email` (EmailStr): User email
- `password` (str): User password

**Response:** `TokenResponse` schema
- `access_token` (str): JWT token for authorization
- `token_type` (str): "bearer"

**Error Handling:** Returns 401 if email/password invalid

**Usage Example:**
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123"}'
# Response: {"access_token": "eyJhbGc...", "token_type": "bearer"}
```

---

##### `GET /user/profile`

**Purpose:** Retrieve current user's profile (protected).

**Headers:** `Authorization: Bearer {token}` (required)

**Response:** `UserResponse` schema

**Error Handling:** Returns 404 if user not found

**Usage Example:**
```bash
curl -H "Authorization: Bearer eyJhbGc..." http://localhost:8000/user/profile
```

---

##### `GET /cases`

**Purpose:** List all cases for current user (protected).

**Headers:** `Authorization: Bearer {token}` (required)

**Response:** List of case objects with `id`, `name`, `status`, `created_at`, `updated_at`

**Usage Example:**
```bash
curl -H "Authorization: Bearer eyJhbGc..." http://localhost:8000/cases
# Response: [{"id": "uuid", "name": "Case 1", "status": "ready", ...}]
```

---

##### `POST /cases`

**Purpose:** Upload a legal document and create case (protected).

**Headers:** `Authorization: Bearer {token}` (required)

**Form Data:**
- `name` (str): Case name (1-255 characters)
- `file` (UploadFile): PDF, DOCX, or TXT document

**Response:** Case object with upload details including `task_id` for background processing

**Error Handling:**
- Returns 400 for unsupported file types
- Returns 400 if file format validation fails
- Returns 500 if blob storage fails

**Validation:**
- Filename validated for path traversal attempts
- File format validated by magic bytes
- Case name validated for length

**Usage Example:**
```bash
curl -X POST http://localhost:8000/cases \
  -H "Authorization: Bearer eyJhbGc..." \
  -F "name=Smith v. Jones" \
  -F "file=@document.pdf"
# Response: {"id": "uuid", "name": "Smith v. Jones", "status": "processing", ...}
```

---

##### `POST /cases/{case_id}/ask`

**Purpose:** Query case document and get RAG-generated answer (protected).

**Parameters:**
- `case_id` (str): UUID of case

**Headers:** `Authorization: Bearer {token}` (required)

**Request Body:** JSON with `question` field (string, 3-5000 characters)

**Response:** Dictionary with:
- `answer` (str): Generated answer with citations
- `sources` (list): Retrieved chunks with metadata
- `citations` (list): Grounded citations with supporting excerpts
- `confidence` (dict): Confidence score and factors
- `tokens_used` (int): Total tokens consumed
- `error` (str or None): Error message if failed

**Error Handling:**
- Returns 400 if question validation fails
- Returns 400 if case not found
- Returns 403 if unauthorized access
- Returns 400 if case still processing
- Returns 500 if RAG query fails

**Usage Example:**
```bash
curl -X POST http://localhost:8000/cases/550e8400-e29b-41d4-a716-446655440000/ask \
  -H "Authorization: Bearer eyJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the court decision?"}'
```

---

##### `GET /cases/{case_id}/status`

**Purpose:** Get processing status of a case (protected).

**Parameters:**
- `case_id` (str): UUID of case

**Headers:** `Authorization: Bearer {token}` (required)

**Response:** Dictionary with `id`, `name`, `status` (processing/ready/error), `created_at`

**Usage Example:**
```bash
curl -H "Authorization: Bearer eyJhbGc..." \
  http://localhost:8000/cases/550e8400-e29b-41d4-a716-446655440000/status
```

---

### backend/config.py

Configuration management using Pydantic Settings for environment variables.

#### Classes

##### `Settings`

**Purpose:** Pydantic model for application configuration management.

**Attributes:**
- `database_url` (str): PostgreSQL connection string
- `openai_api_key` (str): OpenAI API key for embeddings and LLM
- `qdrant_url` (str): Qdrant vector store URL (default: "http://localhost:6333")
- `redis_url` (str): Redis connection URL (default: "redis://localhost:6379/0")
- `celery_broker_url` (str): Celery broker URL (default: redis URL)
- `celery_result_backend` (str): Celery result backend (default: redis)
- `azure_storage_connection_string` (str): Azure Blob Storage connection string
- `secret_key` (str): JWT signing key
- `algorithm` (str): JWT algorithm (default: "HS256")
- `access_token_expire_minutes` (int): Token expiry (default: 1440 = 24 hours)
- `allowed_origins` (str): Comma-separated CORS origins
- `debug` (bool): Debug mode (default: False)
- `cache_enabled` (bool): Query caching enabled (default: True)
- `cache_ttl_seconds` (int): Cache TTL (default: 86400 = 24 hours)

---

##### `Settings.get_allowed_origins_list() -> List[str]`

**Purpose:** Parse comma-separated CORS origins string into list.

**Return Value:** `list[str]` - Individual origin URLs

**Usage Example:**
```python
settings = get_settings()
origins = settings.get_allowed_origins_list()
# Returns ['http://localhost:3000', 'https://example.com'] if ALLOWED_ORIGINS="http://localhost:3000,https://example.com"
```

---

#### Functions

##### `get_settings() -> Settings`

**Purpose:** Get or create cached Settings instance (singleton pattern).

**Return Value:** `Settings` instance with environment variables loaded

**Usage Example:**
```python
settings = get_settings()
api_key = settings.openai_api_key
```

---

### backend/database.py

Database connection and session management for SQLAlchemy.

#### Functions

##### `init_db(database_url: str) -> Tuple[Engine, sessionmaker]`

**Purpose:** Initialize database engine and session factory.

**Parameters:**
- `database_url` (str): Database connection string

**Return Value:** Tuple of:
- `Engine`: SQLAlchemy engine instance
- `sessionmaker`: Session factory for creating sessions

**Configuration:**
- `echo=False`: No SQL logging
- `pool_pre_ping=True`: Test connections before use

**Usage Example:**
```python
engine, SessionLocal = init_db("postgresql://user:password@localhost/lexintel")
```

---

##### `get_engine() -> Engine`

**Purpose:** Get database engine (lazy-initialized at runtime).

**Return Value:** `Engine` instance

**Usage Example:**
```python
engine = get_engine()
```

---

##### `get_session_factory() -> sessionmaker`

**Purpose:** Get session factory (lazy-initialized at runtime).

**Return Value:** `sessionmaker` - Session factory instance

**Usage Example:**
```python
SessionLocal = get_session_factory()
session = SessionLocal()
```

---

##### `get_db() -> Generator[Session, None, None]`

**Purpose:** Dependency for getting database sessions in endpoints.

**Return Value:** Generator yielding `Session` instance

**Error Handling:** Automatically closes session in finally block

**Usage Example:**
```python
@app.get("/data")
def get_data(db: Session = Depends(get_db)):
    return db.query(User).all()
```

---

### backend/models.py

SQLAlchemy ORM models for database tables.

#### Enums

##### `CaseStatus`

String enum for case processing states:
- `PROCESSING = "processing"` - Document being processed
- `READY = "ready"` - Document ready for queries
- `ERROR = "error"` - Processing failed

---

##### `FileType`

String enum for supported document types:
- `PDF = "pdf"` - PDF documents
- `DOCX = "docx"` - Word documents
- `TXT = "txt"` - Text files

---

#### Models

##### `User`

**Purpose:** User account model.

**Columns:**
- `id` (UUID, PK): Unique user identifier
- `email` (String[255], unique): User email address
- `password_hash` (String[255]): Bcrypt hashed password
- `is_deleted` (Boolean): Soft delete flag
- `created_at` (DateTime): Account creation timestamp
- `updated_at` (DateTime): Last update timestamp

**Relationships:**
- `cases` → Case (one-to-many)

**Indexes:**
- email (unique)
- is_deleted

---

##### `Case`

**Purpose:** Legal case document model.

**Columns:**
- `id` (UUID, PK): Unique case identifier
- `user_id` (UUID, FK): Case owner
- `name` (String[255]): Case name
- `blob_storage_path` (String[500]): Azure Blob Storage path
- `file_type` (String[10]): Document type (pdf/docx/txt)
- `status` (String[50]): Processing status (processing/ready/error)
- `is_deleted` (Boolean): Soft delete flag
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

**Relationships:**
- `user` → User (many-to-one)
- `chunks` → Chunk (one-to-many)
- `queries` → Query (one-to-many)

**Indexes:**
- user_id, status (composite)
- is_deleted
- created_at

---

##### `Chunk`

**Purpose:** Document chunk model for RAG retrieval.

**Columns:**
- `id` (UUID, PK): Unique chunk identifier
- `case_id` (UUID, FK): Parent case
- `page_num` (String[50]): Page/section location
- `section_name` (String[255]): Section label
- `content` (Text): Full chunk text content
- `embedding_hash` (String[255]): SHA256 hash for deduplication
- `chunk_sequence` (Integer): Order within case
- `created_at` (DateTime): Creation timestamp

**Relationships:**
- `case` → Case (many-to-one)

**Indexes:**
- case_id, chunk_sequence (composite)

---

##### `Query`

**Purpose:** User query and answer history model.

**Columns:**
- `id` (UUID, PK): Unique query identifier
- `case_id` (UUID, FK): Queried case
- `user_id` (UUID, FK): Query author
- `question` (Text): User question
- `answer` (Text): Generated answer
- `citations` (JSON): List of citation dictionaries
- `created_at` (DateTime): Query timestamp

**Relationships:**
- `case` → Case (many-to-one)

**Indexes:**
- case_id, created_at (composite)

---

##### `ProcessingJob`

**Purpose:** Background job tracking for document processing.

**Columns:**
- `id` (UUID, PK): Unique job identifier
- `case_id` (UUID, FK): Associated case
- `status` (String[50]): Job status (pending/processing/completed/failed)
- `error_message` (String[500]): Error description if failed
- `attempts` (Integer): Number of attempts (default: 0)
- `max_attempts` (Integer): Max retry attempts (default: 3)
- `created_at` (DateTime): Job creation timestamp
- `started_at` (DateTime, optional): Job start timestamp
- `completed_at` (DateTime, optional): Job completion timestamp
- `next_retry_at` (DateTime, optional): Scheduled retry time

---

### backend/schemas.py

Pydantic v2 schemas for request/response validation and serialization.

#### User Schemas

##### `UserCreate`

**Purpose:** User registration request validation.

**Fields:**
- `email` (EmailStr): Valid email address (required)
- `password` (str): 8-128 characters, must contain uppercase letter and digit (required)

**Validators:**
- `password_strength`: Ensures password has uppercase and digit

---

##### `UserResponse`

**Purpose:** User data response (safe output without password).

**Fields:**
- `id` (UUID): User identifier
- `email` (str): Email address
- `created_at` (datetime): Account creation time

---

##### `TokenResponse`

**Purpose:** Login response with authentication token.

**Fields:**
- `access_token` (str): JWT token for Authorization header
- `token_type` (str): Always "bearer"

---

#### Case Schemas

##### `CaseCreate`

**Purpose:** Case creation request.

**Fields:**
- `name` (str): 1-255 characters (required)

---

##### `CaseResponse`

**Purpose:** Case data response.

**Fields:**
- `id` (UUID): Case identifier
- `name` (str): Case name
- `status` (str): processing/ready/error
- `blob_storage_path` (str): Storage location
- `created_at` (datetime): Creation timestamp
- `updated_at` (datetime): Last update timestamp

---

#### Chunk Schemas

##### `ChunkResponse`

**Purpose:** Document chunk response (for retrieval display).

**Fields:**
- `id` (UUID): Chunk identifier
- `case_id` (UUID): Parent case
- `page_num` (str, optional): Location info
- `section_name` (str, optional): Section label
- `content` (str): Chunk text
- `embedding_hash` (str, optional): Deduplication hash
- `chunk_sequence` (int, optional): Order index
- `created_at` (datetime): Creation timestamp

---

#### Query Schemas

##### `QueryCreate`

**Purpose:** Query request validation.

**Fields:**
- `question` (str): 1-1000 characters (required)

---

##### `CitationData`

**Purpose:** Citation metadata in responses.

**Fields:**
- `page` (str): Location reference (required)
- `section` (str, optional): Section name
- `content_snippet` (str): Quote from source (required)
- `score` (float, optional): Relevance score (0.0-1.0)

---

##### `QueryResponse`

**Purpose:** Query result response.

**Fields:**
- `id` (UUID): Query identifier
- `question` (str): Original question
- `answer` (str): Generated answer
- `citations` (list[CitationData]): Citation list (default: empty)
- `created_at` (datetime): Query timestamp

---

### backend/auth.py

JWT authentication and password hashing utilities.

#### Functions

##### `hash_password(password: str) -> str`

**Purpose:** Hash plaintext password using bcrypt.

**Parameters:**
- `password` (str): Plaintext password to hash

**Return Value:** `str` - Bcrypt hash (can be stored in DB)

**Usage Example:**
```python
hashed = hash_password("MySecurePass123")
# Returns: '$2b$12$...'
```

---

##### `verify_password(plain_password: str, hashed_password: str) -> bool`

**Purpose:** Verify plaintext password against stored hash.

**Parameters:**
- `plain_password` (str): Plaintext password from login
- `hashed_password` (str): Hash from database

**Return Value:** `bool` - True if password matches, False otherwise

**Usage Example:**
```python
is_valid = verify_password("MySecurePass123", stored_hash)
```

---

##### `create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str`

**Purpose:** Create JWT access token for user.

**Parameters:**
- `data` (dict): Token claims (typically `{"sub": user_id}`)
- `expires_delta` (timedelta, optional): Custom expiry time

**Return Value:** `str` - Encoded JWT token

**Default Expiry:** 1440 minutes (24 hours) from settings

**Usage Example:**
```python
token = create_access_token({"sub": str(user_id)})
# Or with custom expiry:
token = create_access_token({"sub": str(user_id)}, timedelta(hours=2))
```

---

##### `decode_token(token: str) -> Optional[str]`

**Purpose:** Decode JWT token and extract user_id.

**Parameters:**
- `token` (str): JWT token from Authorization header

**Return Value:** `str` (user_id from 'sub' claim) or `None` if invalid/expired

**Error Handling:** Returns None on JWTError (logs warning, doesn't raise)

**Usage Example:**
```python
user_id = decode_token(token_string)
if user_id:
    # Token valid
else:
    # Token invalid or expired
```

---

### backend/validators.py

Input validation functions with detailed error messages.

#### Functions

##### `validate_filename(filename: str) -> str`

**Purpose:** Validate uploaded filename for security and constraints.

**Parameters:**
- `filename` (str): Filename to validate

**Return Value:** `str` - Validated filename

**Validation Rules:**
- Not empty
- No path traversal attempts (.., ~/, /)
- Max 255 characters
- Only PDF, DOCX, TXT extensions allowed

**Error Handling:** Raises `HTTPException(400)` with descriptive message

**Usage Example:**
```python
try:
    validate_filename("document.pdf")  # OK
    validate_filename("../../../etc/passwd")  # Raises 400
except HTTPException as e:
    print(e.detail)
```

---

##### `validate_question(question: str) -> str`

**Purpose:** Validate user query for length constraints.

**Parameters:**
- `question` (str): User question

**Return Value:** `str` - Trimmed and validated question

**Validation Rules:**
- Not empty
- Min 3 characters
- Max 5000 characters

**Error Handling:** Raises `HTTPException(400)` with descriptive message

**Usage Example:**
```python
validated = validate_question("What was the verdict?")  # OK (11 chars)
validate_question("ab")  # Raises 400 (too short)
```

---

##### `validate_file_type(content_type: str, filename: str) -> str`

**Purpose:** Detect and validate file type based on filename extension.

**Parameters:**
- `content_type` (str): MIME type (not used for validation)
- `filename` (str): Original filename

**Return Value:** `str` - File type ('pdf', 'docx', or 'txt')

**Error Handling:** Raises `ValueError` if unsupported file type

**Usage Example:**
```python
file_type = validate_file_type("application/pdf", "document.pdf")
# Returns: "pdf"

file_type = validate_file_type("", "contract.docx")
# Returns: "docx"
```

---

##### `validate_case_name(name: str) -> str`

**Purpose:** Validate case name for constraints.

**Parameters:**
- `name` (str): Case name

**Return Value:** `str` - Trimmed and validated case name

**Validation Rules:**
- Not empty
- Max 255 characters

**Error Handling:** Raises `HTTPException(400)` if validation fails

**Usage Example:**
```python
validate_case_name("Smith v. Jones")  # OK
validate_case_name("")  # Raises 400 (empty)
```

---

### backend/exceptions.py

Custom exception hierarchy for LexIntel error handling.

#### Exception Classes

##### `LexIntelException`

**Purpose:** Base exception for all LexIntel errors.

**Constructor:**
```python
def __init__(self, message: str, detail: str = None)
```

**Parameters:**
- `message` (str): Human-readable error message
- `detail` (str, optional): Additional context

**Attributes:**
- `message`: Error message
- `detail`: Additional details

---

##### Storage Exceptions

- `StorageException` - Base for storage errors
- `BlobUploadException` - Upload to blob storage failed
- `BlobDownloadException` - Download from blob storage failed
- `BlobDeleteException` - Blob deletion failed

---

##### RAG Exceptions

- `RAGException` - Base for RAG pipeline errors
- `EmbeddingException` - Text embedding generation failed
- `VectorStoreException` - Vector database operation failed
- `QueryProcessingException` - Query processing or LLM answer generation failed

---

##### Document Exceptions

- `DocumentProcessingException` - Base for document processing errors
- `ChunkingException` - Document chunking failed
- `InvalidPDFException` - PDF validation failed

---

##### Validation Exceptions

- `ValidationException` - Input validation failed

---

### backend/services/storage.py

Azure Blob Storage operations for document persistence.

#### Constants

```python
PDF_MAGIC_BYTES = b"%PDF"              # PDF file signature
DOCX_MAGIC_BYTES = b"PK\x03\x04"       # ZIP signature (DOCX format)
```

---

#### Functions

##### `validate_file_format(file_content: bytes, file_type: str) -> bool`

**Purpose:** Validate file content matches declared type using magic bytes.

**Parameters:**
- `file_content` (bytes): Raw file bytes
- `file_type` (str): Declared type ('pdf', 'docx', 'txt')

**Return Value:** `bool` - True if format valid, False otherwise

**Validation Logic:**
- PDF: Starts with `%PDF`
- DOCX: Starts with ZIP signature (0x504B0304)
- TXT: Valid UTF-8 decodable

**Usage Example:**
```python
with open("document.pdf", "rb") as f:
    is_valid = validate_file_format(f.read(), "pdf")
```

---

##### `get_blob_client() -> BlobServiceClient`

**Purpose:** Get Azure Blob Storage client from connection string.

**Return Value:** `BlobServiceClient` instance

**Error Handling:** Raises if Azure connection string not configured

**Usage Example:**
```python
client = get_blob_client()
```

---

##### `async upload_document_to_blob(file_content: bytes, case_id: str, filename: str) -> str`

**Purpose:** Upload document to Azure Blob Storage.

**Parameters:**
- `file_content` (bytes): Raw file bytes
- `case_id` (str): Case UUID (organizes blobs)
- `filename` (str): Original filename

**Return Value:** `str` - Blob path (e.g., "case-uuid/filename.pdf")

**Error Handling:** Raises `BlobUploadException` if upload fails

**Behavior:**
- Creates "cases" container if doesn't exist
- Stores blob in `{case_id}/{filename}` structure

**Usage Example:**
```python
blob_path = await upload_document_to_blob(file_bytes, str(case_id), "contract.pdf")
# Returns: "550e8400-e29b-41d4-a716-446655440000/contract.pdf"
```

---

##### `download_document_from_blob(blob_path: str) -> bytes`

**Purpose:** Download document from Azure Blob Storage.

**Parameters:**
- `blob_path` (str): Blob path from case record

**Return Value:** `bytes` - Raw file content

**Error Handling:** Raises `BlobDownloadException` if download fails

**Usage Example:**
```python
file_bytes = download_document_from_blob("550e8400-e29b-41d4-a716-446655440000/contract.pdf")
```

---

##### `delete_blob(blob_path: str) -> bool`

**Purpose:** Delete blob from Azure Blob Storage.

**Parameters:**
- `blob_path` (str): Blob path to delete

**Return Value:** `bool` - True if successful

**Error Handling:** Raises `BlobDeleteException` if deletion fails

**Usage Example:**
```python
success = delete_blob("550e8400-e29b-41d4-a716-446655440000/contract.pdf")
```

---

### backend/services/chunking.py

Document chunking service for semantic text splitting.

#### Constants

```python
CHUNK_SIZE = 1500              # Characters per chunk (~200-250 words)
CHUNK_OVERLAP = 300            # Overlap between chunks for context
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Splitting preferences
```

---

#### Functions

##### `chunk_pdf(pdf_path: str) -> List[Dict[str, str]]`

**Purpose:** Chunk PDF file into semantic pieces (deprecated, use chunk_document_from_blob).

**Parameters:**
- `pdf_path` (str): Local path to PDF file

**Return Value:** List of chunk dicts with keys:
- `content` (str): Chunk text
- `page_num` (str): 1-indexed page number
- `section_name` (str): "Chunk N"

**Error Handling:**
- Raises `FileNotFoundError` if PDF doesn't exist
- Raises if PDF parsing fails

---

##### `chunk_document_from_blob(blob_content: bytes, file_type: str = "pdf") -> List[Dict[str, str]]`

**Purpose:** Chunk document (PDF, DOCX, or TXT) from blob storage bytes.

**Parameters:**
- `blob_content` (bytes): Raw document bytes
- `file_type` (str): 'pdf', 'docx', or 'txt' (default: 'pdf')

**Return Value:** List of chunk dicts with:
- `content` (str): Chunk text
- `page_num` (str): Location info (page number, "para X", "line X-Y")
- `section_name` (str): "Chunk N"

**Error Handling:** Raises `ValueError` if content empty or chunking fails

**Usage Example:**
```python
chunks = chunk_document_from_blob(file_bytes, "pdf")
# Returns: [
#   {"content": "...", "page_num": "1", "section_name": "Chunk 1"},
#   {"content": "...", "page_num": "2", "section_name": "Chunk 2"}
# ]
```

---

##### `estimate_tokens(chunk_content: str) -> int`

**Purpose:** Estimate token count using OpenAI approximation (1 token ≈ 4 chars).

**Parameters:**
- `chunk_content` (str): Text to estimate

**Return Value:** `int` - Approximate token count

**Usage Example:**
```python
tokens = estimate_tokens(chunk_text)  # Returns ~375 for 1500-char chunk
```

---

### backend/services/text_extraction.py

Multi-format text extraction supporting PDF, DOCX, and TXT.

#### Functions

##### `extract_pdf_text(file_bytes: bytes) -> List[Dict[str, str]]`

**Purpose:** Extract text from PDF using PyMuPDF (fitz).

**Parameters:**
- `file_bytes` (bytes): Raw PDF bytes

**Return Value:** List of section dicts with:
- `content` (str): Page text
- `location` (str): 1-indexed page number
- `location_type` (str): "page"

**Error Handling:** Raises `ValueError` if PDF invalid or empty

**Usage Example:**
```python
sections = extract_pdf_text(pdf_bytes)
# Returns: [
#   {"content": "Page 1 text...", "location": "1", "location_type": "page"},
#   {"content": "Page 2 text...", "location": "2", "location_type": "page"}
# ]
```

---

##### `extract_docx_text(file_bytes: bytes) -> List[Dict[str, str]]`

**Purpose:** Extract text from DOCX using python-docx.

**Parameters:**
- `file_bytes` (bytes): Raw DOCX bytes

**Return Value:** List of section dicts with:
- `content` (str): Paragraph text
- `location` (str): "para X" format
- `location_type` (str): "paragraph"

**Error Handling:** Raises `ValueError` if DOCX invalid

**Usage Example:**
```python
sections = extract_docx_text(docx_bytes)
# Returns: [
#   {"content": "Para 1...", "location": "para 1", "location_type": "paragraph"},
#   {"content": "Para 2...", "location": "para 2", "location_type": "paragraph"}
# ]
```

---

##### `extract_txt_text(file_bytes: bytes, lines_per_section: int = 50) -> List[Dict[str, str]]`

**Purpose:** Extract text from plain TXT files.

**Parameters:**
- `file_bytes` (bytes): Raw TXT bytes
- `lines_per_section` (int): Lines per section (default: 50)

**Return Value:** List of section dicts with:
- `content` (str): Section text
- `location` (str): "line X-Y" format
- `location_type` (str): "line_range"

**Error Handling:** Raises `ValueError` if not valid UTF-8

**Usage Example:**
```python
sections = extract_txt_text(txt_bytes)
# Returns: [
#   {"content": "Lines 1-50...", "location": "line 1-50", "location_type": "line_range"},
#   {"content": "Lines 51-100...", "location": "line 51-100", "location_type": "line_range"}
# ]
```

---

##### `extract_text(file_bytes: bytes, file_type: str) -> List[Dict[str, str]]`

**Purpose:** Router function to extract text from any supported format.

**Parameters:**
- `file_bytes` (bytes): Raw file content
- `file_type` (str): 'pdf', 'docx', or 'txt'

**Return Value:** List of section dicts (format depends on file type)

**Error Handling:** Raises `ValueError` for unsupported types

**Usage Example:**
```python
sections = extract_text(file_bytes, "pdf")
```

---

### backend/services/embeddings.py

OpenAI embeddings service for semantic vector generation.

#### Constants

```python
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # Output vector dimension
```

---

#### Functions

##### `get_embeddings_client() -> OpenAIEmbeddings`

**Purpose:** Get or create OpenAI embeddings client (cached singleton).

**Return Value:** `OpenAIEmbeddings` instance configured with text-embedding-3-large

**Error Handling:** Raises `ValueError` if OPENAI_API_KEY not set

**Usage Example:**
```python
embeddings = get_embeddings_client()
```

---

##### `embed_text(text: str) -> List[float]`

**Purpose:** Embed single text into vector space.

**Parameters:**
- `text` (str): Text to embed

**Return Value:** `list[float]` - 3072-dimensional embedding vector

**Error Handling:**
- Raises `ValueError` if text empty
- Raises `EmbeddingException` if API call fails

**Usage Example:**
```python
query = "What was the court decision?"
embedding = embed_text(query)  # Returns 3072-dim vector
```

---

##### `embed_chunks(chunks: List[str]) -> List[List[float]]`

**Purpose:** Embed multiple chunks in batch for efficiency.

**Parameters:**
- `chunks` (list[str]): Text chunks to embed

**Return Value:** `list[list[float]]` - List of 3072-dimensional vectors

**Error Handling:**
- Raises `ValueError` if list empty or contains empty strings
- Raises `EmbeddingException` if API call fails
- Raises if embedding count doesn't match input count

**Usage Example:**
```python
chunk_texts = [chunk["content"] for chunk in chunks]
embeddings = embed_chunks(chunk_texts)
# Returns: [[0.1, -0.2, ...], [0.3, 0.1, ...], ...]
```

---

##### `estimate_embedding_cost(text_length: int) -> float`

**Purpose:** Estimate cost of embedding text with text-embedding-3-large.

**Parameters:**
- `text_length` (int): Total character count to embed

**Return Value:** `float` - Estimated cost in USD

**Calculation:** Tokens ≈ text_length / 4, Cost = (tokens / 1,000,000) * $0.02

**Usage Example:**
```python
total_chars = sum(len(c["content"]) for c in chunks)
cost = estimate_embedding_cost(total_chars)  # Returns ~$0.000015 for 100KB
```

---

### backend/services/vector_store.py

Qdrant vector database operations for semantic search.

#### Constants

```python
VECTOR_SIZE = 3072              # Matches text-embedding-3-large
DISTANCE_METRIC = "Cosine"      # Cosine similarity
```

---

#### Functions

##### `get_qdrant_client() -> QdrantClient`

**Purpose:** Get or create Qdrant client (cached singleton).

**Return Value:** `QdrantClient` connected to Qdrant server

**Error Handling:** Raises `ValueError` if QDRANT_URL not configured

**Timeout:** 30 seconds per request

**Usage Example:**
```python
client = get_qdrant_client()
```

---

##### `_get_collection_name(case_id: str) -> str`

**Purpose:** Generate collection name from case ID (internal helper).

**Parameters:**
- `case_id` (str): Case UUID

**Return Value:** `str` - Collection name (format: "case_{case_id}")

---

##### `_generate_point_id(chunk_id: str, case_id: str) -> int`

**Purpose:** Generate deterministic point ID for idempotency (internal helper).

**Parameters:**
- `chunk_id` (str): Chunk UUID
- `case_id` (str): Case UUID

**Return Value:** `int` - Positive integer point ID

**Method:** MD5 hash of "case_id:chunk_id" → first 8 bytes as unsigned int

---

##### `create_collection(case_id: str) -> bool`

**Purpose:** Create vector collection for a case.

**Parameters:**
- `case_id` (str): Case UUID

**Return Value:** `bool` - True if successful

**Configuration:**
- Vector size: 3072 dimensions
- Distance metric: Cosine similarity
- Recreates collection if exists

**Error Handling:** Raises `VectorStoreException` if creation fails

**Usage Example:**
```python
create_collection(str(case_id))
```

---

##### `upsert_vectors(case_id: str, chunks: List[Dict], embeddings: List[List[float]]) -> int`

**Purpose:** Insert or update vectors with metadata into collection.

**Parameters:**
- `case_id` (str): Case UUID
- `chunks` (list[dict]): Chunk dicts with `id`, `content`, `page_num`, `section_name`
- `embeddings` (list[list[float]]): 3072-dim vectors matching chunks

**Return Value:** `int` - Number of vectors upserted

**Error Handling:**
- Raises `ValueError` if inputs empty or mismatched
- Raises `VectorStoreException` if upsert fails

**Metadata Stored:** chunk_id, page_num, section_name, content_preview (200 chars)

**Idempotency:** Same chunk always gets same point ID (deterministic hashing)

**Usage Example:**
```python
count = upsert_vectors(str(case_id), chunks_data, embeddings_list)
# Returns: 42 (42 vectors upserted)
```

---

##### `search_vectors(case_id: str, query_embedding: List[float], limit: int = 5) -> List[Dict]`

**Purpose:** Semantic search for similar chunks.

**Parameters:**
- `case_id` (str): Case UUID
- `query_embedding` (list[float]): 3072-dim query vector
- `limit` (int): Max results to return (default: 5)

**Return Value:** List of result dicts with:
- `score` (float): Cosine similarity (0-1)
- `chunk_id` (str): Chunk UUID
- `page_num` (str): Location
- `content_preview` (str): First 200 chars
- `section_name` (str): Section label

**Error Handling:**
- Raises `ValueError` if embedding dimension wrong
- Raises `VectorStoreException` if search fails

**Implementation:** Uses REST API directly for Qdrant 1.7.0 compatibility

**Usage Example:**
```python
results = search_vectors(str(case_id), query_vector, limit=5)
# Returns: [
#   {"score": 0.87, "chunk_id": "uuid1", "page_num": "1", ...},
#   {"score": 0.75, "chunk_id": "uuid2", "page_num": "2", ...}
# ]
```

---

##### `delete_collection(case_id: str) -> bool`

**Purpose:** Delete collection and all vectors for a case.

**Parameters:**
- `case_id` (str): Case UUID

**Return Value:** `bool` - True if successful

**Error Handling:** Raises `VectorStoreException` if deletion fails

---

### backend/services/rag_engine.py

Complete RAG pipeline orchestration for legal document analysis.

#### Constants

```python
CONTEXT_TOKEN_BUDGET = 12_800  # Max tokens for context window
MIN_QUERY_LENGTH = 3           # Minimum query characters
MIN_CONFIDENCE_SCORE = 0.6     # Minimum semantic similarity threshold
RETRIEVAL_TOP_K = 10           # Initial retrieval count
FINAL_CHUNK_COUNT = 4          # Final chunks for context
```

---

#### Token Counting

##### `count_tokens_gpt4o(text: str) -> int`

**Purpose:** Count tokens in text using tiktoken for GPT-4o model.

**Parameters:**
- `text` (str): Text to count tokens for

**Return Value:** `int` - Token count

**Error Handling:** Raises `ValueError` if text empty or encoding fails

**Usage Example:**
```python
tokens = count_tokens_gpt4o("What was the verdict?")  # Returns ~6
```

---

#### Context Formatting

##### `format_legal_context(chunks: List[Dict], case_name: str) -> str`

**Purpose:** Format retrieved chunks into structured legal context with metadata.

**Parameters:**
- `chunks` (list[dict]): Chunks with content, page_num, section_name, score
- `case_name` (str): Case name for header

**Return Value:** `str` - Formatted context string

**Format:**
```
Case: Smith v. Jones
============================================================

--- EXCERPT 1 (Page 1, Section: Judgment, Score: 0.87) ---
...content...

--- EXCERPT 2 (Page 2, Section: Facts, Score: 0.75) ---
...content...
```

**Error Handling:** Raises `ValueError` if chunks list empty

**Usage Example:**
```python
context = format_legal_context(retrieved_chunks, "Smith v. Jones")
```

---

#### Embedding and Retrieval

##### `embed_query(query: str) -> List[float]`

**Purpose:** Embed user query into vector space.

**Parameters:**
- `query` (str): User question

**Return Value:** `list[float]` - 3072-dim embedding

**Error Handling:** Raises if embedding fails

---

##### `retrieve_chunks(case_id: str, query_embedding: List[float], top_k: int = RETRIEVAL_TOP_K) -> List[Dict]`

**Purpose:** Retrieve similar chunks from vector store.

**Parameters:**
- `case_id` (str): Case UUID
- `query_embedding` (list[float]): Query embedding
- `top_k` (int): Number of results (default: 10)

**Return Value:** List of chunk dicts with score, content, page_num, etc.

**Error Handling:** Raises `VectorStoreException` if search fails

---

#### Reranking

##### `rerank_chunks(query: str, chunks: List[Dict], top_k: int = FINAL_CHUNK_COUNT) -> List[Dict]`

**Purpose:** Rerank retrieved chunks using cross-encoder for better relevance.

**Parameters:**
- `query` (str): User query
- `chunks` (list[dict]): Chunks from vector search
- `top_k` (int): Top results after reranking (default: 4)

**Return Value:** Reranked chunks sorted by relevance

**Algorithm:**
- Uses cross-encoder/qnli-distilroberta-base model
- Combines vector similarity (40%) + cross-encoder score (60%)
- Gracefully falls back to vector ranking if reranker unavailable

**Scoring:** `combined_score = (vector_score * 0.4) + (rerank_score * 0.6)`

---

#### Citation Extraction

##### `extract_citations(answer: str, chunks: List[Dict]) -> Tuple[str, List[Dict], bool]`

**Purpose:** Extract citations from answer and validate against retrieved chunks.

**Parameters:**
- `answer` (str): LLM-generated answer
- `chunks` (list[dict]): Retrieved chunks with page_num

**Return Value:** Tuple of:
- `str`: Cleaned answer with hallucinated citations removed
- `list[dict]`: Valid citations with location, type, relevance_score
- `bool`: Whether hallucinations were detected

**Citation Formats Supported:**
- `[Page X]` for PDFs
- `[Paragraph X]` for Word documents
- `[Lines X-Y]` for text files

**Hallucination Detection:** Identifies citations not supported by retrieved chunks

**Usage Example:**
```python
cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)
```

---

##### `ground_citations_in_source(citations: List[Dict], chunks: List[Dict]) -> Tuple[List[Dict], List[Dict], bool]`

**Purpose:** Validate citations have supporting text and extract excerpts.

**Parameters:**
- `citations` (list[dict]): Citations from extract_citations
- `chunks` (list[dict]): Retrieved chunks

**Return Value:** Tuple of:
- Grounded citations with supporting_excerpt and is_grounded flag
- Unsupported claims that couldn't be grounded
- Bool indicating if any claims unsupported

---

#### Confidence Scoring

##### `calculate_answer_confidence(answer: str, citations: List[Dict], chunks: List[Dict], has_hallucinations: bool) -> float`

**Purpose:** Calculate confidence score for answer (0.0-1.0).

**Parameters:**
- `answer` (str): Generated answer
- `citations` (list[dict]): Grounded citations
- `chunks` (list[dict]): Retrieved chunks
- `has_hallucinations` (bool): Whether hallucinations detected

**Return Value:** `float` - Confidence score (0.0-1.0)

**Factors:**
- Citation coverage: % of sentences with citations (30% weight)
- Average relevance: Mean similarity score (50% weight)
- Citation quantity: More citations → higher confidence (20% bonus)
- Hallucination penalty: -0.3 if hallucinations present

---

##### `classify_confidence_level(confidence_score: float) -> str`

**Purpose:** Convert confidence score to categorical level.

**Parameters:**
- `confidence_score` (float): Score from 0.0-1.0

**Return Value:** `str` - "high" (≥0.75), "medium" (≥0.6), "low" (≥0.4), or "none"

---

##### `explain_confidence_score(answer: str, citations: List[Dict], has_hallucinations: bool, confidence_score: float) -> Dict`

**Purpose:** Generate detailed confidence explanation with factor breakdown.

**Return Value:** Dict with:
- `overall_score`: Confidence score
- `rating`: "high", "medium", "low"
- `factors`: Dict with citation_coverage, source_relevance, hallucination_risk, citation_quantity
- `summary`: Human-readable explanation

---

#### Answer Generation

##### `async generate_answer(query: str, context: str, temperature: float = 0.2) -> Tuple[str, int]`

**Purpose:** Generate answer using OpenAI GPT-4o with legal context.

**Parameters:**
- `query` (str): User question
- `context` (str): Formatted context from retrieval
- `temperature` (float): LLM temperature (default: 0.2 for precision)

**Return Value:** Tuple of:
- `str`: Generated answer
- `int`: Tokens consumed by API call

**System Prompt:** Legal expert instructions for factual, cited responses

**Error Handling:** Raises `QueryProcessingException` if API fails

---

#### Main Query Function

##### `async query_case(case_id: str, query: str, db: Session, top_k: int = FINAL_CHUNK_COUNT, temperature: float = 0.2) -> Dict`

**Purpose:** Complete RAG pipeline orchestration function.

**Parameters:**
- `case_id` (str): Case UUID
- `query` (str): User question
- `db` (Session): Database session
- `top_k` (int): Chunks in final context (default: 4)
- `temperature` (float): LLM temperature (default: 0.2)

**Return Value:** Dict with:
- `answer` (str): Generated answer or None if error
- `sources` (list): Retrieved chunks with full content
- `citations` (list): Grounded citations with supporting excerpts
- `confidence` (dict): Confidence level, score, and factors
- `confidence_explanation` (dict): Detailed confidence explanation
- `source_document` (dict): Document summary
- `tokens_used` (int): Total tokens consumed
- `error` (str or None): Error message if processing failed
- `model` (str): "gpt-4o"

**Pipeline Steps:**
1. Validate query (min 3 chars)
2. Embed query
3. Retrieve similar chunks (top 10)
4. Filter by confidence threshold (≥0.6 similarity)
5. Rerank chunks for better relevance
6. Format context with token budgeting
7. Generate answer with GPT-4o
8. Extract and validate citations
9. Ground citations in source text
10. Calculate confidence score
11. Return structured response

**Error Handling:** Returns error_response dict with error message instead of raising

**Usage Example:**
```python
result = await query_case(str(case_id), "What was the verdict?", db)
# Returns: {
#   "answer": "The court decided...",
#   "sources": [...],
#   "confidence": {"level": "high", "score": 0.85, ...},
#   "error": None
# }
```

---

### backend/services/job_processor.py

Background job processor for asynchronous document processing.

#### Functions

##### `get_pending_jobs(db: Session, limit: int = 5) -> List[ProcessingJob]`

**Purpose:** Retrieve pending jobs in FIFO order.

**Parameters:**
- `db` (Session): Database session
- `limit` (int): Max jobs to retrieve (default: 5)

**Return Value:** List of ProcessingJob models in pending status

---

##### `calculate_retry_delay(attempt: int) -> int`

**Purpose:** Calculate retry delay in seconds based on attempt number.

**Parameters:**
- `attempt` (int): Attempt number (1, 2, 3, ...)

**Return Value:** `int` - Delay in seconds

**Schedule:**
- Attempt 1: 0 seconds
- Attempt 2: 5 seconds
- Attempt 3: 10 seconds
- Others: 10 seconds

---

##### `mark_job_complete(case_id: str, db: Session) -> bool`

**Purpose:** Mark job as completed.

**Parameters:**
- `case_id` (str): Case UUID
- `db` (Session): Database session

**Return Value:** `bool` - True if successful, False if job not found

---

##### `mark_job_failed(case_id: str, db: Session, error_message: str, next_retry_at: datetime = None) -> bool`

**Purpose:** Mark job as failed and schedule retry if attempts remaining.

**Parameters:**
- `case_id` (str): Case UUID
- `db` (Session): Database session
- `error_message` (str): Error description
- `next_retry_at` (datetime, optional): Scheduled retry time

**Return Value:** `bool` - True if successful

**Behavior:**
- Increments attempt counter
- Sets status to "failed" if max_attempts exceeded
- Sets status to "pending" if retries remain

---

##### `async process_case(case_id: str, db: Session) -> Dict`

**Purpose:** Process case: download, chunk, embed, and store vectors.

**Parameters:**
- `case_id` (str): Case UUID
- `db` (Session): Database session

**Return Value:** Dict with:
- `success` (bool): Whether processing succeeded
- `chunks_created` (int): Number of chunks created
- `error` (str, optional): Error message if failed

**Steps:**
1. Fetch case from database
2. Download document from blob storage
3. Chunk document (PDF/DOCX/TXT)
4. Delete old chunks (for reprocessing)
5. Generate embeddings
6. Create vector collection
7. Upsert vectors to Qdrant
8. Create Chunk records in DB
9. Update case status to "ready"
10. Update job status to "completed"

**Error Handling:** Catches exceptions, updates case status to "error", returns error dict

---

##### `async run_job_worker(db: Session, max_jobs_per_batch: int = 5, sleep_interval: int = 10, max_iterations: Optional[int] = None) -> None`

**Purpose:** Continuously process pending jobs in batches (worker loop).

**Parameters:**
- `db` (Session): Database session
- `max_jobs_per_batch` (int): Jobs per batch (default: 5)
- `sleep_interval` (int): Sleep seconds between batches (default: 10)
- `max_iterations` (int, optional): Stop after N iterations (for testing)

**Behavior:**
- Polls for pending jobs in FIFO order
- Processes up to max_jobs_per_batch per iteration
- Updates job status to "processing" before processing
- Schedules retries on failure
- Sleeps between batches

**Usage Example:**
```python
# Run worker indefinitely
asyncio.run(run_job_worker(db))

# Run worker for testing (stop after 5 iterations)
asyncio.run(run_job_worker(db, max_iterations=5))
```

---

## Frontend Files

### frontend/app/layout.tsx

Root layout component providing global providers and navigation.

#### Component: `RootLayout`

**Purpose:** Wraps entire application with necessary providers (Next.js root layout).

**Props:**
```typescript
{
  children: React.ReactNode  // Page content
}
```

**Behavior:**
- Wraps children with QueryProvider (React Query)
- Wraps children with AuthProvider (authentication context)
- Renders NavBar component
- Sets global metadata (title, description)

**Structure:**
```tsx
<html>
  <body>
    <QueryProvider>
      <AuthProvider>
        <NavBar />
        <main>{children}</main>
      </AuthProvider>
    </QueryProvider>
  </body>
</html>
```

**Metadata:**
- Title: "LexIntel - Legal RAG"
- Description: "RAG system for legal document analysis"

---

### frontend/app/page.tsx

Home/landing page with authentication redirects.

#### Component: `Home`

**Purpose:** Landing page that redirects authenticated users to dashboard.

**Behavior:**
- Checks authentication status from AuthContext
- Redirects to `/dashboard` if authenticated (useEffect)
- Shows login/register buttons if not authenticated
- Returns null while redirecting to prevent flash

**Content:**
- Heading: "Welcome to LexIntel"
- Subheading: "Legal Document Analysis with RAG"
- Buttons: Login and Register links

**Usage Example:**
```tsx
// Unauthenticated users see:
// Welcome to LexIntel
// [Login] [Register]

// Authenticated users redirected to /dashboard
```

---

### frontend/app/auth/login/page.tsx

User login page with form and authentication.

#### Component: `Login`

**Purpose:** User login with email/password credentials.

**State:**
- `email` (str): User email input
- `password` (str): User password input
- `error` (str | null): Error message

**Handlers:**

##### `handleSubmit(e: React.FormEvent)`

**Purpose:** Handle login form submission with validation.

**Validation:**
- Email required and non-empty
- Password required

**On Success:**
- Sets auth token in context
- Navigates to `/dashboard`

**On Error:** Displays error message from API

**Mutation:** Uses `useMutation` for async login API call

**Usage Example:**
```tsx
// User fills form and submits
// POST /auth/login with {email, password}
// Sets token in AuthContext
// Navigates to dashboard
```

**Form Fields:**
- Email input (type: email)
- Password input (type: password)
- Submit button (disabled while loading)
- Link to registration page

---

### frontend/app/auth/register/page.tsx

User registration page with form validation.

#### Component: `Register`

**Purpose:** Create new user account with email/password.

**State:**
- `email` (str): Email input
- `password` (str): Password input
- `confirmPassword` (str): Password confirmation
- `error` (str | null): Error message

**Handlers:**

##### `handleSubmit(e: React.FormEvent)`

**Purpose:** Handle registration form submission with validation.

**Validation:**
- Email required and non-empty
- Password required
- Password min 6 characters
- Passwords must match

**On Success:**
- Sets auth token in context
- Navigates to `/dashboard`

**On Error:** Displays error message from API

**Mutation:** Uses `useMutation` for async register API call

**Usage Example:**
```tsx
// User fills form and submits
// POST /auth/register with {email, password}
// Auto-login and navigate to dashboard
```

---

### frontend/app/cases/[id]/page.tsx

Case detail page with document query interface.

#### Component: `CaseDetail`

**Purpose:** Display case details and RAG query interface.

**Route Parameters:**
- `id` (str): Case UUID from URL

**State:**
- `question` (str): User query input
- `answer` (QueryAnswer | null): API response
- `copiedIndex` (number | null): Index of copied source

**Queries & Mutations:**

##### Case Status Query

```typescript
useQuery<CaseStatus>({
  queryKey: ['caseStatus', caseId],
  queryFn: async () => GET /cases/{caseId}/status,
  refetchInterval: (query) => {
    // Poll every 2 seconds while processing
    if (query.state.data?.status !== 'processing') {
      return false  // Stop polling when ready/error
    }
    return 2000
  }
})
```

**Purpose:** Poll case processing status

**Response Type:** CaseStatus
- `id` (string): Case UUID
- `status` ('processing' | 'ready' | 'error')
- `error` (string, optional): Error message

---

##### Ask Question Mutation

```typescript
useMutation({
  mutationFn: async (q: string) =>
    POST /cases/{caseId}/ask with {question: q},
  onSuccess: (data) => {
    setAnswer(data)  // Update answer display
    setQuestion('')  // Clear input
  }
})
```

---

**Handlers:**

##### `handleSubmit(e: React.FormEvent)`

**Purpose:** Submit question to RAG engine.

**Validation:** Question must not be empty

**Behavior:** Triggers askMutation with trimmed question

---

##### `handleCopySource(text: string, index: number)`

**Purpose:** Copy source text to clipboard and show feedback.

**Behavior:**
- Copies text to clipboard
- Sets copiedIndex for UI feedback
- Clears feedback after 2 seconds

---

**Rendering:**

**Processing State:**
```tsx
// While status === 'processing'
<SpinnerWithText text="Processing document... This may take a few minutes." />
```

**Error State:**
```tsx
// While status === 'error'
<Alert variant="destructive">
  {caseStatus.error || 'An error occurred...'}
</Alert>
```

**Ready State:**
- Query form with textarea for questions
- Answer display (if available)
- Sources/Citations with copy buttons

**Answer Display:**
```tsx
<div className="bg-blue-50 border border-blue-200 rounded p-4">
  {answer.answer}  // Formatted with whitespace preserved
</div>
```

**Sources Display:**
```tsx
{answer.sources && answer.sources.length > 0 && (
  <div>
    {answer.sources.map((source, index) => (
      <div key={index}>
        <p>Page {source.page}</p>
        <button onClick={() => handleCopySource(source.text, index)}>
          {copiedIndex === index ? 'Copied' : 'Copy'}
        </button>
        <p>{source.text}</p>
      </div>
    ))}
  </div>
)}
```

---

### frontend/app/dashboard/page.tsx

Dashboard with document upload form.

#### Component: `Dashboard`

**Purpose:** User dashboard for uploading new legal documents.

**State:**
- `caseName` (str): Case name input
- `selectedFile` (File | null): Selected document
- `error` (str | null): Error message

**Handlers:**

##### `handleFileSelect(file: File)`

**Purpose:** Update selected file when user chooses document.

**Parameters:**
- `file` (File): Selected file from upload component

**Behavior:** Sets selectedFile state and clears errors

---

##### `handleSubmit(e: React.FormEvent)`

**Purpose:** Submit case upload with validation.

**Validation:**
- Case name required and non-empty
- File must be selected

**Behavior:**
- Creates FormData with case_name and file
- Triggers uploadMutation
- Navigates to case detail page on success

---

**Mutation:** `uploadMutation`

```typescript
useMutation({
  mutationFn: async (formData: FormData) =>
    POST /cases with FormData,
  onSuccess: (data) => {
    router.push(`/cases/${data.id}`)  // Redirect to case page
  },
  onError: (err) => {
    setError(err.response?.data?.detail || 'Failed to upload document')
  }
})
```

---

**Form Fields:**
- Case Name input (text)
- Document upload zone (FileUploadZone component)
- Selected file info (name, size in MB)
- Submit button (disabled while uploading or incomplete)
- Error alert (if validation fails)

---

### frontend/components/file-upload-zone.tsx

Reusable drag-and-drop file upload component.

#### Props

```typescript
interface FileUploadZoneProps {
  onFileSelect: (file: File) => void  // Callback when file selected
  isLoading?: boolean                  // Disable during upload
}
```

#### Constants

```typescript
const MAX_FILE_SIZE = 50 * 1024 * 1024  // 50MB
const ALLOWED_TYPE = 'application/pdf'  // PDF only
```

#### Component: `FileUploadZone`

**Purpose:** Reusable upload component with drag-and-drop and validation.

**State:**
- `isDragging` (bool): Is file being dragged over
- `error` (str | null): Validation error message

**Handlers:**

##### `validateFile(file: File) -> string | null`

**Purpose:** Validate file MIME type and size.

**Validation Rules:**
- Must be PDF (application/pdf)
- Max 50MB

**Return:** Error message or null if valid

---

##### `handleFile(file: File)`

**Purpose:** Process selected file with validation.

**Behavior:**
- Validates file
- Calls onFileSelect callback if valid
- Sets error state if invalid

---

##### `handleDragOver(e: React.DragEvent)`

**Purpose:** Handle drag over event.

**Behavior:** Prevents default, sets isDragging = true

---

##### `handleDragLeave(e: React.DragEvent)`

**Purpose:** Handle drag leave event.

**Behavior:** Prevents default, sets isDragging = false

---

##### `handleDrop(e: React.DragEvent)`

**Purpose:** Handle file drop.

**Behavior:**
- Prevents default
- Extracts first file from transfer
- Calls handleFile

---

##### `handleFileInputChange(e: React.ChangeEvent<HTMLInputElement>)`

**Purpose:** Handle file input change from click dialog.

**Behavior:** Extracts file and calls handleFile

---

##### `handleClick()`

**Purpose:** Open file dialog when zone clicked.

**Behavior:** Triggers hidden input click unless loading

---

**UI:**
```tsx
<div
  onClick={handleClick}
  onDragOver={handleDragOver}
  onDragLeave={handleDragLeave}
  onDrop={handleDrop}
  className={`border-2 border-dashed p-8
    ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50'}
    ${isLoading ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}
  `}
>
  <Upload icon />
  <p>Drag and drop your PDF here, or click to browse</p>
  <p>PDF only, max 50MB</p>
</div>
```

**Error Display:**
```tsx
{error && (
  <Alert variant="destructive">
    <AlertCircle />
    {error}
  </Alert>
)}
```

---

### frontend/components/navbar.tsx

Navigation bar component shown on all pages.

#### Component: `NavBar`

**Purpose:** Global navigation with authentication state.

**State:**
- Uses `useAuth()` hook for `isAuthenticated` and `setToken`

**Handlers:**

##### `handleLogout()`

**Purpose:** Log out current user.

**Behavior:**
- Sets token to null (clears AuthContext)
- Redirects to home page

---

**UI:**
```tsx
<nav className="bg-white shadow">
  <div className="flex justify-between items-center">
    <Link href="/">LexIntel</Link>
    <div className="flex gap-4">
      {isAuthenticated ? (
        <>
          <Link href="/dashboard">Dashboard</Link>
          <button onClick={handleLogout}>Logout</button>
        </>
      ) : null}
    </div>
  </div>
</nav>
```

**Always Visible:**
- Logo/home link

**Authenticated Only:**
- Dashboard link
- Logout button (red)

---

### frontend/lib/auth-context.tsx

Authentication context for managing user session state.

#### Types

```typescript
interface AuthContextType {
  token: string | null           // JWT access token
  isAuthenticated: boolean       // token !== null
  setToken: (token: string | null) => void
}
```

#### Component: `AuthProvider`

**Purpose:** Provider component for authentication context.

**Props:**
```typescript
{
  children: React.ReactNode  // Child components
}
```

**State:**
- `token` (string | null): JWT access token
- `isMounted` (bool): Hydration flag for SSR safety

**Effects:**

##### Mount Effect

**Purpose:** Load token from localStorage on component mount.

**Behavior:**
- Checks if window is defined (client-side)
- Reads 'access_token' from localStorage
- Sets isMounted = true

**Usage:** Prevents hydration mismatch errors

---

##### `setToken(newToken: string | null) -> void`

**Purpose:** Update token state and persist to localStorage.

**Behavior:**
- Updates state
- Stores in localStorage if newToken provided
- Removes from localStorage if newToken is null

**Usage Example:**
```typescript
const { setToken } = useAuth()
setToken(jwtToken)  // Login
setToken(null)      // Logout
```

---

#### Hook: `useAuth()`

**Purpose:** Use authentication context in components.

**Return Value:** `AuthContextType` with token, isAuthenticated, setToken

**Error Handling:** Throws error if used outside AuthProvider

**Usage Example:**
```tsx
export default function MyComponent() {
  const { isAuthenticated, setToken } = useAuth()

  if (!isAuthenticated) {
    return <div>Please log in</div>
  }

  return <div>Welcome!</div>
}
```

---

**Hydration Safety:**
- Returns null until mounted (prevents SSR mismatch)
- Loads token from localStorage only in browser
- Ensures consistent client/server rendering

---

### frontend/lib/query-provider.tsx

React Query provider setup for data management.

#### Component: `QueryProvider`

**Purpose:** Set up React Query client for the application.

**Props:**
```typescript
{
  children: ReactNode  // Child components
}
```

**Configuration:**
```typescript
const queryClient = new QueryClient()
```

**Default Settings:**
- Query retry: 3 times
- Query stale time: 5 minutes
- Cache time: 10 minutes
- Refetch on focus: enabled

**Usage Example:**
```tsx
// In layout.tsx
<QueryProvider>
  <MyApp />
</QueryProvider>

// In component
const { data, isLoading } = useQuery({
  queryKey: ['cases'],
  queryFn: () => apiClient.get('/cases')
})
```

**Benefits:**
- Automatic caching
- Request deduplication
- Refetch management
- Error handling
- Loading states

---

## Cross-References and Dependencies

### Key API Flow
1. **Registration** → POST /auth/register → Create User → JWT token
2. **Login** → POST /auth/login → Verify credentials → JWT token
3. **Upload Case** → POST /cases → Store blob → Queue job → Async processing
4. **Query Case** → POST /cases/{id}/ask → Vector search → RAG → Answer
5. **Job Processing** → Download → Chunk → Embed → Upsert vectors

### Database Schema
- User (1) ← → (many) Case
- Case (1) ← → (many) Chunk
- Case (1) ← → (many) Query
- Case (1) ← → (many) ProcessingJob

### Service Dependencies
- RAG Engine → Embeddings → Vector Store → Qdrant
- RAG Engine → LLM (OpenAI GPT-4o)
- Job Processor → Text Extraction → Chunking
- Main.py → All services

### Frontend Component Tree
```
RootLayout
├── QueryProvider
├── AuthProvider
├── NavBar
└── Main
    ├── page.tsx (home)
    ├── dashboard/page.tsx
    ├── cases/[id]/page.tsx
    ├── auth/login/page.tsx
    └── auth/register/page.tsx
```

---

## Error Handling Summary

### HTTP Status Codes
- **400**: Bad Request (validation failure)
- **401**: Unauthorized (missing/invalid token)
- **403**: Forbidden (authorization check failed)
- **404**: Not Found (resource missing)
- **500**: Internal Server Error (unexpected failure)

### Custom Exceptions
- `LexIntelException` - Base exception
- `BlobUploadException` - Storage failures
- `EmbeddingException` - Embedding service failures
- `VectorStoreException` - Vector DB failures
- `QueryProcessingException` - RAG pipeline failures
- `ValidationException` - Input validation failures

### Frontend Error Handling
- Alert components for error display
- Mutation error handling with user feedback
- Graceful fallbacks (return null, show spinner)
- Copy button feedback with timeout

---

End of File Reference Guide
