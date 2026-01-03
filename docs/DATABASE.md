# Database Models & Schema

> PostgreSQL database with pgvector for semantic search

---

## 📐 Database Schema

```
┌─────────────────────────────────────────────────────────────┐
│                         CASES                              │
├─────────────────────────────────────────────────────────────┤
│ id (PK, UUID)                                              │
│ name (String)                                              │
│ case_number (String, UNIQUE)                               │
│ practice_area (String)                                     │
│ status (Enum: active/closed/archived)                      │
│ description (Text)                                         │
│ created_at (DateTime)                                      │
│ updated_at (DateTime)                                      │
└─────────────────────────────────────────────────────────────┘
         │
         │ 1:N
         └──────────┐
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                      DOCUMENTS                              │
├─────────────────────────────────────────────────────────────┤
│ id (PK, UUID)                                              │
│ case_id (FK → Cases)                                       │
│ title (String)                                             │
│ filename (String)                                          │
│ type (Enum: brief/complaint/discovery/statute/...)        │
│ extracted_text (Text)                                      │
│ page_count (Integer)                                       │
│ file_size (Integer)                                        │
│ file_path (String)                                         │
│ processing_status (Enum: pending/extracted/indexed/failed) │
│ error_message (Text)                                       │
│ indexed_at (DateTime)                                      │
│ created_at (DateTime)                                      │
│ updated_at (DateTime)                                      │
└─────────────────────────────────────────────────────────────┘
         │
         │ 1:N
         └──────────┐
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                   DOCUMENT_CHUNKS                           │
├─────────────────────────────────────────────────────────────┤
│ id (PK, UUID)                                              │
│ document_id (FK → Documents)                               │
│ chunk_text (Text)                                          │
│ chunk_index (Integer)                                      │
│ embedding (pgvector, 1536 dimensions)                      │
│ search_vector (tsvector for full-text search)              │
└─────────────────────────────────────────────────────────────┘

         ┌──────────────────────────┐
         │                          │
         │  CHAT_CONVERSATIONS      │
         │                          │
         │  id (PK, UUID)           │
         │  case_id (FK → Cases)    │
         │  title (String)          │
         │  token_count (Integer)   │
         │  message_count (Integer) │
         │  created_at (DateTime)   │
         │  updated_at (DateTime)   │
         └──────────────────────────┘
                    │
                    │ 1:N
                    ↓
         ┌──────────────────────────┐
         │   CHAT_MESSAGES          │
         │                          │
         │  id (PK, UUID)           │
         │  conversation_id (FK)    │
         │  role (String)           │
         │  content (Text)          │
         │  tokens_used (Integer)   │
         │  source_document_ids []  │
         │  created_at (DateTime)   │
         │  updated_at (DateTime)   │
         └──────────────────────────┘
```

---

## 🗄️ Models (app/models/)

### Base Model (base.py)

```python
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, DateTime, func

class Base(DeclarativeBase):
    """Base class for all ORM models"""
    pass

class TimestampMixin:
    """Add created_at and updated_at to any model"""
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
```

---

### Case Model (case.py)

```python
from sqlalchemy import Column, String, Text, Enum
from app.models.base import Base, TimestampMixin
import enum

class CaseStatus(str, enum.Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"

class Case(Base, TimestampMixin):
    __tablename__ = "cases"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    case_number = Column(String, unique=True, nullable=False, index=True)
    practice_area = Column(String, nullable=False)
    status = Column(Enum(CaseStatus), default=CaseStatus.ACTIVE, nullable=False, index=True)
    description = Column(Text)

    # Relationships (lazy loaded)
    documents = relationship("Document", back_populates="case", cascade="all, delete-orphan")
    chats = relationship("ChatConversation", back_populates="case", cascade="all, delete-orphan")
```

**Indexes**:
- `case_number` (unique)
- `status` (frequently filtered)

---

### Document Model (document.py)

```python
from sqlalchemy import Column, String, Text, Integer, ForeignKey, ARRAY, Enum, DateTime
from sqlalchemy.orm import relationship
from app.models.base import Base, TimestampMixin
import enum

class DocumentType(str, enum.Enum):
    BRIEF = "brief"
    COMPLAINT = "complaint"
    DISCOVERY = "discovery"
    STATUTE = "statute"
    TRANSCRIPT = "transcript"
    CONTRACT = "contract"
    EVIDENCE = "evidence"
    OTHER = "other"

class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"              # Uploaded, waiting to process
    EXTRACTED = "extracted"          # Text extracted, chunked
    INDEXED = "indexed"              # Embeddings generated
    FAILED = "failed"                # Processing failed

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    type = Column(Enum(DocumentType), nullable=False)
    extracted_text = Column(Text)                          # Full extracted text
    page_count = Column(Integer)
    file_size = Column(Integer)
    file_path = Column(String)                             # /app/uploads/{doc_id}/file.pdf
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, index=True)
    error_message = Column(Text)                           # If processing failed
    indexed_at = Column(DateTime)                          # When indexing completed

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)             # 4000 chars per chunk
    chunk_index = Column(Integer, nullable=False)         # Order of chunks
    embedding = Column(String)                            # pgvector (JSON string)
    search_vector = Column(String)                        # PostgreSQL tsvector

    # Relationships
    document = relationship("Document", back_populates="chunks")
```

**Indexes**:
- `case_id` (FK lookup)
- `processing_status` (status filtering)
- `document_id` (FK lookup)
- `embedding` (pgvector index - created separately)

---

### Chat Models

```python
from sqlalchemy import Column, String, Text, Integer, ForeignKey, ARRAY
from sqlalchemy.orm import relationship
from app.models.base import Base, TimestampMixin

class ChatConversation(Base, TimestampMixin):
    __tablename__ = "chat_conversations"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    title = Column(String, default="Untitled Conversation")
    token_count = Column(Integer, default=0)              # Total tokens used
    message_count = Column(Integer, default=0)            # Number of messages

    # Relationships
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")

class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("chat_conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)                 # "user" or "assistant"
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, default=0)
    source_document_ids = Column(ARRAY(String), default=[])  # Which docs were used

    # Relationships
    conversation = relationship("ChatConversation", back_populates="messages")
```

---

## 🔧 Database Operations

### Initialize Database

```python
# app/database.py
async def init_db():
    """Initialize database tables and extensions"""
    from app.models.base import Base

    async with engine.begin() as conn:
        # Enable pgvector for semantic search
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Enable pg_trgm for full-text search
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

# Called in FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
```

---

## 🔍 Query Examples

### Get Case with Documents
```python
stmt = select(Case).where(Case.id == case_id)
result = await db.execute(stmt)
case = result.scalar_one_or_none()

# Access related documents (lazy loaded)
for doc in case.documents:
    print(doc.title)
```

### Get Documents by Status
```python
from sqlalchemy import and_

stmt = select(Document).where(
    and_(
        Document.case_id == case_id,
        Document.processing_status == ProcessingStatus.INDEXED
    )
)
result = await db.execute(stmt)
documents = result.scalars().all()
```

### Get Document Chunks with Embeddings
```python
stmt = select(DocumentChunk).where(
    and_(
        DocumentChunk.document_id == doc_id,
        DocumentChunk.embedding != None
    )
)
result = await db.execute(stmt)
chunks = result.scalars().all()
```

---

## 📊 Indexes & Performance

### Automatic Indexes
```python
# Primary keys
id = Column(String, primary_key=True)

# Foreign keys
case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)

# Frequently filtered fields
status = Column(Enum(...), index=True)
processing_status = Column(Enum(...), index=True)
```

### To Add: Vector Indexes
```sql
-- For pgvector semantic search (speeds up similarity queries)
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### To Add: Full-Text Indexes
```sql
-- For faster full-text search
CREATE INDEX ON document_chunks USING gin (search_vector);
```

---

## 🔐 Data Integrity

### Cascade Deletes
```python
# Deleting a Case deletes all related Documents & DocumentChunks
case.delete()  # Cascades to documents → chunks → chats → messages

# Deleting a Document deletes chunks
document.delete()  # Cascades to chunks
```

### Unique Constraints
```python
case_number = Column(String, unique=True, nullable=False)  # No duplicate case numbers
```

### Not Null Constraints
```python
id = Column(String, primary_key=True)                      # Always required
case_id = Column(String, ForeignKey(...), nullable=False)  # Always required
```

---

## 📝 Data Types

### String (Text Fields)
- `String` - Variable length, indexed (case_number, filename, title)
- `Text` - Large text, not indexed (extracted_text, content)

### Numeric
- `Integer` - Whole numbers (page_count, file_size, tokens_used)

### Enums
- `Enum(...)` - Fixed set of values, indexed (status, processing_status)

### Arrays
- `ARRAY(String)` - List of strings (source_document_ids)

### Timestamps
- `DateTime` - With server defaults and auto-updates (created_at, updated_at)

### Vector
- pgvector format - Stored as JSON string in `embedding` column

---

## 🚀 Migrations (Future)

When models change, use Alembic:

```bash
# Create migration
alembic revision --autogenerate -m "Add new field"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

Currently using SQLAlchemy `create_all()` for MVP. Switch to Alembic for production.

---

## 📚 References

- **SQLAlchemy ORM**: https://docs.sqlalchemy.org/en/20/orm/
- **Async SQLAlchemy**: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- **pgvector**: https://github.com/pgvector/pgvector
- **PostgreSQL Full-Text**: https://www.postgresql.org/docs/current/textsearch.html
