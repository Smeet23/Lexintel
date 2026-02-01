# LexIntel Technology Stack

## Overview

LexIntel is a legal document Retrieval-Augmented Generation (RAG) system built with a modern, scalable architecture. This document provides comprehensive information about all technologies, versions, and infrastructure components used in the system.

---

## Backend Technologies

### FastAPI & Web Framework

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **FastAPI** | 0.109.0 | REST API Framework | High-performance async Python web framework for building APIs |
| **Uvicorn** | 0.27.0 | ASGI Server | Lightning-fast ASGI server for running FastAPI application |
| **Python-multipart** | 0.0.6 | File Upload Handling | Supports multipart/form-data for document uploads |

### Database & ORM

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **PostgreSQL** | 14+ | Primary Database | Relational database for users, cases, queries, and metadata |
| **SQLAlchemy** | 2.0.23 | ORM | SQL toolkit and Object-Relational Mapping |
| **Psycopg2-binary** | 2.9.9 | PostgreSQL Adapter | PostgreSQL adapter for Python |
| **Alembic** | 1.13.0 | Database Migrations | SQL schema migration tool for version control |

### Vector Database

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Qdrant** | 1.7.0 | Vector Store | Semantic search and similarity matching for document embeddings |
| **Qdrant-client** | 1.16.1 | Python Client | Official Python client for Qdrant API |

### RAG Pipeline & NLP

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **LangChain** | 0.1.20 | RAG Orchestration | Framework for building LLM-powered applications |
| **LangChain-community** | 0.0.38 | Community Integrations | Community-contributed integrations for LangChain |
| **LangChain-openai** | 0.0.8 | OpenAI Integration | LangChain binding for OpenAI models |
| **OpenAI** | 1.12.0 | LLM API | GPT-4o for generation, text-embedding-3-large for embeddings |
| **Sentence-transformers** | 2.2.2 | Embeddings | Alternative embedding model support |
| **Tiktoken** | 0.5.2 | Token Counting | OpenAI token counting for rate limiting |

### Document Processing

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **PyMuPDF** | 1.24.0 | PDF Processing | PDF text extraction and manipulation |
| **Python-docx** | 1.1.0 | DOCX Processing | Microsoft Word document parsing |
| **Docling** | Not included | Multi-format Support | Recommended for production (Python 3.10+); currently uses PyMuPDF + python-docx |

### Cloud Storage

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Azure-storage-blob** | 12.19.0 | Cloud Storage | Azure Blob Storage for document persistence |
| **Azurite** | Latest | Storage Emulator | Local Azure Storage emulation for development |

### Asynchronous Processing

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Celery** | 5.3.4 | Task Queue | Distributed task queue for background jobs |
| **Redis** | 7 | Message Broker | Message broker and result backend for Celery |
| **Aiofiles** | 23.2.1 | Async File I/O | Asynchronous file operations |

### Authentication & Security

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Python-jose** | 3.3.0 | JWT Handling | JWT creation and verification |
| **Passlib** | 1.7.4 | Password Hashing | Password hashing utilities (Note: Consider upgrading to latest version) |
| **Bcrypt** | 4.1.1 | Cryptographic Hash | Strong password hashing algorithm |
| **Email-validator** | 2.1.0 | Email Validation | RFC 5321/5322 compliant email validation |

### Configuration & Validation

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Pydantic** | 2.5.3 | Data Validation | Data validation using Python type hints |
| **Pydantic-settings** | 2.1.0 | Settings Management | Environment configuration management |
| **Python-dotenv** | 1.0.0 | Environment Variables | Load environment variables from .env files |

### Testing

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Pytest** | 7.4.3 | Testing Framework | Python testing framework |
| **Pytest-cov** | 4.1.0 | Coverage Analysis | Code coverage measurement |
| **Pytest-asyncio** | 0.21.1 | Async Testing | Testing support for async functions |
| **HTTPx** | 0.25.2 | HTTP Client | Modern async/sync HTTP client for testing |

### Utilities

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **SQLAlchemy-utils** | 0.41.1 | Database Utilities | Utility functions and custom types for SQLAlchemy |

---

## Frontend Technologies

### Core Framework

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Next.js** | 14.0.0 | React Framework | Production-grade React framework with SSR, SSG, and API routes |
| **React** | 18.2.0 | UI Library | JavaScript library for building user interfaces |
| **TypeScript** | 5.9.3 | Type Safety | Superset of JavaScript with static typing |

### Styling & UI

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Tailwind CSS** | 4.1.18 | CSS Framework | Utility-first CSS framework for rapid UI development |
| **Tailwind-merge** | 3.4.0 | CSS Optimization | Merge Tailwind CSS class names without conflicts |
| **Autoprefixer** | 10.4.23 | CSS Processing | Add vendor prefixes automatically |
| **PostCSS** | 8.5.6 | CSS Parser | Transform CSS with JavaScript plugins |

### UI Components & Icons

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **@radix-ui/react-slot** | 1.2.4 | Component Composition | Composable component system |
| **Lucide-react** | 0.563.0 | Icon Library | Beautiful, consistent React icon library |
| **class-variance-authority** | 0.7.1 | Component Variants | CSS-in-JS pattern for component variants |
| **Clsx** | 2.1.1 | Class Utilities | Utility for constructing className strings |

### HTTP & Data Fetching

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **Axios** | 1.13.3 | HTTP Client | Promise-based HTTP client for API requests |
| **@tanstack/react-query** | 5.90.20 | Data Fetching | Server state management and caching |

### Development & Linting

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **ESLint** | 8.0.0+ | Code Linting | JavaScript code quality and style |
| **eslint-config-next** | 14.0.0 | Next.js Linting | ESLint configuration for Next.js |

### Type Definitions

| Technology | Version | Purpose | Details |
|-----------|---------|---------|---------|
| **@types/node** | 25.0.10 | Node.js Types | TypeScript definitions for Node.js |
| **@types/react** | 19.2.9 | React Types | TypeScript definitions for React |

---

## Infrastructure & Deployment

### Containerization

| Technology | Purpose | Details |
|-----------|---------|---------|
| **Docker** | Container Runtime | Containerizes backend, frontend, and worker services |
| **Docker Compose** | Multi-container Orchestration | Defines and manages all services in development and deployment |

### Core Services

#### PostgreSQL Database
```yaml
Image: postgres:16-alpine
Port: 5432
Volumes: postgres_data
Environment:
  - POSTGRES_USER: legal_user
  - POSTGRES_PASSWORD: (configurable)
  - POSTGRES_DB: legal_rag
  - max_connections: 50
  - shared_buffers: 256MB
Health Check: pg_isready
```

**Purpose:** Stores user accounts, case information, queries, and metadata
**Storage:** Persistent volume (postgres_data)

#### Qdrant Vector Database
```yaml
Image: qdrant/qdrant:v1.7.0
Port: 6333
Volumes: qdrant_data
Health Check: /readyz endpoint
```

**Purpose:** Manages document embeddings and semantic search
**Storage:** Persistent volume (qdrant_data)
**Endpoint:** http://qdrant:6333

#### Redis Cache & Broker
```yaml
Image: redis:7-alpine
Port: 6379
Volumes: redis_data
```

**Purpose:** Celery message broker and result backend, query caching
**Storage:** Persistent volume (redis_data)
**Database Configuration:**
- **Database 0 (redis://redis:6379/0):** Celery task queue broker - stores pending task messages
- **Database 1 (redis://redis:6379/1):** Celery result backend - stores completed task results and status

#### Azurite (Azure Storage Emulator)
```yaml
Image: mcr.microsoft.com/azure-storage/azurite:latest
Ports: 10000, 10001
Volumes: azurite_data
Command: azurite-blob --blobHost 0.0.0.0 --blobPort 10000
```

**Purpose:** Local development alternative to Azure Blob Storage
**Storage:** Persistent volume (azurite_data)
**Port Details:**
- **Port 10000:** Blob storage service (used by application)
- **Port 10001:** Queue storage service

#### FastAPI Backend
```yaml
Build: ./backend/Dockerfile
Port: 8000
Depends On: postgres, qdrant, azurite, redis
Health Check: http://localhost:8000/health
```

**Purpose:** RESTful API for document management and querying
**Base Runtime:** Python 3.11-slim
**Non-root User:** appuser (UID 1000)
**CMD:** uvicorn main:app --host 0.0.0.0 --port 8000

#### Celery Worker
```yaml
Build: ./backend/Dockerfile
Command: celery -A celery_app worker -l info
Depends On: postgres, redis
```

**Purpose:** Background task processing for document indexing and retrieval
**Runtime:** Shares backend image

#### Next.js Frontend
```yaml
Build: ./frontend/Dockerfile
Port: 3000
Base Image: node:18-alpine
Depends On: backend
CMD: npm run dev (development)
```

**Purpose:** Web interface for document management and legal research
**Development Command:** npm run dev
**Production Command:** npm run build && npm start

### Persistent Volumes

| Volume | Purpose | Mount Point |
|--------|---------|------------|
| **postgres_data** | PostgreSQL database files | /var/lib/postgresql/data |
| **qdrant_data** | Vector database storage | /qdrant/storage |
| **redis_data** | Redis persistence | /data |
| **azurite_data** | Azure Storage emulator data | /data |
| **lexintel_storage** | Application temporary storage | /tmp/lexintel_storage |

### Docker Network

**Network Name:** lexintel (bridge)

All services communicate over this isolated network:
- postgres:5432
- qdrant:6333
- redis:6379
- azurite:10000
- backend:8000
- frontend:3000

---

## Configuration Management

### Environment Variables

#### Database Configuration
```
DATABASE_URL=postgresql://legal_user:password@postgres:5432/legal_rag
POSTGRES_USER=legal_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=legal_rag
```

#### OpenAI API Configuration
```
OPENAI_API_KEY=sk-xxx
```

**Note:** Required for embeddings and LLM generation

#### Vector Database Configuration
```
QDRANT_URL=http://localhost:6333
```

#### Azure Blob Storage Configuration
```
AZURE_STORAGE_CONNECTION_STRING=...
```

**Development:** Uses Azurite emulator
**Production:** Use actual Azure credentials

#### Message Broker Configuration
```
CELERY_BROKER_URL=redis://redis:6379/0           # Database 0: task queue
CELERY_RESULT_BACKEND=redis://redis:6379/1       # Database 1: task results
REDIS_URL=redis://localhost:6379/0                # Cache URL (typically uses Database 0)
```

**Note:** Two-database setup isolates task messages (DB 0) from task results (DB 1) to prevent conflicts

#### Security Configuration
```
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

#### CORS Configuration
```
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

**Format:** Comma-separated list (no spaces) of allowed frontend origins for CORS

**Examples:**
- Single origin: `http://localhost:3000`
- Multiple origins: `http://localhost:3000,https://app.example.com,https://staging.example.com`
- Production: `https://yourdomain.com` (no localhost)

#### Application Environment
```
DEBUG=True (development) | False (production)
ENVIRONMENT=development | production
```

#### Query Caching
```
CACHE_ENABLED=True
CACHE_TTL_SECONDS=86400
```

#### Frontend Environment Variables
```
NEXT_PUBLIC_API_URL=http://localhost:8000 (development) | https://api.yourdomain.com (production)
```

**Note:** Frontend environment variables must be prefixed with `NEXT_PUBLIC_` to be accessible in the browser. This is the API endpoint for backend communication.

**Development:** http://localhost:8000
**Production:** https://yourdomain.com/api or dedicated API domain

### Settings Class (Backend)

Located in `/backend/config.py`

```python
class Settings(BaseSettings):
    # Database
    database_url: str

    # OpenAI
    openai_api_key: str

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Redis
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Azure
    azure_storage_connection_string: str

    # JWT
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # CORS
    allowed_origins: str = "http://localhost:3000"

    # Environment
    debug: bool = False

    # Query Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400
```

---

## API Integration Matrix

### External Services

| Service | Purpose | Endpoints | Authentication | Response Format |
|---------|---------|-----------|-----------------|-----------------|
| **OpenAI** | LLM & Embeddings | https://api.openai.com/v1 | API Key (Bearer) | JSON |
| **Azure Blob Storage** | Document Storage | https://{account}.blob.core.windows.net | Connection String | Binary/Metadata |
| **Qdrant** | Vector Search | http://qdrant:6333 | None (internal) | JSON |
| **PostgreSQL** | Data Storage | localhost:5432 | Connection String | SQL Protocol |
| **Redis** | Caching/Messaging | redis:6379 | None (internal) | RESP Protocol |

### OpenAI Models & Endpoints

| Endpoint | Model | Purpose | Input | Output |
|----------|-------|---------|-------|--------|
| `/v1/embeddings` | text-embedding-3-large | Document & query embeddings | Text | Vector (3072 dims) |
| `/v1/chat/completions` | gpt-4o | Legal analysis and generation | Text/Context | Text |

### Backend API Endpoints

#### Authentication Endpoints
```
POST   /api/auth/register              - User registration
POST   /api/auth/login                 - User login
POST   /api/auth/logout                - User logout
POST   /api/auth/refresh               - Token refresh
```

#### Case Management Endpoints
```
GET    /api/cases                      - List user's cases
POST   /api/cases                      - Create new case
GET    /api/cases/{case_id}            - Get case details
PUT    /api/cases/{case_id}            - Update case
DELETE /api/cases/{case_id}            - Delete case
```

#### Document Management Endpoints
```
POST   /api/documents                  - Upload document
GET    /api/documents/{doc_id}         - Get document info
DELETE /api/documents/{doc_id}         - Delete document
GET    /api/cases/{case_id}/documents  - List case documents
```

#### Query & RAG Endpoints
```
POST   /api/query                      - Execute RAG query
GET    /api/query/{query_id}           - Get query results
GET    /api/cases/{case_id}/queries    - List case queries
```

#### Health & Status
```
GET    /health                         - API health check
GET    /docs                           - Swagger API documentation
GET    /redoc                          - ReDoc API documentation
```

### Frontend API Communication

**Base URL (Development):** http://localhost:8000
**Base URL (Production):** Configurable via NEXT_PUBLIC_API_URL

**HTTP Client:** Axios with React Query
**Headers:** Content-Type: application/json, Authorization: Bearer <token>

---

## Version Compatibility

### Python Compatibility

| Component | Minimum Version | Recommended | Maximum |
|-----------|-----------------|-------------|---------|
| **Python** | 3.9 | 3.11 | 3.12 |
| **Backend Runtime** | 3.9 | 3.11-slim | 3.12 |

**Current:** Python 3.11-slim (in Dockerfile)

### Node.js & Frontend Compatibility

| Component | Minimum Version | Recommended | Maximum |
|-----------|-----------------|-------------|---------|
| **Node.js** | 16 | 18 | 20 |
| **NPM** | 8.x | 9.x | 10.x |

**Current:** Node 18-alpine (in Dockerfile)

**Node.js 18 Rationale:**
- Long-term support (LTS) until April 2025
- Stable and widely tested in production systems
- Better performance than Node 16, maintains compatibility
- Node 20 LTS available; upgrade path after Next.js 14 stabilizes further
- Alpine variant minimizes container size (~150MB vs ~350MB with full image)

### Database Compatibility

| Component | Minimum Version | Recommended | Maximum |
|-----------|-----------------|-------------|---------|
| **PostgreSQL** | 14 | 14 | 16 |
| **Qdrant** | 1.0 | 1.7.0 | 2.0+ |
| **Redis** | 6.0 | 7.0 | 7.2+ |

### Browser Support (Frontend)

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | Latest 2 versions | Fully supported |
| Firefox | Latest 2 versions | Fully supported |
| Safari | Latest 2 versions | Fully supported |
| Edge | Latest 2 versions | Fully supported |
| Mobile | Chrome/Safari | Responsive design |

### Dependency Compatibility Notes

- **FastAPI 0.109.0** requires Python 3.7+ but tested with 3.11
- **SQLAlchemy 2.0.23** uses SQLAlchemy 2.0 style (no legacy mode)
- **Pydantic 2.5.3** requires Python 3.7+ and breaks compatibility with v1
- **LangChain 0.1.20** compatible with OpenAI 1.x
- **Next.js 14.0.0** uses React 18+ and requires Node 16.8+

---

## Performance Characteristics

### Latency Metrics

| Operation | Expected Latency | Conditions | Notes |
|-----------|------------------|-----------|-------|
| **API Health Check** | < 50ms | Cached | Direct response |
| **User Login** | 100-200ms | Database lookup + token generation | Includes password hashing |
| **Create Case** | 50-100ms | Single DB insert | Basic metadata only |
| **Document Upload** | 500ms - 5s | Size dependent | PDF: < 50MB, includes validation |
| **Document Indexing** | 5-30s | Per document | Depends on pages, async task |
| **Embedding Generation** | 2-10s | Per document | OpenAI API latency |
| **Vector Search** | 50-200ms | Qdrant query | Depends on collection size |
| **RAG Query** | 3-15s | End-to-end | Embedding + search + generation |
| **Cache Hit Response** | 10-50ms | Cached result | Redis lookup |

### Throughput Metrics

| Metric | Value | Conditions | Notes |
|--------|-------|-----------|-------|
| **Concurrent Users** | 50+ | Development (4-core machine, 8GB RAM) | Limited by resources |
| **Requests/sec** | 100+ | Simple queries | Depends on endpoint |
| **Documents/case** | 100+ | Unlimited in design | Storage limited |
| **Queries/day** | Unlimited | Rate limited by OpenAI quota | Batch processing supported |
| **Token Budget** | Depends on plan | OpenAI account | GPT-4o input: ~$0.005/1K tokens |
| **Embedding Reuse** | Up to 24hrs | Cache TTL | Saves API calls |

### Storage Metrics

| Component | Storage Per Unit | Growth Rate | Notes |
|-----------|------------------|-------------|-------|
| **PostgreSQL** | ~50KB per case | Linear with cases | Indexed metadata |
| **Qdrant Vectors** | ~24KB per document | Linear with docs | 3072 dims × 8 bytes (float32) |
| **Blob Storage** | 100% of file size | Linear with docs | PDF, DOCX, TXT support |
| **Redis Cache** | Variable | TTL-based eviction | 24hr default TTL |
| **Vector Index** | 2-3x of vectors | Index overhead | Depends on similarity metric |

### Network Bandwidth

| Operation | Bandwidth | Direction | Notes |
|-----------|-----------|-----------|-------|
| **Document Upload** | File size | Client → Server | Multipart/form-data |
| **Query Response** | 1-10MB | Server → Client | JSON with citations |
| **Embedding Sync** | 1-50KB per doc | Backend → Qdrant | Batch inserts |
| **Cache Sync** | < 1KB per cache hit | Backend → Redis | Serialized results |

---

## Deployment Recommendations

### Development Environment

**Setup:**
```bash
# Requirements
- Docker Desktop (with Docker Compose)
- Python 3.11+ (for local testing without Docker)
- Node.js 18+
- OpenAI API key

# Start all services
docker-compose up --build

# Services available at:
# - Backend: http://localhost:8000
# - Frontend: http://localhost:3000
# - Qdrant: http://localhost:6333
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
```

**Configuration:**
```
DEBUG=True
ENVIRONMENT=development
AZURE_STORAGE_CONNECTION_STRING=UseDevelopmentStorage=true
```

**Resource Requirements:**
- CPU: 4 cores minimum
- RAM: 8GB minimum (16GB recommended)
- Disk: 20GB for databases + volumes

**Health Checks:**
- Backend: `curl http://localhost:8000/health`
- Qdrant: `curl http://localhost:6333/readyz`
- PostgreSQL: `pg_isready -h localhost -p 5432`
- Redis: `redis-cli ping`

### Staging Environment

**Infrastructure:**
```yaml
- Docker Swarm or Kubernetes cluster
- Managed PostgreSQL (AWS RDS, Azure Database)
- Managed Redis (AWS ElastiCache, Azure Cache)
- Managed Qdrant (Qdrant Cloud)
- Real Azure Blob Storage (not emulator)
```

**Configuration:**
```
DEBUG=False
ENVIRONMENT=staging
USE_REAL_SERVICES=True
ENABLE_MONITORING=True
```

**Resource Requirements:**
- Backend: 2 CPU, 4GB RAM per instance (min 2 replicas)
- PostgreSQL: 2 CPU, 8GB RAM, 100GB SSD
- Redis: 1 CPU, 2GB RAM (single node acceptable)
- Qdrant: 2 CPU, 4GB RAM, 50GB SSD

**Scaling:**
- Horizontal scaling via load balancer
- Auto-scaling based on CPU/memory
- Database connection pooling (PgBouncer)

### Production Environment

**High Availability Setup:**

```yaml
Load Balancer:
  - AWS ALB / Azure LB (for API)
  - CloudFront / CDN (for frontend)

Backend:
  - Minimum 3 replicas
  - Auto-scaling 3-10 instances
  - CPU-based scaling: 60-80% threshold

PostgreSQL:
  - Primary + standby replica
  - Automated backups (daily)
  - Point-in-time recovery enabled
  - 200GB SSD minimum
  - High availability mode (streaming replication)

Redis:
  - Cluster mode with 6 nodes (3 primary, 3 replica)
  - Persistence enabled (AOF + RDB)
  - Eviction policy: allkeys-lru

Qdrant:
  - Cluster mode (if available in plan)
  - Daily snapshots to cold storage
  - 100GB SSD minimum

Azure Blob Storage:
  - Production account credentials
  - Geo-redundancy enabled
  - Lifecycle policies for old documents

Monitoring:
  - Prometheus + Grafana
  - CloudWatch / Application Insights
  - ELK Stack for logging
  - Sentry for error tracking
```

**Configuration:**
```
DEBUG=False
ENVIRONMENT=production
SECRET_KEY=<strong-random-key>
ALLOWED_ORIGINS=https://yourdomain.com
AZURE_STORAGE_CONNECTION_STRING=<production-account>
OPENAI_API_KEY=<production-key>
POSTGRES_PASSWORD=<strong-password>
DATABASE_URL=<managed-service-url>
```

**Resource Requirements:**
- Backend: 4 CPU, 8GB RAM per instance (3+ replicas)
- PostgreSQL: 8+ CPU, 32GB+ RAM, 500GB+ SSD
- Redis: 4 CPU, 16GB RAM (cluster)
- Qdrant: 8 CPU, 32GB RAM, 500GB+ SSD
- Total: 25+ CPU, 88GB+ RAM, 1TB+ storage

**Backup & Disaster Recovery:**
```
PostgreSQL:
  - Full backup daily
  - Incremental backups every 6 hours
  - PITR enabled for 30 days
  - Backup location: Separate region

Qdrant:
  - Snapshots daily to cold storage
  - Can rebuild from PostgreSQL metadata

Blob Storage:
  - Geo-redundant replication
  - Versioning enabled
  - Retention policies per compliance

RPO: 6 hours
RTO: 2 hours
```

**Security Hardening:**
- SSL/TLS certificates for all endpoints
- Network policies/security groups
- Regular security audits
- API rate limiting per user
- DDoS protection
- Web Application Firewall (WAF)
- Encrypted database backups

### Performance Optimization

**Backend:**
```
# Connection pooling
SQLALCHEMY_POOL_SIZE=20
SQLALCHEMY_MAX_OVERFLOW=10
SQLALCHEMY_POOL_PRE_PING=True

# Query optimization
- Enable query result caching (24hrs)
- Use database indexes on frequently queried fields
- Batch embedding operations

# Async optimization
- Use async database operations
- Celery task optimization
- Connection pooling for external APIs
```

**Frontend:**
```
# Build optimization
- Next.js production build
- Code splitting enabled
- Image optimization
- Font subsetting

# Runtime optimization
- React Query caching (default 5min)
- Incremental Static Regeneration (ISR)
- Dynamic imports for large components
```

**Database:**
```
# Indexing strategy
CREATE INDEX idx_documents_case_id ON documents(case_id);
CREATE INDEX idx_embeddings_doc_id ON embeddings(document_id);
CREATE INDEX idx_queries_case_id ON queries(case_id);

# Partitioning
- Partition vectors table by document_id
- Archive old queries quarterly
```

---

## Dependency Management

### Lock Files

**Backend:**
- `requirements.txt` - Pinned Python versions
- `Pipfile.lock` (optional) - For Pipenv users

**Frontend:**
- `package-lock.json` - Npm lock file
- `yarn.lock` (alternative) - Yarn lock file

### Update Strategy

**Security Updates:**
```bash
# Backend
pip-audit
pip install --upgrade pip
pip install -U -r requirements.txt

# Frontend
npm audit
npm audit fix
npm update
```

**Minor/Patch Updates:**
```bash
# Quarterly for stable libraries
# Before major version updates

# Backend example:
pip install --upgrade fastapi uvicorn

# Frontend example:
npm update --save
```

**Major Version Updates:**
```bash
# Annually, with testing
# Breaking changes require code updates

# Example: FastAPI 0.x → 1.x
# - Run full test suite
# - Update deployment docs
# - Deploy to staging first
```

### Pinned Dependencies

**Critical (must pin exact):**
- FastAPI: 0.109.0
- SQLAlchemy: 2.0.23
- OpenAI: 1.12.0
- Pydantic: 2.5.3
- React: 18.2.0
- Next.js: 14.0.0
- TypeScript: 5.9.3

**Stable (can allow minor updates):**
- Celery, Redis clients
- HTTP clients (httpx, axios)
- UI libraries

**Flexible (auto-update safe):**
- Testing frameworks (pytest)
- Linters (ESLint)
- Dev dependencies

### Compatibility Notes

**Breaking Changes:**
1. **SQLAlchemy 2.0** - Dropped legacy query API
2. **Pydantic v2** - Field validators changed
3. **LangChain 0.1+** - API restructuring
4. **Next.js 13+** - App Router introduced

**Tested Combinations:**
```
✓ Python 3.11 + FastAPI 0.109.0 + SQLAlchemy 2.0.23
✓ Node 18 + Next.js 14.0.0 + React 18.2.0 + Tailwind 4.1.18
✓ PostgreSQL 16 + Qdrant 1.7.0 + Redis 7
✓ OpenAI API v1 + LangChain 0.1.20
```

---

## Summary

LexIntel's tech stack is designed for:
- **Performance:** Async processing, vector search, caching
- **Scalability:** Containerized architecture, load balancing ready
- **Maintainability:** Modern frameworks, type safety, clear separation of concerns
- **Security:** Industry-standard authentication, encrypted storage, role-based access
- **Cost:** Open-source infrastructure, serverless-ready, optimized API usage

All components are industry-proven and widely used in production systems.
