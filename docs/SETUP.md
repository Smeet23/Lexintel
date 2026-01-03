# Development Setup & Deployment

> Getting LexIntel running locally

---

## 🚀 Quick Start (Docker)

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key (https://platform.openai.com/account/api-keys)

### 1. Clone Repository
```bash
git clone git@github.com-personalwork:Smeet23/Lexintel.git
cd Lexintel
```

### 2. Set Environment Variables
```bash
# Copy example file
cp backend/.env.example backend/.env

# Edit backend/.env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

### 3. Start Services
```bash
# Build and start all containers
docker-compose up -d

# Watch logs
docker-compose logs -f backend
```

### 4. Verify Services
```bash
# Check all containers running
docker-compose ps

# Test API health
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# Swagger UI
open http://localhost:8000/docs
```

### 5. Stop Services
```bash
docker-compose down
```

---

## 🐳 Docker Compose Services

### PostgreSQL (Port 5432)
```
Container: lex-intel-postgres
Database: lex_intel_dev
User: lex_user
Password: lex_password
```

**Connect**:
```bash
docker-compose exec postgres psql -U lex_user -d lex_intel_dev
```

### Redis (Port 6379)
```
Container: lex-intel-redis
Use: Job queue, caching
```

**Check Connection**:
```bash
docker-compose exec redis redis-cli ping
# Expected: PONG
```

### Azurite (Ports 10000-10002)
```
Container: lex-intel-azurite
Blob Storage: http://localhost:10000
Queue Storage: http://localhost:10001
Table Storage: http://localhost:10002
```

### FastAPI Backend (Port 8000)
```
Container: lex-intel-backend
API: http://localhost:8000
Docs: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
```

### Celery Worker
```
Container: lex-intel-celery-worker
Processes async tasks
```

---

## 💻 Local Development (Without Docker)

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis
- Node.js (optional, for dev tools)

### 1. Create Virtual Environment
```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup PostgreSQL
```bash
# Create database
createdb -U postgres lex_intel_dev

# Create user
psql -U postgres -c "CREATE USER lex_user WITH PASSWORD 'lex_password';"
psql -U postgres -c "ALTER ROLE lex_user WITH CREATEDB;"
psql -U postgres -c "ALTER DATABASE lex_intel_dev OWNER TO lex_user;"

# Enable extensions
psql -U lex_user -d lex_intel_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -U lex_user -d lex_intel_dev -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

### 4. Environment Variables
```bash
# Create .env file in backend/
cat > backend/.env << 'EOF'
DATABASE_URL=postgresql://lex_user:lex_password@localhost:5432/lex_intel_dev
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-your-key-here
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:10000/devstoreaccount1;
DEBUG=True
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE=104857600
ALLOWED_EXTENSIONS=.pdf,.docx,.txt,.doc,.pptx
CHUNK_SIZE=4000
CHUNK_OVERLAP=400
EOF
```

### 5. Start Services

**Terminal 1: FastAPI Backend**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**Terminal 2: Celery Worker**
```bash
cd backend
celery -A app.celery_app worker -l info
```

**Terminal 3: Redis** (if not running as service)
```bash
redis-server
```

---

## 📋 Environment Variables

### Required
```
OPENAI_API_KEY=sk-...  # Get from https://platform.openai.com
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### Optional
```
DEBUG=True
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE=104857600
ALLOWED_EXTENSIONS=.pdf,.docx,.txt
CHUNK_SIZE=4000
CHUNK_OVERLAP=400
```

---

## 🧪 Testing

### Run All Tests
```bash
cd backend
pytest tests/
```

### Run Specific Test
```bash
pytest tests/unit/test_cases.py::test_create_case
```

### With Coverage
```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

### Test Async Code
```python
# pytest.ini or pyproject.toml includes asyncio_mode = "auto"
# Just use async/await in tests:

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

---

## 📊 Code Quality

### Format with Black
```bash
black backend/
```

### Type Check with mypy
```bash
mypy backend/
```

### Lint with flake8
```bash
flake8 backend/
```

### All Together
```bash
black backend/ && mypy backend/ && flake8 backend/ && pytest backend/tests/
```

---

## 🔍 Debugging

### View Logs
```bash
# Backend logs
docker-compose logs -f backend

# Worker logs
docker-compose logs -f celery-worker

# Database logs
docker-compose logs -f postgres

# Redis logs
docker-compose logs -f redis
```

### Database Connection
```bash
# From within Docker
docker-compose exec postgres psql -U lex_user -d lex_intel_dev

# From local (if PostgreSQL running locally)
psql -U lex_user -d lex_intel_dev
```

### Check Task Queue
```bash
# Celery status
celery -A app.celery_app inspect active

# Active tasks
celery -A app.celery_app inspect active_queues

# Worker stats
celery -A app.celery_app inspect stats
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Create case
curl -X POST http://localhost:8000/cases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Case",
    "case_number": "2024-001",
    "practice_area": "contracts",
    "status": "active",
    "description": "Test"
  }'

# Upload document
curl -X POST "http://localhost:8000/documents/upload?case_id=YOUR_CASE_ID" \
  -F "file=@/path/to/document.pdf"

# List documents
curl http://localhost:8000/documents
```

---

## 🐛 Common Issues

### Issue: Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or change port in docker-compose.yml or uvicorn command
uvicorn app.main:app --port 8001
```

### Issue: Database Connection Failed
```bash
# Check PostgreSQL is running
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres

# Wait for health check
docker-compose exec postgres pg_isready -U lex_user
```

### Issue: Celery Tasks Not Processing
```bash
# Check Redis is running
docker-compose logs redis

# Check worker is listening
docker-compose logs celery-worker

# Restart worker
docker-compose restart celery-worker
```

### Issue: OpenAI API Errors
```bash
# Check API key in .env
docker-compose exec backend env | grep OPENAI

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer sk-your-key-here"
```

### Issue: File Upload Not Working
```bash
# Check upload directory exists
docker-compose exec backend ls -la /app/uploads

# Check permissions
docker-compose exec backend stat /app/uploads

# Increase max file size in .env
MAX_UPLOAD_SIZE=104857600  # 100MB
```

---

## 🚀 Production Deployment

### Use Uvicorn with Gunicorn
```bash
pip install gunicorn
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Use Cloud Deployment
```bash
# Heroku
git push heroku main

# AWS ECS
aws ecs deploy-service ...

# Google Cloud Run
gcloud run deploy lexintel ...

# Azure Container Instances
az container create ...
```

### Environment Variables
```bash
# Production should have secure secrets management
# Use environment-specific .env files
export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql://prod-user:pass@prod-db:5432/lex_intel
export REDIS_URL=redis://prod-redis:6379
export DEBUG=False
export ENVIRONMENT=production
```

### Database Migrations
```bash
# When using Alembic (future):
alembic upgrade head
```

---

## 📝 Pre-Commit Hooks

### Setup (Optional)
```bash
pip install pre-commit
pre-commit install
```

### Create .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
```

---

## 🔐 Security Checklist

- [ ] Never commit `.env` file
- [ ] Use environment variables for secrets
- [ ] OpenAI API key from secure source
- [ ] Database password strong and unique
- [ ] Set `DEBUG=False` in production
- [ ] Use HTTPS in production
- [ ] Configure CORS properly for production
- [ ] Validate all user input
- [ ] Log security events

---

## 📚 References

- **FastAPI Deploy**: https://fastapi.tiangolo.com/deployment/
- **PostgreSQL Setup**: https://www.postgresql.org/docs/
- **Redis Setup**: https://redis.io/docs/getting-started/
- **Docker Compose**: https://docs.docker.com/compose/
