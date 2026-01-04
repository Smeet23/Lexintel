# LexIntel Service Management
# Commands to manage all services (PostgreSQL, Redis, Azurite, Backend, Workers)

set shell := ["bash", "-c"]

# Display all available commands
help:
    @just --list

# ============================================================================
# SERVICE MANAGEMENT
# ============================================================================

# Start all services (Docker Compose)
up:
    docker-compose up -d
    @echo "✓ All services starting..."
    @just health-check

# Start all services with logs
up-logs:
    docker-compose up

# Stop all services gracefully
down:
    docker-compose down
    @echo "✓ All services stopped"

# Stop and remove volumes (clean slate)
down-clean:
    docker-compose down -v
    @echo "✓ All services stopped and volumes removed"

# Restart all services
restart:
    docker-compose restart
    @echo "✓ All services restarted"

# Restart a specific service
restart-service service:
    docker-compose restart {{service}}
    @echo "✓ {{service}} restarted"

# ============================================================================
# STATUS & HEALTH CHECKS
# ============================================================================

# Check health of all services
health-check:
    @echo "Checking service health..."
    @docker-compose ps
    @echo ""
    @echo "Testing API health..."
    @curl -s http://localhost:8000/health | jq . || echo "API not ready yet"

# Show status of all containers
status:
    docker-compose ps

# ============================================================================
# LOGS & DEBUGGING
# ============================================================================

# View logs for all services
logs:
    docker-compose logs -f

# View logs for a specific service
logs-service service:
    docker-compose logs -f {{service}}

# View recent logs (last 50 lines)
logs-recent:
    docker-compose logs --tail=50

# View backend logs
logs-backend:
    docker-compose logs -f backend

# View worker logs
logs-worker:
    docker-compose logs -f workers

# View database logs
logs-db:
    docker-compose logs -f postgres

# View redis logs
logs-redis:
    docker-compose logs -f redis

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

# Connect to PostgreSQL
db-connect:
    docker-compose exec postgres psql -U lex_user -d lex_intel_dev

# Show database info
db-info:
    docker-compose exec postgres psql -U lex_user -d lex_intel_dev -c "\l"

# Reset database (drop and recreate)
db-reset:
    @echo "Resetting database..."
    docker-compose down -v
    docker-compose up -d postgres
    @echo "Waiting for PostgreSQL to be ready..."
    @sleep 5
    @echo "✓ Database reset. Start other services with: just up"

# Run database migrations (if using Alembic in future)
db-migrate:
    docker-compose exec backend alembic upgrade head

# ============================================================================
# REDIS OPERATIONS
# ============================================================================

# Check Redis connection
redis-ping:
    docker-compose exec redis redis-cli ping

# Redis CLI
redis-cli:
    docker-compose exec redis redis-cli

# Flush Redis cache (be careful!)
redis-flush:
    @echo "Flushing all Redis data..."
    docker-compose exec redis redis-cli FLUSHALL
    @echo "✓ Redis cleared"

# ============================================================================
# TESTING
# ============================================================================

# Run all tests
test:
    docker-compose exec backend pytest tests/

# Run tests with coverage
test-coverage:
    docker-compose exec backend pytest tests/ --cov=app --cov-report=html
    @echo "✓ Coverage report generated at htmlcov/index.html"

# Run specific test file
test-file file:
    docker-compose exec backend pytest {{file}} -v

# Run tests matching pattern
test-pattern pattern:
    docker-compose exec backend pytest -k {{pattern}} -v

# ============================================================================
# CODE QUALITY
# ============================================================================

# Format code with Black
format:
    docker-compose exec backend black app/ tests/

# Type check with mypy
type-check:
    docker-compose exec backend mypy app/

# Lint with flake8
lint:
    docker-compose exec backend flake8 app/ tests/

# Run all checks (format, lint, type-check)
quality:
    just format && just lint && just type-check
    @echo "✓ All quality checks passed"

# ============================================================================
# BUILD & REBUILD
# ============================================================================

# Build all images
build:
    docker-compose build
    @echo "✓ All images built"

# Rebuild backend image
build-backend:
    docker-compose build backend
    @echo "✓ Backend image rebuilt"

# Rebuild worker image
build-worker:
    docker-compose build workers
    @echo "✓ Worker image rebuilt"

# Rebuild and start all services
rebuild:
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    @just health-check

# ============================================================================
# API TESTING
# ============================================================================

# Test API health endpoint
api-health:
    @curl -s http://localhost:8000/health | jq .

# Open API docs in browser
api-docs:
    @echo "Opening API docs at http://localhost:8000/docs"
    open http://localhost:8000/docs || xdg-open http://localhost:8000/docs || true

# List all cases
api-cases:
    @curl -s http://localhost:8000/cases | jq .

# ============================================================================
# ENVIRONMENT & SETUP
# ============================================================================

# Setup environment file from example
setup-env:
    @if [ ! -f apps/backend/.env ]; then \
        cp apps/backend/.env.example apps/backend/.env; \
        echo "✓ Created .env file. Edit apps/backend/.env with your OpenAI API key"; \
    else \
        echo ".env already exists"; \
    fi

# Show environment variables (sanitized)
show-env:
    @echo "Configuration:"
    @grep -E "^[^#]" apps/backend/.env 2>/dev/null | sed 's/\(.*=\).*/\1***/' || echo "No .env file found"

# ============================================================================
# UTILITIES
# ============================================================================

# Clean up Docker (remove unused images, containers, networks)
docker-clean:
    docker system prune -f
    @echo "✓ Docker cleaned up"

# Full cleanup (use with caution!)
clean-all:
    @echo "Cleaning up everything..."
    docker-compose down -v
    docker system prune -f
    @echo "✓ Everything cleaned up"

# Show all container IDs and names
docker-containers:
    docker-compose ps -a

# Export service environment variables for debugging
export-env service:
    docker-compose exec {{service}} env | grep -E "DATABASE|REDIS|AZURE|OPENAI" | sort
