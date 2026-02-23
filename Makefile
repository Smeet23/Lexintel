.PHONY: help setup setup-backend setup-frontend env \
       dev backend frontend celery \
       docker-up docker-down docker-logs \
       db-init db-migrate db-reset \
       test test-backend lint \
       status stop clean reset install

# Default
help:
	@echo "Veritas AI (Lexintel) - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Install all dependencies (backend + frontend)"
	@echo "  make setup-backend  Install backend Python deps only"
	@echo "  make setup-frontend Install frontend Node deps only"
	@echo "  make env            Create .env files from examples"
	@echo ""
	@echo "Docker (infrastructure):"
	@echo "  make docker-up      Start PostgreSQL, Qdrant, Redis, Azurite"
	@echo "  make docker-down    Stop all Docker services"
	@echo "  make docker-logs    Tail Docker service logs"
	@echo ""
	@echo "Development:"
	@echo "  make dev            Start backend + frontend + celery (all-in-one)"
	@echo "  make backend        Start FastAPI backend only (port 8000)"
	@echo "  make frontend       Start Next.js frontend only (port 3000)"
	@echo "  make celery         Start Celery worker only"
	@echo ""
	@echo "Database:"
	@echo "  make db-init        Run alembic migrations"
	@echo "  make db-migrate     Auto-generate new migration"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all backend tests"
	@echo "  make lint           Run frontend lint + type check"
	@echo ""
	@echo "Utilities:"
	@echo "  make status         Check which services are running"
	@echo "  make stop           Kill all dev processes"
	@echo "  make clean          Remove build artifacts"
	@echo "  make reset          Full reset (clean + reinstall)"

# ============================================
# SETUP
# ============================================

setup: setup-backend setup-frontend env
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. Edit backend/.env with your GOOGLE_API_KEY"
	@echo "  2. make docker-up"
	@echo "  3. make db-init"
	@echo "  4. make dev"

setup-backend:
	@echo "Installing backend dependencies..."
	@test -d backend/venv || python3 -m venv backend/venv
	@backend/venv/bin/pip install --upgrade pip setuptools wheel -q
	@backend/venv/bin/pip install -r backend/requirements.txt
	@echo "Backend ready."

setup-frontend:
	@echo "Installing frontend dependencies..."
	@cd frontend && npm install --legacy-peer-deps 2>/dev/null || cd frontend && npm install
	@echo "Frontend ready."

install: setup

env:
	@test -f backend/.env || (cp backend/.env.example backend/.env && echo "Created backend/.env — edit with your API keys")
	@test -f backend/.env && echo "backend/.env exists"
	@test -f frontend/.env || (test -f frontend/.env.example && cp frontend/.env.example frontend/.env && echo "Created frontend/.env") || true
	@test -f frontend/.env && echo "frontend/.env exists"

# ============================================
# DOCKER (infrastructure services)
# ============================================

docker-up:
	@echo "Starting infrastructure services..."
	@docker compose up -d postgres qdrant azurite redis
	@echo ""
	@echo "Services:"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Qdrant:     localhost:6333"
	@echo "  Redis:      localhost:6379"
	@echo "  Azurite:    localhost:10000"

docker-down:
	@docker compose down

docker-logs:
	@docker compose logs -f

# ============================================
# DEVELOPMENT
# ============================================

backend:
	@echo "Starting backend on http://localhost:8000"
	@echo "API docs: http://localhost:8000/docs"
	@cd backend && ../backend/venv/bin/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	@echo "Starting frontend on http://localhost:3000"
	@cd frontend && npm run dev

celery:
	@echo "Starting Celery worker..."
	@cd backend && ../backend/venv/bin/celery -A backend.celery_app worker -l info

dev:
	@echo "Starting all services... (Ctrl+C to stop)"
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo ""
	@$(MAKE) -j3 backend frontend celery

# ============================================
# DATABASE
# ============================================

db-init:
	@echo "Running migrations..."
	@backend/venv/bin/alembic upgrade head
	@echo "Database up to date."

db-migrate:
	@echo "Generating migration..."
	@backend/venv/bin/alembic revision --autogenerate -m "$(msg)"

db-reset:
	@echo "Resetting database (destructive)..."
	@docker compose down -v
	@docker compose up -d postgres
	@sleep 3
	@$(MAKE) db-init

# ============================================
# TESTING
# ============================================

test:
	@echo "Running backend tests..."
	@cd backend && ../backend/venv/bin/pytest -v $(args)

test-backend: test

lint:
	@echo "Linting frontend..."
	@cd frontend && npm run lint

# ============================================
# UTILITIES
# ============================================

status:
	@echo "Service Status:"
	@printf "  Backend (8000):  " && (curl -sf http://localhost:8000/health > /dev/null && echo "UP" || echo "DOWN")
	@printf "  Frontend (3000): " && (curl -sf http://localhost:3000 > /dev/null && echo "UP" || echo "DOWN")
	@printf "  Qdrant (6333):   " && (curl -sf http://localhost:6333/readyz > /dev/null && echo "UP" || echo "DOWN")
	@printf "  Redis (6379):    " && (docker exec lexintel_redis redis-cli ping 2>/dev/null | grep -q PONG && echo "UP" || echo "DOWN")
	@printf "  Postgres (5432): " && (docker exec lexintel_postgres pg_isready 2>/dev/null | grep -q "accepting" && echo "UP" || echo "DOWN")

stop:
	@echo "Stopping dev processes..."
	@-pkill -f "uvicorn backend" 2>/dev/null || true
	@-pkill -f "next-router-worker" 2>/dev/null || true
	@-pkill -f "celery.*backend" 2>/dev/null || true
	@echo "Stopped."

clean:
	@rm -rf frontend/.next frontend/node_modules/.cache
	@find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf backend/.pytest_cache
	@echo "Cleaned."

reset: stop clean
	@$(MAKE) setup
	@echo "Reset complete. Run: make dev"
