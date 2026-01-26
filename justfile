#!/usr/bin/env just --justfile

# LexIntel Development Commands
# Usage: just <recipe>
# List recipes: just --list

set shell := ["bash", "-c"]

# Default recipe - show help
default:
    @just --list

# ============================================
# QUICK START
# ============================================

# First time setup (run this first!)
first-time:
    @echo "🚀 LexIntel First-Time Setup"
    @echo "=============================="
    @echo ""
    @just setup
    @just env
    @echo ""
    @echo "✅ Setup complete!"
    @echo ""
    @echo "Next steps:"
    @echo "1️⃣  Edit backend/.env with your API keys:"
    @echo "   - OPENAI_API_KEY"
    @echo "   - AZURE_STORAGE_CONNECTION_STRING"
    @echo ""
    @echo "2️⃣  Start services:"
    @echo "   just dev"
    @echo ""

# ============================================
# SETUP & INSTALLATION
# ============================================

# Install all dependencies (backend + frontend)
setup:
    #!/bin/bash
    set -e

    echo "🔧 Installing backend dependencies..."
    if [ ! -d "backend/venv" ]; then
        echo "   Creating Python virtual environment..."
        python3 -m venv backend/venv
    fi

    source backend/venv/bin/activate
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    pip install -r backend/requirements.txt
    deactivate
    echo "✓ Backend dependencies installed"
    echo ""

    echo "🔧 Installing frontend dependencies..."
    cd frontend
    npm install --legacy-peer-deps 2>/dev/null || npm install
    cd ..
    echo "✓ Frontend dependencies installed"
    echo ""
    echo "✅ Dependencies ready!"

# Setup environment files
env:
    #!/bin/bash

    echo "📝 Setting up environment files..."

    # Backend .env
    if [ ! -f backend/.env ]; then
        echo "   Creating backend/.env..."
        cp backend/.env.example backend/.env
        echo "   ⚠️  IMPORTANT: Edit backend/.env with your API keys"
    else
        echo "   ✓ backend/.env already exists"
    fi

    # Frontend .env.local
    if [ ! -f frontend/.env.local ]; then
        echo "   Creating frontend/.env.local..."
        echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > frontend/.env.local
        echo "   ✓ frontend/.env.local created"
    else
        echo "   ✓ frontend/.env.local already exists"
    fi
    echo ""

# Initialize database
db-init:
    #!/bin/bash
    echo "🗄️  Initializing database..."
    source backend/venv/bin/activate
    cd backend
    alembic upgrade head
    deactivate
    echo "✓ Database initialized"

# ============================================
# DOCKER SETUP
# ============================================

# Start all services with Docker Compose
docker-up:
    @echo "🐳 Starting all services with Docker Compose..."
    docker-compose up -d
    @echo ""
    @echo "✓ Services started:"
    @echo "   • PostgreSQL:  localhost:5432"
    @echo "   • Qdrant:      localhost:6333"
    @echo "   • Backend API: http://localhost:8000"
    @echo ""
    @echo "Next: just dev"

# Stop all Docker services
docker-down:
    @echo "🛑 Stopping all services..."
    docker-compose down
    @echo "✓ Services stopped"

# ============================================
# DEVELOPMENT - ALL IN ONE
# ============================================

# START EVERYTHING: Backend, frontend, jobs - one command!
dev:
    #!/bin/bash
    set -e

    # Check dependencies
    if [ ! -d "backend/venv" ]; then
        echo "❌ Backend venv not found"
        echo ""
        echo "Run setup first:"
        echo "  just setup"
        exit 1
    fi

    if [ ! -d "frontend/node_modules" ]; then
        echo "❌ Frontend node_modules not found"
        echo ""
        echo "Run setup first:"
        echo "  just setup"
        exit 1
    fi

    echo "🚀 Starting LexIntel development environment..."
    echo ""
    echo "📡 Backend:      http://localhost:8000"
    echo "📖 API Docs:     http://localhost:8000/docs"
    echo "⚛️  Frontend:     http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""

    # Start services in background
    (
        echo "Starting backend..."
        source backend/venv/bin/activate
        cd backend
        uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
    ) &
    BACKEND_PID=$!

    sleep 3

    (
        echo "Starting frontend..."
        cd frontend
        npm run dev
    ) &
    FRONTEND_PID=$!

    sleep 3

    (
        echo "Starting job processor..."
        source backend/venv/bin/activate
        python -m backend.services.job_processor run_worker
    ) &
    JOBS_PID=$!

    echo ""
    echo "✅ All services started!"
    echo ""

    # Wait for all processes
    wait

# ============================================
# BACKEND COMMANDS
# ============================================

# Start backend server only
backend-start:
    #!/bin/bash
    if [ ! -d "backend/venv" ]; then
        echo "❌ Run 'just setup' first"
        exit 1
    fi
    echo "📡 Starting backend on http://localhost:8000"
    echo "📖 API docs: http://localhost:8000/docs"
    echo ""
    source backend/venv/bin/activate
    cd backend
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start background job processor only
jobs-start:
    #!/bin/bash
    if [ ! -d "backend/venv" ]; then
        echo "❌ Run 'just setup' first"
        exit 1
    fi
    echo "⚙️  Starting background job processor..."
    source backend/venv/bin/activate
    python -m backend.services.job_processor run_worker

# Run backend tests
backend-test:
    #!/bin/bash
    echo "🧪 Running backend tests..."
    source backend/venv/bin/activate
    cd backend
    pytest -v

# Backend console
backend-shell:
    #!/bin/bash
    echo "🐍 Starting Python shell with backend context..."
    source backend/venv/bin/activate
    cd backend
    python

# View backend API documentation
backend-docs:
    @echo "📖 Opening http://localhost:8000/docs..."
    @open http://localhost:8000/docs 2>/dev/null || xdg-open http://localhost:8000/docs 2>/dev/null || echo "Visit: http://localhost:8000/docs"

# Check backend health
backend-health:
    #!/bin/bash
    echo "🏥 Checking backend health..."
    curl -s http://localhost:8000/health 2>/dev/null | python3 -m json.tool || echo "❌ Backend not responding"

# ============================================
# FRONTEND COMMANDS
# ============================================

# Start frontend dev server only
frontend-dev:
    #!/bin/bash
    echo "⚛️  Starting frontend..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "Installing dependencies..."
        npm install --legacy-peer-deps
    fi
    npm run dev

# Alias for frontend-dev
frontend-start: frontend-dev

# Build frontend for production
frontend-build:
    @echo "🏗️  Building frontend for production..."
    cd frontend && npm run build
    @echo "✓ Build complete"

# Run frontend linter
frontend-lint:
    @echo "📝 Linting frontend..."
    cd frontend && npm run lint

# Frontend type check
frontend-types:
    @echo "🔍 Checking TypeScript types..."
    cd frontend && npx tsc --noEmit

# ============================================
# UTILITIES
# ============================================

# Check if all services are running
status:
    #!/bin/bash
    echo "📊 Service Status:"
    echo ""
    echo -n "Backend (localhost:8000):  "
    curl -s http://localhost:8000/health >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"
    echo -n "Frontend (localhost:3000): "
    curl -s http://localhost:3000 >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"
    echo -n "Qdrant (localhost:6333):   "
    curl -s http://localhost:6333/health >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"

# Stop all running services
stop:
    #!/bin/bash
    echo "🛑 Stopping all services..."
    pkill -f "uvicorn" || true
    pkill -f "npm run dev" || true
    pkill -f "python -m backend" || true
    sleep 1
    echo "✓ All services stopped"

# Clean build artifacts
clean:
    #!/bin/bash
    echo "🧹 Cleaning build artifacts..."
    rm -rf frontend/.next 2>/dev/null || true
    rm -rf frontend/node_modules/.cache 2>/dev/null || true
    rm -rf backend/__pycache__ 2>/dev/null || true
    rm -rf backend/.pytest_cache 2>/dev/null || true
    echo "✓ Cleaned"

# Reset frontend (reinstall dependencies)
frontend-reset:
    #!/bin/bash
    echo "🔄 Resetting frontend..."
    cd frontend
    rm -rf node_modules package-lock.json
    npm install --legacy-peer-deps 2>/dev/null || npm install
    echo "✓ Frontend reset"

# Reset backend (reinstall dependencies)
backend-reset:
    #!/bin/bash
    echo "🔄 Resetting backend..."
    rm -rf backend/venv
    python3 -m venv backend/venv
    source backend/venv/bin/activate
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    pip install -r backend/requirements.txt
    deactivate
    echo "✓ Backend reset"

# Full reset and setup again
reset: stop clean backend-reset frontend-reset setup env
    @echo "✓ Full reset complete"
    @echo ""
    @echo "To start again:"
    @echo "  just dev"

# Delete and recreate database (careful!)
db-reset:
    #!/bin/bash
    echo "⚠️  DATABASE RESET - This will delete all data"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v
        sleep 2
        docker-compose up -d
        sleep 3
        just db-init
        echo "✓ Database reset complete"
    else
        echo "Reset cancelled"
    fi

# ============================================
# ALIASES
# ============================================

# ============================================
# ONE COMMAND TO RUN EVERYTHING
# ============================================

# Backend + Databases in Docker (frontend separate)
up-d:
    #!/bin/bash

    echo "🚀 Starting Backend & Database Services..."
    echo ""

    # Check if Docker is running
    if ! docker ps > /dev/null 2>&1; then
        echo "❌ Docker is not running!"
        echo ""
        echo "Start Docker Desktop:"
        echo "  open /Applications/Docker.app"
        exit 1
    fi

    echo "🐳 Starting Docker services..."
    docker-compose up -d postgres qdrant azurite redis backend celery-worker 2>&1 | grep -v "WARN.*obsolete" || true
    sleep 3

    echo ""
    echo "============================================"
    echo "✅ Backend services started!"
    echo "============================================"
    echo ""
    echo "Services available at:"
    echo "  • Backend:    http://localhost:8000"
    echo "  • API Docs:   http://localhost:8000/docs"
    echo "  • PostgreSQL: localhost:5432"
    echo "  • Qdrant:     localhost:6333"
    echo ""
    echo "To start frontend in another terminal:"
    echo "  just frontend-dev"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f"
    echo ""
    echo "Press Ctrl+C to stop backend services"
    echo ""

    # Keep running
    docker-compose logs -f

# Alias: just start = just up-d
start: up-d

# Alias: just run = just up-d
run: up-d

# Alias: just dev-all = just up-d
dev-all: up-d
