#!/bin/bash

# Migration script: file_path → blob_url
# Handles database migration and service rebuild

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  LexIntel Migration: file_path → blob_url (Presigned URLs)     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Stop services
echo -e "${YELLOW}[1/4]${NC} Stopping services..."
docker-compose down
echo -e "${GREEN}✅ Services stopped${NC}"
echo ""

# Step 2: Start database only
echo -e "${YELLOW}[2/4]${NC} Starting PostgreSQL..."
docker-compose up -d postgres
sleep 3
echo -e "${GREEN}✅ PostgreSQL started${NC}"
echo ""

# Step 3: Run migration
echo -e "${YELLOW}[3/4]${NC} Running database migration..."
docker-compose run --rm backend python migrate_file_path_to_blob_url.py
MIGRATION_EXIT=$?

if [ $MIGRATION_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Migration failed!${NC}"
    docker-compose down
    exit 1
fi

echo -e "${GREEN}✅ Migration completed${NC}"
echo ""

# Step 4: Rebuild and restart all services
echo -e "${YELLOW}[4/4]${NC} Rebuilding and starting all services..."
docker-compose build
docker-compose up -d
echo -e "${GREEN}✅ Services rebuilt and started${NC}"
echo ""

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Verify
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   MIGRATION SUCCESSFUL ✅                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Test extraction: Upload a document via API"
echo "  2. Monitor: just logs-worker | grep extract"
echo "  3. Check database: just db-connect"
echo ""
echo "Commands:"
echo "  just up              - Start all services"
echo "  just logs-worker     - View worker logs"
echo "  just db-connect      - Connect to PostgreSQL"
echo "  curl http://localhost:8000/health  - Test API"
echo ""
