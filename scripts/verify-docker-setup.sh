#!/bin/bash
set -e

echo "========================================="
echo "Docker Compose Setup Verification"
echo "========================================="

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker."
    exit 1
fi

echo ""
echo "1️⃣  Starting services..."
docker-compose up -d

echo ""
echo "2️⃣  Waiting for services to become healthy (max 60s)..."
sleep 10

RETRY=0
MAX_RETRIES=10
while [ $RETRY -lt $MAX_RETRIES ]; do
    HEALTHY=$(docker-compose ps | grep -c "healthy" || true)
    if [ "$HEALTHY" -ge 5 ]; then  # 5 services should be healthy
        echo "✅ Services becoming healthy..."
        break
    fi
    RETRY=$((RETRY + 1))
    sleep 3
done

echo ""
echo "3️⃣  Checking service status..."
docker-compose ps

echo ""
echo "4️⃣  Verifying PostgreSQL..."
if docker-compose exec -T postgres pg_isready -U legal_user &> /dev/null; then
    echo "✅ PostgreSQL: READY"
else
    echo "❌ PostgreSQL: NOT READY"
    exit 1
fi

echo ""
echo "5️⃣  Verifying Qdrant..."
if curl -f http://localhost:6333/readyz &> /dev/null; then
    echo "✅ Qdrant: READY"
else
    echo "❌ Qdrant: NOT READY"
    exit 1
fi

echo ""
echo "6️⃣  Verifying Azurite..."
if curl -f http://localhost:10000/ &> /dev/null; then
    echo "✅ Azurite: READY"
else
    echo "❌ Azurite: NOT READY"
    exit 1
fi

echo ""
echo "7️⃣  Verifying Redis..."
if docker-compose exec -T redis redis-cli ping &> /dev/null; then
    echo "✅ Redis: READY"
else
    echo "❌ Redis: NOT READY"
    exit 1
fi

echo ""
echo "8️⃣  Running migrations..."
if docker-compose exec -T backend alembic upgrade head &> /dev/null; then
    echo "✅ Migrations: COMPLETE"
else
    echo "❌ Migrations: FAILED"
    exit 1
fi

echo ""
echo "9️⃣  Verifying FastAPI..."
sleep 5
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ FastAPI: READY"
else
    echo "❌ FastAPI: NOT READY"
    docker-compose logs backend | tail -20
    exit 1
fi

echo ""
echo "========================================="
echo "✨ All services verified successfully!"
echo "========================================="
echo ""
echo "🌐 Access points:"
echo "   - API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Frontend: http://localhost:3000"
echo "   - PostgreSQL: localhost:5432"
echo "   - Qdrant: http://localhost:6333"
echo "   - Redis: localhost:6379"
echo ""
echo "📝 View logs: docker-compose logs -f [service]"
echo "🛑 Stop: docker-compose down"
echo ""
