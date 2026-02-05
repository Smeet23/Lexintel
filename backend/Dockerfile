# Multi-stage build
FROM python:3.11-slim as builder

WORKDIR /tmp

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /tmp/lexintel_storage && \
    chown -R appuser:appuser /app /tmp/lexintel_storage

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy application code (keeping backend folder structure)
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set PYTHONPATH for module imports
ENV PYTHONPATH=/app

# Run migrations and start app (run alembic from /app since alembic.ini uses backend/alembic path)
CMD ["sh", "-c", "cd /app && python -m alembic -c backend/alembic.ini upgrade head && uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
