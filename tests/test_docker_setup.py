"""Test Docker Compose setup and service connectivity"""
import os
import pytest
import httpx
import psycopg2
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from azure.storage.blob import BlobServiceClient


class TestDockerServices:
    """Test all Docker services are accessible and healthy"""

    def test_postgres_connection(self):
        """PostgreSQL is accessible and initialized"""
        # This test requires PostgreSQL running at localhost:5432
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://legal_user:dev_password_change_in_prod@localhost:5432/legal_rag"
        )

        # Should connect without error
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1

        engine.dispose()

    def test_postgres_has_tables(self):
        """PostgreSQL has all required tables"""
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://legal_user:dev_password_change_in_prod@localhost:5432/legal_rag"
        )

        engine = create_engine(db_url)

        with engine.connect() as conn:
            # List all tables
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            ))
            tables = {row[0] for row in result}

        # All required tables should exist
        required_tables = {"users", "cases", "chunks", "queries"}
        assert required_tables.issubset(tables), f"Missing tables. Got: {tables}"

        engine.dispose()

    def test_qdrant_connection(self):
        """Qdrant is accessible and ready"""
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        # Should connect and get collections without error
        client = QdrantClient(qdrant_url)
        collections = client.get_collections()
        assert collections is not None

    def test_azurite_connection(self):
        """Azurite blob storage is accessible"""
        conn_str = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING",
            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXOU+FIH3Iv5/5M=;BlobEndpoint=http://localhost:10000/devstoreaccount1;"
        )

        # Should connect and get account info without error
        client = BlobServiceClient.from_connection_string(conn_str)
        account_info = client.get_account_information()
        assert account_info is not None

    def test_backend_health_endpoint(self):
        """FastAPI backend health endpoint responds"""
        # This test requires FastAPI running at localhost:8000
        response = httpx.get("http://localhost:8000/health", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_docker_compose_file_exists(self):
        """docker-compose.yml exists"""
        assert os.path.exists("docker-compose.yml")

    def test_backend_dockerfile_exists(self):
        """Dockerfile exists for backend"""
        assert os.path.exists("backend/Dockerfile")

    def test_verify_script_exists(self):
        """Verification script exists and is executable"""
        script_path = "scripts/verify-docker-setup.sh"
        assert os.path.exists(script_path)
        assert os.access(script_path, os.X_OK)
