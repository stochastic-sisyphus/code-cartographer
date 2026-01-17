"""
Tests for FastAPI server and endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from code_cartographer.api.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestBasicEndpoints:
    """Test suite for basic API endpoints."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Code Warp House" in response.text
        assert "html" in response.text.lower()

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "code-warp-house"
        assert "version" in data

    def test_list_projects_empty(self, client):
        """Test listing projects when none exist."""
        response = client.get("/api/v1/projects")
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert isinstance(data["projects"], list)

    def test_get_nonexistent_project(self, client):
        """Test getting a project that doesn't exist."""
        response = client.get("/api/v1/projects/nonexistent123")
        assert response.status_code == 404

    def test_api_docs_available(self, client):
        """Test that API documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Code Warp House API"


class TestAnalysisEndpoints:
    """Test suite for analysis endpoints."""

    def test_analyze_invalid_path(self, client):
        """Test analyzing a non-existent path."""
        response = client.post(
            "/api/v1/projects/analyze",
            json={
                "project_path": "/nonexistent/path/that/does/not/exist"
            }
        )
        assert response.status_code == 404

    def test_analyze_requires_directory(self, client, tmp_path):
        """Test that analysis requires a directory, not a file."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        response = client.post(
            "/api/v1/projects/analyze",
            json={
                "project_path": str(test_file)
            }
        )
        assert response.status_code == 400


class TestTemporalEndpoints:
    """Test suite for temporal analysis endpoints."""

    def test_timeline_nonexistent_project(self, client):
        """Test getting timeline for non-existent project."""
        response = client.get("/api/v1/projects/nonexistent/timeline")
        assert response.status_code == 404

    def test_commits_nonexistent_project(self, client):
        """Test getting commits for non-existent project."""
        response = client.get("/api/v1/projects/nonexistent/commits")
        assert response.status_code == 404


@pytest.mark.skip(reason="WebSocket testing requires additional setup")
class TestWebSocket:
    """Test suite for WebSocket functionality."""

    def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/api/v1/ws/analysis/test_project") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert data["project_id"] == "test_project"
