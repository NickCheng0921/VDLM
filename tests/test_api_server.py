"""
Unit tests for the VDLM API server completions endpoint.
Run api_server before running 'pytest'
"""

from fastapi.testclient import TestClient
from api_server import app


def test_read_main_completions():
    with TestClient(app) as client:
        payload = {
            "model": "vdlm-v1",
            "prompt": "Hello world",
            "max_tokens": 50,
            "temperature": 0.7,
        }
        response = client.post("/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "vdlm-v1"
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]


def test_validation_error():
    with TestClient(app) as client:
        # Missing required 'model' and 'prompt'
        response = client.post("/completions", json={})
        assert response.status_code == 422
