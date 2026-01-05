# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Cheng

"""
Tests for the VDLM API server completions endpoint.

Runs engine in mock mode
"""

from fastapi.testclient import TestClient
from api_server import app, engine

engine.is_mock = True


def test_read_main_completions():
    with TestClient(app) as client:
        # Wait for mock engine to be ready after 0.5s sleep
        import time

        for _ in range(20):
            if engine.ready_event.is_set():
                break
            time.sleep(0.1)

        payload = {
            "model": "vdlm-v1",
            "prompt": "Hello world",
            "max_tokens": 64,
            "temperature": 0.7,
        }
        response = client.post("/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "vdlm-v1"
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]


def test_model_loading_503():
    """Test that we get a 503 if the model is not ready."""
    with TestClient(app) as client:
        # Manually clear the event to simulate loading state
        engine.ready_event.clear()

        payload = {
            "model": "vdlm-v1",
            "prompt": "Test",
        }
        response = client.post("/completions", json=payload)
        assert response.status_code == 503
        assert (
            response.json()["detail"]
            == "Model is still loading. Please try again later."
        )

        # Reset event for other tests (though client teardown might handle it, better safe)
        engine.ready_event.set()


def test_validation_block_size_error():
    """Test that requests with max_tokens not divisible by block_length return an error."""
    with TestClient(app) as client:
        # Wait for mock engine
        import time

        for _ in range(20):
            if engine.ready_event.is_set():
                break
            time.sleep(0.1)

        payload = {
            "model": "vdlm-v1",
            "prompt": "Test",
            "max_tokens": 50,  # Not divisible by default block_length 32
        }
        response = client.post("/completions", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "divisible by block_length" in data["detail"]


def test_validation_custom_block_length_success():
    """Test that max_tokens=16 works if we set block_length=16."""
    with TestClient(app) as client:
        # Wait for mock engine
        import time

        for _ in range(20):
            if engine.ready_event.is_set():
                break
            time.sleep(0.1)

        payload = {
            "model": "vdlm-v1",
            "prompt": "Test",
            "max_tokens": 16,
            "block_length": 16,
            "steps": 16,  # 16 % (16/16) == 0 => OK
        }
        response = client.post("/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) > 0


def test_validation_steps_error():
    """Test that steps must be divisible by num_blocks."""
    with TestClient(app) as client:
        # Wait for mock engine
        import time

        for _ in range(20):
            if engine.ready_event.is_set():
                break
            time.sleep(0.1)

        payload = {
            "model": "vdlm-v1",
            "prompt": "Test",
            "max_tokens": 64,
            "block_length": 32,  # num_blocks = 2
            "steps": 3,  # 3 % 2 != 0 => Error
        }
        response = client.post("/completions", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "steps" in data["detail"] and "num_blocks" in data["detail"]


def test_validation_error():
    with TestClient(app) as client:
        # Missing required 'model' and 'prompt'
        response = client.post("/completions", json={})
        assert response.status_code == 422
