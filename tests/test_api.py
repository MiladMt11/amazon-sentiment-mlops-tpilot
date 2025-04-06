import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

# -------------------------------
# Core Functionality Tests
# -------------------------------

def test_predict_valid_request():
    """
    Should return 200 OK and a valid prediction for normal input.
    """
    payload = {"text": "I loved this book. It was amazing!"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in ["positive", "neutral", "negative"]
    assert "latency_ms" in data
    assert "input_truncated" in data

# -------------------------------
# Edge Case Tests
# -------------------------------

def test_predict_missing_text_field():
    """
    Should return 422 Unprocessable Entity if 'text' field is missing.
    """
    payload = {}  # No "text" key at all
    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_empty_string_input():
    """
    Should return 400 Bad Request if text is empty.
    """
    payload = {"text": ""}
    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Input text must not be empty"

def test_predict_non_string_input():
    """
    Should return 422 if 'text' is not a string (e.g., a number).
    """
    payload = {"text": 123}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_very_long_input_truncation_warning():
    """
    Should still return 200 but with a truncation warning if input exceeds token limit.
    """
    long_text = "This is a test sentence. " * 1000
    payload = {"text": long_text}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in ["positive", "neutral", "negative"]
    assert data.get("input_truncated") is True

