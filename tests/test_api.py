import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

# -------------------------------
# ✅ Core Functionality Tests
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

# -------------------------------
# 🧪 Edge Case Tests
# -------------------------------

def test_predict_missing_text_field():
    """
    Should return 422 Unprocessable Entity if 'text' field is missing.
    """
    payload = {}  # Missing 'text'
    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # FastAPI's default for validation errors

def test_predict_empty_string_input():
    """
    Should return 200 and still include a prediction for empty string input.
    """
    payload = {"text": ""}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["positive", "neutral", "negative"]

def test_predict_non_string_input():
    """
    Should return 422 if 'text' is not a string (e.g., a number).
    """
    payload = {"text": 123}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # Pydantic should reject this
