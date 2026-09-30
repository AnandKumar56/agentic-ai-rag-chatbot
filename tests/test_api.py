import pytest
from fastapi.testclient import TestClient
from app import app
from unittest.mock import patch

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.query_rag")
def test_chat(mock_query_rag):
    mock_query_rag.return_value = {
        "query": "What is Agentic AI?",
        "final_answer": "Agentic AI refers to systems capable of autonomous actions.",
        "retrieved_context_chunks": ["Agentic AI refers to systems capable..."],
        "confidence_score": 0.95
    }
    
    response = client.post("/chat", json={"query": "What is Agentic AI?"})
    assert response.status_code == 200
    data = response.json()
    assert "final_answer" in data
    assert "retrieved_context_chunks" in data
    assert "confidence_score" in data
    assert data["confidence_score"] == 0.95
