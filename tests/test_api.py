from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api.server import app, pipeline
from tests.test_pipeline import MockChatModel


@pytest.fixture
def client(tmp_path: Path, knowledge_dir: Path):
    pipeline.model = MockChatModel(response_text="Standard shipping takes 3-5 business days.")
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_chat_rag_endpoint(client: TestClient) -> None:
    payload = {
        "session_id": "api-session-1",
        "message": "What is the standard delivery shipping time?",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["session_id"] == "api-session-1"
    assert "3-5 business days" in data["response"]
    assert "shipping.md" in data["sources"]
    assert data["ticket_id"] is None


def test_chat_ticket_creation_and_retrieval(client: TestClient) -> None:
    session_id = "api-ticket-session"

    # Turn 1: Describe payment problem
    r1 = client.post("/chat", json={"session_id": session_id, "message": "My card was charged twice."})
    assert r1.status_code == 200
    assert r1.json()["ticket_id"] is None

    # Turn 2: Provide name
    r2 = client.post("/chat", json={"session_id": session_id, "message": "My name is John Doe"})
    assert r2.status_code == 200
    assert r2.json()["ticket_id"] is None

    # Turn 3: Provide email
    r3 = client.post("/chat", json={"session_id": session_id, "message": "john.doe@example.com"})
    assert r3.status_code == 200
    ticket_id = r3.json()["ticket_id"]
    assert ticket_id is not None
    assert ticket_id.startswith("CST-2026-")

    # Fetch created ticket via GET /tickets/{ticket_id}
    ticket_res = client.get(f"/tickets/{ticket_id}")
    assert ticket_res.status_code == 200
    ticket_data = ticket_res.json()
    assert ticket_data["ticket_id"] == ticket_id
    assert ticket_data["customer_name"] == "John Doe"
    assert ticket_data["customer_email"] == "john.doe@example.com"
    assert ticket_data["category"] == "payment"
    assert ticket_data["status"] == "open"


def test_get_unknown_ticket_returns_404(client: TestClient) -> None:
    response = client.get("/tickets/NON-EXISTENT-ID")
    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found"
