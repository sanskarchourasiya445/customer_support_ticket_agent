import pytest

from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")
    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_session_all_fields_complete() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Duplicate charge on credit card",
        category="payment",
    )
    assert state.missing_ticket_fields() == []


def test_session_store_isolation() -> None:
    store = SessionStore()
    s1 = store.get_or_create("session-1")
    s2 = store.get_or_create("session-2")

    s1.customer_name = "Alice"
    s2.customer_name = "Bob"

    assert store.get_or_create("session-1").customer_name == "Alice"
    assert store.get_or_create("session-2").customer_name == "Bob"


def test_blank_session_id_raises() -> None:
    store = SessionStore()
    with pytest.raises(ValueError):
        store.get_or_create("   ")
