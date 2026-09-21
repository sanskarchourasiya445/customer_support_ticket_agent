from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    first = repository.create("session-1", request)
    second = repository.create("session-1", request)

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_repository_unique_ids_across_sessions() -> None:
    repository = TicketRepository()
    req1 = TicketCreate(
        customer_name="User 1",
        customer_email="user1@example.com",
        issue_description="Order never arrived",
        category="order",
        summary="Missing order package",
    )
    req2 = TicketCreate(
        customer_name="User 2",
        customer_email="user2@example.com",
        issue_description="Account password reset broken",
        category="account",
        summary="Password reset issue",
    )

    t1 = repository.create("session-1", req1)
    t2 = repository.create("session-2", req2)

    assert t1.ticket_id != t2.ticket_id
    assert t1.ticket_id == "CST-2026-0001"
    assert t2.ticket_id == "CST-2026-0002"
    assert len(list(repository.all())) == 2


def test_repository_get_existing_and_unknown() -> None:
    repository = TicketRepository()
    req = TicketCreate(
        customer_name="User",
        customer_email="user@example.com",
        issue_description="Technical glitch on website checkout",
        category="technical",
        summary="Checkout technical error",
    )
    created = repository.create("session-1", req)

    fetched = repository.get(created.ticket_id)
    assert fetched is not None
    assert fetched.ticket_id == created.ticket_id
    assert fetched.customer_name == "User"
    assert fetched.status == "open"

    assert repository.get("UNKNOWN-ID") is None
