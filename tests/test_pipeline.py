from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest

from src.config import Settings
from src.pipeline import SupportPipeline
from src.utils.errors import ComponentNotReadyError


class MockChatModel(BaseChatModel):
    response_text: str = "Standard delivery takes 3 to 5 business days."

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response_text))])

    @property
    def _llm_type(self) -> str:
        return "mock-chat"


def get_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vdb"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


@pytest.mark.asyncio
async def test_uninitialized_pipeline_rejects_chat(tmp_path: Path, knowledge_dir: Path) -> None:
    pipeline = SupportPipeline(get_test_settings(tmp_path), knowledge_dir)
    with pytest.raises(ComponentNotReadyError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
async def test_rag_policy_answer_with_sources(tmp_path: Path, knowledge_dir: Path) -> None:
    pipeline = SupportPipeline(get_test_settings(tmp_path), knowledge_dir)
    pipeline.model = MockChatModel(response_text="Standard shipping takes 3-5 business days according to policy.")
    await pipeline.initialize()

    response = await pipeline.process("session-rag-1", "How long does standard delivery take?")
    assert response.success is True
    assert "3-5 business days" in response.response
    assert "shipping.md" in response.sources
    assert response.ticket_id is None


@pytest.mark.asyncio
async def test_unknown_question_states_no_answer(tmp_path: Path, knowledge_dir: Path) -> None:
    pipeline = SupportPipeline(get_test_settings(tmp_path), knowledge_dir)
    pipeline.model = MockChatModel(response_text="Unused")
    await pipeline.initialize()

    # Query with non-matching retriever
    # Simulate an empty retriever search by querying something with no retrieved documents or empty search
    pipeline.retriever.search = lambda q, limit=None: []  # type: ignore[assignment]

    response = await pipeline.process("session-unknown-1", "What is the recipe for chocolate cake?")
    assert response.success is True
    assert "knowledge base does not contain" in response.response.lower()
    assert response.sources == []
    assert response.ticket_id is None


@pytest.mark.asyncio
async def test_multi_turn_ticket_creation(tmp_path: Path, knowledge_dir: Path) -> None:
    pipeline = SupportPipeline(get_test_settings(tmp_path), knowledge_dir)
    pipeline.model = MockChatModel()
    await pipeline.initialize()

    session_id = "session-ticket-test-1"

    # Turn 1: Initial complaint (payment double charge)
    r1 = await pipeline.process(session_id, "My payment was charged twice for order #1234.")
    assert r1.ticket_id is None
    assert "name" in r1.response.lower()

    # Turn 2: Provide name
    r2 = await pipeline.process(session_id, "My name is Rahul Sharma.")
    assert r2.ticket_id is None
    assert "email" in r2.response.lower()

    # Turn 3: Provide email (triggers ticket creation)
    r3 = await pipeline.process(session_id, "rahul.sharma@example.com")
    assert r3.ticket_id is not None
    assert r3.ticket_id.startswith("CST-2026-")
    assert "ticket has been created" in r3.response.lower()

    # Verify ticket in repository
    ticket = pipeline.tickets.get(r3.ticket_id)
    assert ticket is not None
    assert ticket.customer_name == "Rahul Sharma"
    assert ticket.customer_email == "rahul.sharma@example.com"
    assert ticket.category == "payment"
    assert ticket.status == "open"


@pytest.mark.asyncio
async def test_duplicate_ticket_protection_in_pipeline(tmp_path: Path, knowledge_dir: Path) -> None:
    pipeline = SupportPipeline(get_test_settings(tmp_path), knowledge_dir)
    pipeline.model = MockChatModel()
    await pipeline.initialize()

    session_id = "session-dup-test"

    # Complete flow to create ticket
    await pipeline.process(session_id, "I have a payment issue and was charged twice.")
    await pipeline.process(session_id, "Rahul")
    r3 = await pipeline.process(session_id, "rahul@example.com")
    original_ticket_id = r3.ticket_id
    assert original_ticket_id is not None

    # Repeated message in same session
    r4 = await pipeline.process(session_id, "I was charged twice.")
    assert r4.ticket_id == original_ticket_id
    assert "already active" in r4.response.lower() or original_ticket_id in r4.response
    assert len(list(pipeline.tickets.all())) == 1
