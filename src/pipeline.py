from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import SupportWorkflowState, build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        await self.retriever.initialize()
        self.workflow = build_support_workflow(
            self.model,
            retriever=self.retriever,
            sessions=self.sessions,
            tickets=self.tickets,
        )
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        clean_session_id = session_id.strip() if session_id else ""
        clean_message = message.strip() if message else ""

        if not clean_session_id:
            raise ValueError("session_id must not be blank")
        if not clean_message:
            raise ValueError("message must not be blank")

        conv_state = self.sessions.get_or_create(clean_session_id)
        conv_state.history.append({"role": "user", "content": clean_message})

        initial_state: SupportWorkflowState = {
            "session_id": clean_session_id,
            "customer_message": clean_message,
            "messages": [],
            "retrieved_chunks": [],
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": conv_state.ticket_id,
        }

        try:
            result = await self.workflow.ainvoke(initial_state)
        except ComponentNotReadyError:
            raise
        except AgentProcessingError:
            raise
        except Exception as exc:
            raise AgentProcessingError(f"Agent turn processing failed: {exc}") from exc

        response_text = (result.get("response_text") or "").strip()
        if not response_text:
            response_text = "I received your message, but could not generate a response. Please try again."

        sources = result.get("sources", [])
        ticket_id = result.get("ticket_id")

        # Validate ticket_id against ticket repository
        if ticket_id and not self.tickets.get(ticket_id):
            ticket_id = None

        conv_state.history.append({"role": "assistant", "content": response_text})
        if ticket_id:
            conv_state.ticket_id = ticket_id

        return ChatResponse(
            success=True,
            session_id=clean_session_id,
            response=response_text,
            sources=sources,
            ticket_id=ticket_id,
        )
