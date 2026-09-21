import json
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, EmailStr, Field, TypeAdapter, ValidationError

from src.llm.prompts import ANSWER_TEMPLATE, DECISION_SYSTEM_PROMPT, SYSTEM_PROMPT
from src.models import TicketCategory, TicketCreate
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError


class SupportWorkflowState(TypedDict, total=False):
    """Typed state shared by the supplied LangGraph node skeletons."""

    session_id: str
    customer_message: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict[str, str]]
    route: str
    extracted_fields: dict[str, str]
    response_text: str
    sources: list[str]
    ticket_id: str | None


class DecisionOutput(BaseModel):
    route: Literal["answer", "ticket"] = Field(
        description="Choose 'answer' for general informational policy questions. Choose 'ticket' for complaints, issues, bug reports, human assistance requests, or providing ticket details."
    )
    customer_name: str | None = Field(
        default=None,
        description="Customer full name if provided in this message, else null.",
    )
    customer_email: str | None = Field(
        default=None,
        description="Customer email address if provided in this message, else null.",
    )
    issue_description: str | None = Field(
        default=None,
        description="Specific problem or issue description if provided in this message, else null.",
    )
    category: Literal["order", "payment", "account", "technical", "other"] | None = Field(
        default=None,
        description="Category of the ticket if identifiable, else null.",
    )


def build_support_workflow(
    model: BaseChatModel,
    retriever: KnowledgeRetriever | None = None,
    sessions: SessionStore | None = None,
    tickets: TicketRepository | None = None,
):
    """Build the LangGraph support workflow with retrieval, decision, answering, and ticket creation."""

    email_adapter = TypeAdapter(EmailStr)

    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        customer_message = (state.get("customer_message") or "").strip()
        chunks: list[dict[str, str]] = []

        if retriever is not None and customer_message:
            try:
                chunks = await retriever.search(customer_message)
            except Exception:
                chunks = []

        # Deduplicate chunks by content
        seen_content: set[str] = set()
        deduped_chunks: list[dict[str, str]] = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if content and content not in seen_content:
                seen_content.add(content)
                deduped_chunks.append(chunk)

        return {"retrieved_chunks": deduped_chunks}

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        customer_message = (state.get("customer_message") or "").strip()
        session_id = state.get("session_id", "")
        retrieved_chunks = state.get("retrieved_chunks", [])

        # Check if current session is already in active ticket collection
        conv_state = sessions.get_or_create(session_id) if sessions and session_id else None
        has_pending_ticket = bool(
            conv_state and not conv_state.ticket_id and (
                conv_state.customer_name or conv_state.customer_email or
                conv_state.issue_description or conv_state.category
            )
        )

        decision: DecisionOutput | None = None
        context_str = "\n\n".join(c.get("content", "") for c in retrieved_chunks)

        # Attempt structured output extraction
        try:
            structured_model = model.with_structured_output(DecisionOutput)
            result = await structured_model.ainvoke([
                SystemMessage(content=DECISION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Knowledge context:\n{context_str}\n\nCustomer message: {customer_message}"
                ),
            ])
            if isinstance(result, DecisionOutput):
                decision = result
            elif isinstance(result, dict):
                decision = DecisionOutput(**result)
        except Exception:
            decision = None

        # Fallback heuristic if LLM structured output parsing fails
        if decision is None:
            # Detect email
            email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", customer_message)
            extracted_email = email_match.group(0) if email_match else None

            # Detect name
            name_match = re.search(r"(?:my name is|i am|i'm|this is)\s+([A-Za-z\s]+)", customer_message, re.IGNORECASE)
            extracted_name = name_match.group(1).strip() if name_match else None

            # Detect category
            detected_category = None
            msg_lower = customer_message.lower()
            if any(w in msg_lower for w in ["payment", "charged", "billing", "card", "refund", "transaction"]):
                detected_category = "payment"
            elif any(w in msg_lower for w in ["order", "shipping", "deliver", "tracking", "package"]):
                detected_category = "order"
            elif any(w in msg_lower for w in ["account", "password", "login", "sign in"]):
                detected_category = "account"
            elif any(w in msg_lower for w in ["bug", "error", "broken", "technical", "crash"]):
                detected_category = "technical"

            # Detect ticket keywords
            ticket_keywords = ["issue", "problem", "charged", "wrong", "broken", "ticket", "refund", "cancel", "help", "support"]
            is_ticket_intent = (
                has_pending_ticket
                or any(kw in msg_lower for kw in ticket_keywords)
                or bool(extracted_email)
                or bool(extracted_name and has_pending_ticket)
            )

            # If user message is just a standalone name like "Rahul Sharma" during pending ticket
            if has_pending_ticket and not extracted_name and not extracted_email and re.match(r"^[A-Za-z\s]{2,40}$", customer_message) and len(customer_message.split()) <= 4:
                extracted_name = customer_message.strip()

            decision = DecisionOutput(
                route="ticket" if is_ticket_intent else "answer",
                customer_name=extracted_name,
                customer_email=extracted_email,
                issue_description=customer_message if (is_ticket_intent and not has_pending_ticket and len(customer_message) >= 5) else None,
                category=detected_category,
            )

        # If a session is in the middle of ticket collection, prioritize ticket route
        if has_pending_ticket and decision.route != "ticket":
            decision.route = "ticket"

        extracted: dict[str, str] = {}
        if decision.customer_name:
            extracted["customer_name"] = str(decision.customer_name).strip()
        if decision.customer_email:
            extracted["customer_email"] = str(decision.customer_email).strip()
        if decision.issue_description:
            extracted["issue_description"] = str(decision.issue_description).strip()
        if decision.category:
            extracted["category"] = str(decision.category).strip()

        return {
            "route": decision.route,
            "extracted_fields": extracted,
        }

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        customer_message = (state.get("customer_message") or "").strip()
        retrieved_chunks = state.get("retrieved_chunks", [])

        if not retrieved_chunks:
            return {
                "response_text": "I'm sorry, but our knowledge base does not contain information to answer that question.",
                "sources": [],
                "ticket_id": None,
            }

        context_str = "\n\n".join(
            f"[{c.get('source', 'document')}]\n{c.get('content', '')}"
            for c in retrieved_chunks
        )

        prompt_content = ANSWER_TEMPLATE.format(
            context=context_str,
            message=customer_message,
        )

        try:
            response = await model.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ])
            response_text = response.content if hasattr(response, "content") else str(response)
        except Exception:
            # Ground response in retrieved policy context if upstream model connection fails
            first_chunk = retrieved_chunks[0].get("content", "").strip() if retrieved_chunks else ""
            response_text = first_chunk or "I'm sorry, but our knowledge base does not contain information to answer that question."


        # Extract unique source names
        sources = sorted(list({c["source"] for c in retrieved_chunks if c.get("source")}))

        return {
            "response_text": response_text,
            "sources": sources,
            "ticket_id": None,
        }

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        session_id = state.get("session_id", "")
        customer_message = (state.get("customer_message") or "").strip()
        extracted = state.get("extracted_fields", {})

        if not sessions:
            raise AgentProcessingError("Session store is unavailable")
        if not tickets:
            raise AgentProcessingError("Ticket repository is unavailable")

        conv_state = sessions.get_or_create(session_id)

        # Check if a ticket has already been created in this session
        if conv_state.ticket_id:
            return {
                "response_text": f"A support ticket for this issue is already active under Ticket ID: **{conv_state.ticket_id}**.",
                "sources": [],
                "ticket_id": conv_state.ticket_id,
            }

        # Merge newly extracted fields into conv_state without overwriting valid data
        if extracted.get("customer_name") and not conv_state.customer_name:
            conv_state.customer_name = extracted["customer_name"]

        if extracted.get("customer_email") and not conv_state.customer_email:
            try:
                email_adapter.validate_python(extracted["customer_email"])
                conv_state.customer_email = extracted["customer_email"]
            except ValidationError:
                pass

        if extracted.get("issue_description") and not conv_state.issue_description:
            conv_state.issue_description = extracted["issue_description"]

        if extracted.get("category") and not conv_state.category:
            cat = extracted["category"].lower()
            if cat in ["order", "payment", "account", "technical", "other"]:
                conv_state.category = cat

        # Heuristic checks if fields were missed by extraction
        if not conv_state.customer_email:
            email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", customer_message)
            if email_match:
                try:
                    email_adapter.validate_python(email_match.group(0))
                    conv_state.customer_email = email_match.group(0)
                except ValidationError:
                    pass

        # If name is missing and message looks like a direct name response (e.g. "Rahul", "My name is John")
        if not conv_state.customer_name and conv_state.issue_description:
            name_match = re.match(r"^(?:my name is |i am |i'm )?([a-zA-Z\s]{2,40})$", customer_message, re.IGNORECASE)
            if name_match and "@" not in customer_message and not any(kw in customer_message.lower() for kw in ["issue", "order", "payment", "card", "charge"]):
                conv_state.customer_name = name_match.group(1).strip()

        # If issue description is still missing but message is a substantial description
        if not conv_state.issue_description and len(customer_message) >= 5 and "@" not in customer_message and not conv_state.customer_name == customer_message:
            conv_state.issue_description = customer_message

        # Check for missing required fields
        missing = conv_state.missing_ticket_fields()

        # If category is the only missing field, infer a reasonable category or default to 'other'
        if missing == ["category"] and conv_state.issue_description:
            desc_lower = conv_state.issue_description.lower()
            if any(w in desc_lower for w in ["pay", "charge", "card", "billed", "refund", "transaction", "fee"]):
                conv_state.category = "payment"
            elif any(w in desc_lower for w in ["order", "shipping", "deliver", "package", "item", "tracking"]):
                conv_state.category = "order"
            elif any(w in desc_lower for w in ["account", "password", "login", "sign in", "auth"]):
                conv_state.category = "account"
            elif any(w in desc_lower for w in ["bug", "error", "broken", "crash", "glitch", "technical"]):
                conv_state.category = "technical"
            else:
                conv_state.category = "other"
            missing = conv_state.missing_ticket_fields()

        # If fields are still missing, ask for ONLY ONE missing field
        if missing:
            next_field = missing[0]
            if next_field == "customer_name":
                prompt_text = "To help create your support ticket, could you please provide your full name?"
            elif next_field == "customer_email":
                name_prefix = f"Thank you, {conv_state.customer_name}. " if conv_state.customer_name else ""
                prompt_text = f"{name_prefix}Could you please share your email address so our team can follow up?"
            elif next_field == "issue_description":
                prompt_text = "Could you please describe the issue you are experiencing in detail?"
            elif next_field == "category":
                prompt_text = "Please specify the category of your request: order, payment, account, technical, or other."
            else:
                prompt_text = f"Please provide your {next_field.replace('_', ' ')}."

            return {
                "response_text": prompt_text,
                "sources": [],
                "ticket_id": None,
            }

        # All 4 fields are present - create the ticket!
        summary = f"[{conv_state.category.capitalize()}] {conv_state.issue_description}"
        if len(summary) > 150:
            summary = summary[:147] + "..."
        if len(summary) < 5:
            summary = f"{conv_state.category.capitalize()} issue request"

        ticket_create = TicketCreate(
            customer_name=conv_state.customer_name,
            customer_email=conv_state.customer_email,
            issue_description=conv_state.issue_description,
            category=conv_state.category,  # type: ignore[arg-type]
            summary=summary,
        )

        ticket = tickets.create(session_id, ticket_create)
        conv_state.ticket_id = ticket.ticket_id

        confirmation_text = (
            f"Your support ticket has been created successfully!\n\n"
            f"• **Ticket ID:** `{ticket.ticket_id}`\n"
            f"• **Category:** {ticket.category.capitalize()}\n"
            f"• **Summary:** {ticket.summary}\n"
            f"• **Customer:** {ticket.customer_name} ({ticket.customer_email})\n\n"
            f"Our customer support team has received your request and will follow up with you shortly."
        )

        return {
            "response_text": confirmation_text,
            "sources": [],
            "ticket_id": ticket.ticket_id,
        }

    def select_route(state: SupportWorkflowState) -> str:
        route = state.get("route", "answer")
        if route not in ["answer", "ticket"]:
            raise AgentProcessingError(f"Unrecognized workflow route: {route}")
        return route

    graph = StateGraph(SupportWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("answer", answer)
    graph.add_node("ticket", collect_or_create)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        select_route,
        {"answer": "answer", "ticket": "ticket"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("ticket", END)
    return graph.compile()
