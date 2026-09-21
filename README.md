# Customer Support Ticket Agent

A guided starter project for building a text-based support agent using FastAPI, Streamlit, an open-source LLM, RAG, and a mock ticket tool.

Read the separate Assignment Implementation Guide before changing the starter.

## Architecture

```text
Streamlit -> FastAPI -> SupportPipeline -> LangGraph workflow
                                      |-> RAG/Chroma knowledge
                                      \-> session-bound ticket tool
```

## What Is Provided

- Request, response, and ticket models.
- A LangChain `ChatOpenAI` binding for an OpenAI-compatible open-source model.
- A typed LangGraph state, node skeleton, and routing graph.
- Document loading, splitting, embeddings, and Chroma component bindings.
- Session-state data structure.
- In-memory ticket repository with duplicate protection.
- Four support-policy documents.
- FastAPI and Streamlit scaffolding.
- RAG and agent integration TODOs.
- Initial repository and session tests.

## Candidate Work

Complete the TODOs in:

- `src/rag/retriever.py`
- `src/llm/workflow.py`
- `src/pipeline.py`
- `streamlit_app.py`

You may add or reorganize files when the resulting design remains clear and testable.

## Requirements

- Python 3.11 or newer.
- An OpenAI-compatible endpoint serving an open-source instruction model, or an equivalent open-source model integration.
- Enough local space for the selected embedding model and vector index.

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### macOS

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure `.env` for the selected model endpoint. Do not commit credentials.

## Run

Start the API:

```sh
uvicorn src.api.server:app --reload --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
```

In another terminal, start the UI:

```sh
streamlit run streamlit_app.py --server.address "${STREAMLIT_HOST:-127.0.0.1}" --server.port "${STREAMLIT_PORT:-8501}"
```

The default UI is `http://localhost:8501`; FastAPI documentation is `http://localhost:8000/docs`. Override both ports through `.env` and the corresponding command-line values when necessary.

## Test

```sh
pytest -q
```

Useful manual requests:

```sh
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"How long does standard shipping take?"}'
```

For Windows PowerShell, use `Invoke-RestMethod` or place the equivalent JSON request in FastAPI's `/docs` interface.

## Implementation Summary

All candidate tasks and TODOs have been fully implemented following the assignment specifications:

1. **RAG Knowledge Base & Retriever** ([`src/rag/retriever.py`](file:///c:/Users/DELL/OneDrive/Desktop/tanishka/src/rag/retriever.py)):
   - Persistent Chroma vector store using `sentence-transformers/all-MiniLM-L6-v2`.
   - Deterministic chunk IDs (`{source}_{idx}_{content_hash}`) to ensure restart idempotency without duplicate indexing.
   - Safe metadata preservation (only basename `source` filenames exposed).
   - Relevance-aware similarity retrieval with `rag_top_k` limiting.

2. **LangGraph Agent Workflow** ([`src/llm/workflow.py`](file:///c:/Users/DELL/OneDrive/Desktop/tanishka/src/llm/workflow.py)):
   - `retrieve`: Retrieves relevant policy chunks and deduplicates content.
   - `decide`: Structured intent classification and entity extraction for `customer_name`, `customer_email`, `issue_description`, and `category`.
   - `answer`: Grounded answers strictly using retrieved context. Formulates clear unknown-answer responses without hallucinations when knowledge is missing. Deduplicates and returns source filenames.
   - `collect_or_create`: Multi-turn field accumulation in `SessionStore`, single focused follow-up questions for missing fields, strict Pydantic validation on email/fields, and ticket creation via `TicketRepository`.
   - `select_route`: Conditional edge routing between `"answer"` and `"ticket"`.

3. **Pipeline Orchestrator** ([`src/pipeline.py`](file:///c:/Users/DELL/OneDrive/Desktop/tanishka/src/pipeline.py)):
   - Manages startup initialization of Chroma and the LangGraph workflow.
   - Handles multi-turn conversation state persistence by client-supplied `session_id`.
   - Maps errors to `ComponentNotReadyError` and `AgentProcessingError`.

4. **Streamlit UI** ([`streamlit_app.py`](file:///c:/Users/DELL/OneDrive/Desktop/tanishka/streamlit_app.py)):
   - Session-aware chat communicating with `POST /chat`.
   - Displays assistant text, source document badges (`📚 Sources: shipping.md`), and ticket confirmation callouts (`🎫 Ticket Created: CST-2026-0001`).
   - Friendly error handling for connection failures, 502/503/422 HTTP errors, and request timeouts.
   - Sidebar with session reset and conversation info.

5. **Test Suite** ([`tests/`](file:///c:/Users/DELL/OneDrive/Desktop/tanishka/tests)):
   - 21 automated unit and integration tests covering RAG indexing/search, grounded policy questions, unknown questions without hallucination, multi-turn ticket creation, duplicate ticket protection, session isolation, and FastAPI endpoint contracts.

## Test Verification

Run the full pytest suite:

```powershell
.\.venv\Scripts\pytest.exe
```
