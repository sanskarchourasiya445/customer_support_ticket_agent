from pathlib import Path

import pytest

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError


def test_loader_preserves_source_names(knowledge_dir: Path) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir: Path) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


@pytest.mark.asyncio
async def test_retriever_uninitialized_raises(knowledge_dir: Path, tmp_path: Path) -> None:
    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vdb"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-col",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    with pytest.raises(ComponentNotReadyError):
        await retriever.search("shipping")


@pytest.mark.asyncio
async def test_retriever_blank_query_returns_empty(knowledge_dir: Path, tmp_path: Path) -> None:
    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vdb"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-col",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()
    results = await retriever.search("   ")
    assert results == []


@pytest.mark.asyncio
async def test_retriever_initialization_and_search(knowledge_dir: Path, tmp_path: Path) -> None:
    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vdb"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-col",
        rag_top_k=2,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    # Re-initialization should be idempotent
    await retriever.initialize()

    results = await retriever.search("How long does standard delivery take?")
    assert len(results) > 0
    assert any("shipping.md" in r["source"] for r in results)
    assert any("business days" in r["content"] for r in results)
