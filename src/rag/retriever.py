import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retrieval scaffold with stable output contracts."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    async def initialize(self) -> None:
        documents = split_support_documents(load_support_documents(self.documents_dir))
        embeddings = build_embeddings(self.settings)

        vector_db_path = Path(self.settings.vector_db_path)
        vector_db_path.mkdir(parents=True, exist_ok=True)

        # Sanitize metadata to ensure source contains only the filename
        for chunk in documents:
            if "source" in chunk.metadata:
                chunk.metadata["source"] = Path(chunk.metadata["source"]).name

        # Generate deterministic IDs based on source filename and content hash
        ids = [
            f"{chunk.metadata.get('source', 'doc')}_{i}_{hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()[:12]}"
            for i, chunk in enumerate(documents)
        ]

        store = Chroma(
            collection_name=self.settings.rag_collection,
            embedding_function=embeddings,
            persist_directory=str(vector_db_path),
        )

        store.add_documents(documents=documents, ids=ids)
        self._store = store

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        if self._store is None:
            raise ComponentNotReadyError("Retriever is not initialized")

        if not query or not query.strip():
            return []

        k = limit if limit is not None else self.settings.rag_top_k

        docs = self._store.similarity_search(query.strip(), k=k)

        results: list[dict[str, str]] = []
        for doc in docs:
            source = Path(doc.metadata.get("source", "")).name
            results.append({
                "content": doc.page_content,
                "source": source,
            })
        return results
