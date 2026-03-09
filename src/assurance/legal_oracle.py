"""
Legal Oracle
------------
Wraps the RAG document processor to retrieve EU AI Act passages that
provide legal grounding for policy violations detected during assurance
runs.

Data flow:
  EU-AI-ACT.pdf  →  document_processor (FAISS index)
                 →  LegalOracle.retrieve_relevant_law(query)
                 →  judge.py enriches verdict with real article citations
"""

from ..document_processor import ComplianceDocumentProcessor


class LegalOracle:
    """Queries the persisted EU AI Act FAISS index to justify policy verdicts.

    Uses lazy loading: the vector store is only read from disk on the first
    query, so constructing the oracle is always cheap.
    """

    def __init__(self, vector_db_path: str = "vector_db"):
        # Reuse the same embedding model + FAISS wrapper from the RAG layer.
        self._processor = ComplianceDocumentProcessor(vector_db_path=vector_db_path)
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load the vector store on the first call."""
        if self._loaded:
            return True
        store = self._processor.load_vector_store()
        if store is None:
            print(
                "⚠️  Legal Oracle: vector index not found. "
                "Run with --ingest to build it from data/ first."
            )
            return False
        self._loaded = True
        return True

    # ------------------------------------------------------------------
    # Public interface used by judge.py
    # ------------------------------------------------------------------

    def retrieve_relevant_law(self, query: str, k: int = 2) -> list:
        """Return the top-k most relevant EU AI Act chunks for a query string."""
        if not self._ensure_loaded():
            return []
        return self._processor.retrieve_chunks(query, k=k)

    def format_legal_citations(self, docs: list) -> list:
        """Convert retrieved document chunks into readable citation strings."""
        return self._processor._format_citations(docs)
