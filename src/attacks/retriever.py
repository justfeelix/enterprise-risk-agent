"""Simulated retrieval layer for prompt-injection experiments."""

from typing import Dict, List, Optional

from ..document_processor import ComplianceDocumentProcessor

_PROCESSOR: Optional[ComplianceDocumentProcessor] = None


def _get_processor() -> ComplianceDocumentProcessor:
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = ComplianceDocumentProcessor(vector_db_path="vector_db")
        _PROCESSOR.load_vector_store()
    return _PROCESSOR


def retrieve_documents(query: str, k: int = 3) -> List[Dict[str, str]]:
    """Return retrieved knowledge-base documents from the existing vector_db."""
    processor = _get_processor()
    docs = processor.retrieve_chunks(query, k=k)

    retrieved_docs: List[Dict[str, str]] = []
    for index, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        page = metadata.get("page")
        doc_id = f"kb_{index}"
        if page is not None:
            doc_id = f"kb_doc_page_{page}"

        retrieved_docs.append(
            {
                "doc_id": doc_id,
                "content": getattr(doc, "page_content", ""),
                "source": "knowledge_base",
            }
        )

    return retrieved_docs
