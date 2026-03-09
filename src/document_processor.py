"""
Document Processor
------------------
Loads, chunks, embeds, and indexes enterprise PDFs into a local FAISS
vector database using a privacy-first HuggingFace embedding model.

Role in the assurance pipeline:
  data/*.pdf  →  load_documents()  →  chunk_text()  →  create_embeddings()
              →  vector_db/  (persisted FAISS index)
              →  LegalOracle (src/assurance/legal_oracle.py) reads it
              →  judge.py enriches every violation verdict with cited articles
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (API keys)
load_dotenv()


class ComplianceDocumentProcessor:
    """Processes compliance and financial documents for risk analysis."""

    def __init__(self, vector_db_path: str = "vector_db"):
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None

    def load_documents(self, data_dir: str = "data"):
        """Load PDF documents from the specified directory."""
        if not data_dir:
            print("⚠️ No data directory specified. Please provide a path to the PDFs.")
            return []
        print(f"📂 Scanning '{data_dir}' for PDF files...")
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".pdf"):
                    print(f"  - {filename}")
        print(f"📥 Loading PDFs from '{data_dir}'...")
        # PyPDFDirectoryLoader recursively loads PDF files in a directory.
        loader = PyPDFDirectoryLoader(data_dir, glob="**/*.pdf")
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document pages.")
        return documents

    def chunk_text(self, documents):
        """Split documents into smaller overlapping chunks."""
        print("✂️ Chunking text...")
        # We use a 200 character overlap so a sentence cut in half isn't lost
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Split text into {len(chunks)} searchable chunks.")
        return chunks

    def create_embeddings(self, chunks):
        """Create and persist FAISS embeddings for document chunks."""
        if not chunks:
            print("⚠️ No chunks available to embed.")
            return None

        print("🧠 Creating local embeddings with HuggingFace model...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_db_path)
        print(f"✅ Saved FAISS index to '{self.vector_db_path}'.")
        return self.vector_store

    def load_vector_store(self):
        """Load a previously saved FAISS vector store from disk."""
        if not os.path.exists(self.vector_db_path):
            print(
                f"⚠️ Vector DB not found at '{self.vector_db_path}'. "
                "Run create_embeddings first."
            )
            return None

        self.vector_store = FAISS.load_local(
            self.vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self.vector_store

    def retrieve_chunks(self, query: str, k: int = 4) -> List:
        """Retrieve the most relevant chunks for a query."""
        if self.vector_store is None:
            self.load_vector_store()
        if self.vector_store is None:
            return []

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        return results

    def _format_citations(self, docs: List) -> List[str]:
        """Build citation strings from chunk metadata."""
        citations = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown_source")
            page = doc.metadata.get("page")
            page_value = page + 1 if isinstance(page, int) else "unknown"
            citations.append(f"{source} (page {page_value})")
        return citations

    def query_document(self, query: str, k: int = 4):
        """Answer a query using retrieved evidence and return citations."""
        if not query.strip():
            return {"answer": "Query is empty.", "citations": []}

        retrieved_docs = self.retrieve_chunks(query, k=k)
        if not retrieved_docs:
            return {
                "answer": "No evidence found. Build or load the vector index first.",
                "citations": [],
            }

        context_blocks = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown_source")
            page = doc.metadata.get("page")
            page_value = page + 1 if isinstance(page, int) else "unknown"
            context_blocks.append(
                f"[Evidence {idx}] Source: {source}, Page: {page_value}\n"
                f"{doc.page_content}"
            )
        context = "\n\n".join(context_blocks)

        # If no OpenAI key is configured, return a retrieval-only response.
        if not os.getenv("OPENAI_API_KEY"):
            return {
                "answer": (
                    "OPENAI_API_KEY not found. Returning retrieved evidence only. "
                    "Configure a key to generate a synthesized answer."
                ),
                "citations": self._format_citations(retrieved_docs),
                "evidence": context,
            }

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = (
            "You are a financial and compliance research assistant. "
            "Answer ONLY using the provided evidence. "
            "If evidence is insufficient, say so clearly.\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{context}\n\n"
            "Provide a concise answer."
        )
        answer = llm.invoke(prompt).content

        return {
            "answer": answer,
            "citations": self._format_citations(retrieved_docs),
            "evidence": context,
        }


# --- Testing Block ---
if __name__ == "__main__":
    processor = ComplianceDocumentProcessor()

    # 1. Load the PDF
    docs = processor.load_documents()

    # 2. Chunk the text
    if docs:
        doc_chunks = processor.chunk_text(docs)
        processor.create_embeddings(doc_chunks)

        # Print a sample to prove it worked
        print("\n--- SAMPLE CHUNK ---")
        print(doc_chunks[0].page_content)
        print(f"--- METADATA: {doc_chunks[0].metadata} ---")

        sample_query = (
            "What AI practices are prohibited under Article 5 of the EU AI Act?"
        )
        result = processor.query_document(sample_query)
        print("\n--- SAMPLE QUERY ---")
        print(f"Q: {sample_query}")
        print(f"A: {result['answer']}")
        print("Citations:")
        for c in result["citations"]:
            print(f" - {c}")
    else:
        print("⚠️ No documents found. Did you put a PDF in the data/ folder?")
