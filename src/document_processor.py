import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (API keys)
load_dotenv()


class ComplianceDocumentProcessor:
    """Processes compliance and financial documents for risk analysis."""

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
        pass

    def query_document(self, query: str):
        pass


# --- Testing Block ---
if __name__ == "__main__":
    processor = ComplianceDocumentProcessor()

    # 1. Load the PDF
    docs = processor.load_documents()

    # 2. Chunk the text
    if docs:
        doc_chunks = processor.chunk_text(docs)
        # Print a sample to prove it worked
        print("\n--- SAMPLE CHUNK ---")
        print(doc_chunks[0].page_content)
        print(f"--- METADATA: {doc_chunks[0].metadata} ---")
    else:
        print("⚠️ No documents found. Did you put a PDF in the data/ folder?")
