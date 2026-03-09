# Autonomous Risk & Compliance Syndicate

An enterprise-grade Multi-Agent AI system built with **LangChain**, designed for analyzing financial and legal PDFs to support risk assessment and regulatory compliance workflows.

## Overview

The Autonomous Risk & Compliance Syndicate leverages large language models and retrieval-augmented generation (RAG) to ingest, process, and query complex compliance documents — including financial reports, legal contracts, and regulatory filings.

## Features

- **PDF Ingestion**: Load and parse financial and legal PDF documents automatically.
- **Intelligent Chunking**: Split large documents into semantically meaningful chunks for accurate retrieval.
- **Vector Embeddings**: Generate and persist document embeddings using FAISS for fast similarity search.
- **Natural Language Querying**: Ask plain-English questions against your compliance document library.
- **Multi-Agent Architecture**: Coordinated AI agents handle different aspects of the risk and compliance pipeline.

## Project Structure

```
enterprise-risk-agent/
├── data/               # Raw PDF documents
├── vector_db/          # Persisted FAISS vector database
├── src/
│   ├── __init__.py
│   └── document_processor.py   # Core document processing class
├── .env.example        # Environment variable template
├── requirements.txt    # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/justfeelix/enterprise-risk-agent.git
   cd enterprise-risk-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and set your OPENAI_API_KEY
   ```

4. Add your PDF documents to the `data/` directory.

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core LangChain framework |
| `langchain-openai` | OpenAI integration for LangChain |
| `pypdf` | PDF parsing |
| `faiss-cpu` | Vector similarity search |
| `python-dotenv` | Environment variable management |

## License

See [LICENSE](LICENSE) for details.
