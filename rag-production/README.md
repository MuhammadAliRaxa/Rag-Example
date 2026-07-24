# RAG Production Architecture

Enterprise-grade Retrieval-Augmented Generation (RAG) system modular codebase.

## Directory Structure

- `src/ingestion`: Loaders, parsers, semantic chunkers, embedding wrapper, and deduplication.
- `src/retrieval`: Vector DB search, BM25 keyword search, hybrid RRF fusion, reranker, query rewriter.
- `src/generation`: System prompt templates, LLM client wrapper, citation mapping.
- `src/memory`: Redis-backed multi-session conversation memory.
- `src/evaluation`: Golden dataset loader, retrieval & answer quality metrics, evaluation CLI runner.
- `src/api`: FastAPI entrypoint, REST endpoints (`/chat`, `/ingest`, `/health`), Pydantic schemas, middleware.
- `src/config`: Settings & logging configuration.

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run API Server**:
   ```bash
   make serve
   ```

3. **Run Ingestion Pipeline**:
   ```bash
   make ingest FILE=data/raw/sample.pdf
   ```

4. **Run Evaluation**:
   ```bash
   make eval
   ```

5. **Run Tests**:
   ```bash
   pytest
   ```
