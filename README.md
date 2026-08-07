# Multi-PDF RAG Agent

A Multi-PDF Retrieval-Augmented Generation system that answers
questions across multiple PDF documents using:

- Vector semantic retrieval
- BM25 keyword retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Duplicate-aware retrieval
- Source and page citations
- LangGraph workflow
- NVIDIA AI Endpoints
- Streamlit interface

## Architecture

```text
Multiple PDFs
     |
     v
PDF Processing Agent
     |
     +----------------------+
     |                      |
     v                      v
Vector Retrieval Agent   BM25 Retrieval Agent
     |                      |
     +----------+-----------+
                |
                v
        RRF Fusion Agent
                |
                v
         Reranking Agent
                |
                v
      Duplicate-Aware Agent
                |
                v
           Answer Agent
                |
                v
      Answer with Citations