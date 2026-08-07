from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    llm_model: str = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile",
    )

    llm_temperature: float = float(
        os.getenv("LLM_TEMPERATURE", "0.1")
    )

    llm_max_tokens: int = int(
        os.getenv("LLM_MAX_TOKENS", "700")
    )

    llm_timeout_seconds: float = float(
        os.getenv("LLM_TIMEOUT_SECONDS", "180")
    )

    llm_max_retries: int = int(
        os.getenv("LLM_MAX_RETRIES", "3")
    )

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    chunk_size: int = int(
        os.getenv("CHUNK_SIZE", "1000")
    )

    chunk_overlap: int = int(
        os.getenv("CHUNK_OVERLAP", "150")
    )

    vector_top_k: int = int(
        os.getenv("VECTOR_TOP_K", "20")
    )

    bm25_top_k: int = int(
        os.getenv("BM25_TOP_K", "20")
    )

    rrf_top_k: int = int(
        os.getenv("RRF_TOP_K", "20")
    )

    rerank_top_k: int = int(
        os.getenv("RERANK_TOP_K", "10")
    )

    final_context_k: int = int(
        os.getenv("FINAL_CONTEXT_K", "5")
    )

    rrf_constant: int = int(
        os.getenv("RRF_CONSTANT", "60")
    )

    duplicate_threshold: float = float(
        os.getenv("DUPLICATE_THRESHOLD", "0.94")
    )


settings = Settings()