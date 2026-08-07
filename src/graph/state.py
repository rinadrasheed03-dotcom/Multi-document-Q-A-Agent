from __future__ import annotations

from typing import Any, TypedDict

from src.models import RetrievalResult


class RAGState(TypedDict, total=False):
    """Shared state passed between LangGraph agents."""

    query: str

    vector_results: list[RetrievalResult]
    bm25_results: list[RetrievalResult]
    fused_results: list[RetrievalResult]
    reranked_results: list[RetrievalResult]
    final_results: list[RetrievalResult]

    answer: str
    sources: list[dict[str, Any]]
    metrics: dict[str, Any]