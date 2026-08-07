from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """A chunk extracted from a PDF page."""

    chunk_id: str
    text: str
    pdf_name: str
    page_number: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A chunk returned by a retrieval agent."""

    chunk: DocumentChunk
    score: float
    rank: int
    retriever: str

    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None

    duplicate_sources: list[dict[str, Any]] = field(
        default_factory=list
    )


@dataclass
class RAGResponse:
    """Final response returned by the workflow."""

    answer: str
    sources: list[dict[str, Any]]
    metrics: dict[str, Any]