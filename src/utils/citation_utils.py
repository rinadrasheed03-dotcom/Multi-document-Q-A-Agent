from __future__ import annotations

from src.models import RetrievalResult


def build_source_label(result: RetrievalResult) -> str:
    """Create a readable citation label."""

    return (
        f"{result.chunk.pdf_name}, "
        f"page {result.chunk.page_number}"
    )


def build_sources(
    results: list[RetrievalResult],
) -> list[dict]:
    """Build unique source records for the final response."""

    sources: list[dict] = []
    seen: set[tuple[str, int]] = set()

    for result in results:
        source_key = (
            result.chunk.pdf_name,
            result.chunk.page_number,
        )

        if source_key not in seen:
            seen.add(source_key)

            sources.append(
                {
                    "pdf_name": result.chunk.pdf_name,
                    "page_number": result.chunk.page_number,
                    "chunk_id": result.chunk.chunk_id,
                    "reranker_score": result.reranker_score,
                    "duplicate_sources": result.duplicate_sources,
                }
            )

        for duplicate in result.duplicate_sources:
            duplicate_key = (
                duplicate["pdf_name"],
                duplicate["page_number"],
            )

            if duplicate_key in seen:
                continue

            seen.add(duplicate_key)

            sources.append(
                {
                    "pdf_name": duplicate["pdf_name"],
                    "page_number": duplicate["page_number"],
                    "chunk_id": duplicate["chunk_id"],
                    "reranker_score": None,
                    "duplicate_of": result.chunk.chunk_id,
                    "duplicate_sources": [],
                }
            )

    return sources