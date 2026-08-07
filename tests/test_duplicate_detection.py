import numpy as np

from src.agents.duplicate_aware_agent import (
    DuplicateAwareAgent,
)
from src.models import DocumentChunk, RetrievalResult


class FakeEmbeddingService:
    """Return deterministic test embeddings."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = []

        for text in texts:
            lowered = text.lower()

            if "26 weeks" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return np.asarray(vectors, dtype=np.float32)


def make_result(
    chunk_id: str,
    text: str,
    pdf_name: str,
) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        pdf_name=pdf_name,
        page_number=1,
        chunk_index=0,
    )

    return RetrievalResult(
        chunk=chunk,
        score=1.0,
        rank=1,
        retriever="reranker",
        reranker_score=1.0,
    )


def test_semantic_duplicates_are_grouped() -> None:
    results = [
        make_result(
            "a",
            "Employees receive 26 weeks of maternity leave.",
            "policy_a.pdf",
        ),
        make_result(
            "b",
            "Maternity leave entitlement is 26 weeks.",
            "policy_b.pdf",
        ),
        make_result(
            "c",
            "Leave requests must use the HR portal.",
            "policy_c.pdf",
        ),
    ]

    agent = DuplicateAwareAgent(
        embedding_service=FakeEmbeddingService(),
        similarity_threshold=0.94,
    )

    filtered = agent.filter_duplicates(
        results=results,
        final_k=5,
    )

    assert len(filtered) == 2
    assert len(filtered[0].duplicate_sources) == 1
    assert (
        filtered[0].duplicate_sources[0]["pdf_name"]
        == "policy_b.pdf"
    )