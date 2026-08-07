from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.models import RetrievalResult


class RerankingAgent:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        query_chunk_pairs = [
            [query, result.chunk.text]
            for result in results
        ]

        scores = self.model.predict(query_chunk_pairs)

        for result, score in zip(results, scores):
            result.reranker_score = float(score)
            result.score = float(score)
            result.retriever = "reranker"

        reranked_results = sorted(
            results,
            key=lambda result: (
                result.reranker_score
                if result.reranker_score is not None
                else float("-inf")
            ),
            reverse=True,
        )

        selected_results = reranked_results[:top_k]

        for rank, result in enumerate(
            selected_results,
            start=1,
        ):
            result.rank = rank

        return selected_results