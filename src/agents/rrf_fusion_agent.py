from __future__ import annotations

from copy import deepcopy

from src.models import RetrievalResult


class RRFFusionAgent:
    """Fuse vector and BM25 result rankings using RRF."""

    def __init__(
        self,
        constant: int = 60,
    ) -> None:
        self.constant = constant

    def fuse(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        fused: dict[str, RetrievalResult] = {}
        rrf_scores: dict[str, float] = {}

        result_lists = [
            vector_results,
            bm25_results,
        ]

        for result_list in result_lists:
            for result in result_list:
                chunk_id = result.chunk.chunk_id

                contribution = 1.0 / (
                    self.constant + result.rank
                )

                rrf_scores[chunk_id] = (
                    rrf_scores.get(chunk_id, 0.0)
                    + contribution
                )

                if chunk_id not in fused:
                    fused[chunk_id] = deepcopy(result)
                    fused[chunk_id].retriever = "rrf"
                else:
                    existing = fused[chunk_id]

                    if result.vector_score is not None:
                        existing.vector_score = (
                            result.vector_score
                        )

                    if result.bm25_score is not None:
                        existing.bm25_score = (
                            result.bm25_score
                        )

        sorted_items = sorted(
            fused.items(),
            key=lambda item: rrf_scores[item[0]],
            reverse=True,
        )

        final_results: list[RetrievalResult] = []

        for rank, (chunk_id, result) in enumerate(
            sorted_items[:top_k],
            start=1,
        ):
            result.rank = rank
            result.score = rrf_scores[chunk_id]
            result.rrf_score = rrf_scores[chunk_id]

            final_results.append(result)

        return final_results