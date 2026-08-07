from __future__ import annotations

import numpy as np

from src.ingestion.embeddings import EmbeddingService
from src.models import RetrievalResult
from src.utils.text_utils import text_hash


class DuplicateAwareAgent:
    """
    Remove redundant chunks while preserving duplicate provenance.

    Exact duplicates are detected through normalized text hashes.
    Semantic duplicates are detected through embedding similarity.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.94,
    ) -> None:
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    def filter_duplicates(
        self,
        results: list[RetrievalResult],
        final_k: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        texts = [result.chunk.text for result in results]

        embeddings = (
            self.embedding_service.embed_documents(texts)
        )

        selected_results: list[RetrievalResult] = []
        selected_embeddings: list[np.ndarray] = []
        selected_hashes: list[str] = []

        for result, embedding in zip(results, embeddings):
            current_hash = text_hash(result.chunk.text)

            duplicate_index = self._find_duplicate(
                current_hash=current_hash,
                current_embedding=embedding,
                selected_hashes=selected_hashes,
                selected_embeddings=selected_embeddings,
            )

            if duplicate_index is not None:
                representative = selected_results[
                    duplicate_index
                ]

                representative.duplicate_sources.append(
                    {
                        "pdf_name": result.chunk.pdf_name,
                        "page_number": result.chunk.page_number,
                        "chunk_id": result.chunk.chunk_id,
                        "text": result.chunk.text,
                    }
                )

                continue

            selected_results.append(result)
            selected_embeddings.append(embedding)
            selected_hashes.append(current_hash)

            if len(selected_results) >= final_k:
                break

        for rank, result in enumerate(
            selected_results,
            start=1,
        ):
            result.rank = rank
            result.retriever = "duplicate-aware"

        return selected_results

    def _find_duplicate(
        self,
        current_hash: str,
        current_embedding: np.ndarray,
        selected_hashes: list[str],
        selected_embeddings: list[np.ndarray],
    ) -> int | None:
        for index, selected_hash in enumerate(selected_hashes):
            if current_hash == selected_hash:
                return index

        for index, selected_embedding in enumerate(
            selected_embeddings
        ):
            similarity = float(
                np.dot(
                    current_embedding,
                    selected_embedding,
                )
            )

            if similarity >= self.similarity_threshold:
                return index

        return None