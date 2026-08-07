from __future__ import annotations

import faiss
import numpy as np

from src.ingestion.embeddings import EmbeddingService
from src.models import DocumentChunk, RetrievalResult


class VectorRetrievalAgent:
    """Retrieve semantically similar chunks using FAISS."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
    ) -> None:
        self.embedding_service = embedding_service
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def build_index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "Cannot build vector index without chunks."
            )

        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]

        self.embeddings = (
            self.embedding_service.embed_documents(texts)
        )

        vector_dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(vector_dimension)
        self.index.add(self.embeddings)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        if self.index is None:
            raise RuntimeError(
                "Vector index has not been initialized."
            )

        query_embedding = (
            self.embedding_service.embed_query(query)
        )

        result_count = min(top_k, len(self.chunks))

        scores, indexes = self.index.search(
            query_embedding,
            result_count,
        )

        results: list[RetrievalResult] = []

        for rank, (score, index) in enumerate(
            zip(scores[0], indexes[0]),
            start=1,
        ):
            if index < 0:
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(index)],
                    score=float(score),
                    rank=rank,
                    retriever="vector",
                    vector_score=float(score),
                )
            )

        return results