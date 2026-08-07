from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from src.models import DocumentChunk, RetrievalResult
from src.utils.text_utils import tokenize


class BM25RetrievalAgent:
    """Retrieve chunks using lexical keyword matching."""

    def __init__(self) -> None:
        self.index: BM25Okapi | None = None
        self.chunks: list[DocumentChunk] = []

    def build_index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "Cannot build BM25 index without chunks."
            )

        self.chunks = chunks

        tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self.index = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        if self.index is None:
            raise RuntimeError(
                "BM25 index has not been initialized."
            )

        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)

        top_indexes = np.argsort(scores)[::-1][
            : min(top_k, len(self.chunks))
        ]

        results: list[RetrievalResult] = []

        for rank, index in enumerate(top_indexes, start=1):
            score = float(scores[index])

            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(index)],
                    score=score,
                    rank=rank,
                    retriever="bm25",
                    bm25_score=score,
                )
            )

        return results