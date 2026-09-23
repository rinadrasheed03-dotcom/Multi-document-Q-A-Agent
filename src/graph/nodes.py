from __future__ import annotations

import time
from typing import Any

from src.agents.QuestionGenerationAgent import AnswerAgent
from src.agents.bm25_retrieval_agent import BM25RetrievalAgent
from src.agents.duplicate_aware_agent import DuplicateAwareAgent
from src.agents.reranking_agent import RerankingAgent
from src.agents.rrf_fusion_agent import RRFFusionAgent
from src.agents.vector_retrieval_agent import (
    VectorRetrievalAgent,
)
from src.graph.state import RAGState
from src.utils.citation_utils import build_sources


class RAGNodes:
    """LangGraph node functions wrapping retrieval agents."""

    def __init__(
        self,
        vector_agent: VectorRetrievalAgent,
        bm25_agent: BM25RetrievalAgent,
        rrf_agent: RRFFusionAgent,
        reranking_agent: RerankingAgent,
        duplicate_agent: DuplicateAwareAgent,
        answer_agent: AnswerAgent,
        vector_top_k: int,
        bm25_top_k: int,
        rrf_top_k: int,
        rerank_top_k: int,
        final_context_k: int,
    ) -> None:
        self.vector_agent = vector_agent
        self.bm25_agent = bm25_agent
        self.rrf_agent = rrf_agent
        self.reranking_agent = reranking_agent
        self.duplicate_agent = duplicate_agent
        self.answer_agent = answer_agent

        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_top_k = rrf_top_k
        self.rerank_top_k = rerank_top_k
        self.final_context_k = final_context_k

    def vector_retrieval_node(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        results = self.vector_agent.retrieve(
            query=state["query"],
            top_k=self.vector_top_k,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics = dict(state.get("metrics", {}))
        metrics["vector_retrieval_ms"] = elapsed_ms
        metrics["vector_candidates"] = len(results)

        return {
            "vector_results": results,
            "metrics": metrics,
        }

    def bm25_retrieval_node(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        results = self.bm25_agent.retrieve(
            query=state["query"],
            top_k=self.bm25_top_k,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics = dict(state.get("metrics", {}))
        metrics["bm25_retrieval_ms"] = elapsed_ms
        metrics["bm25_candidates"] = len(results)

        return {
            "bm25_results": results,
            "metrics": metrics,
        }

    def rrf_fusion_node(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        results = self.rrf_agent.fuse(
            vector_results=state["vector_results"],
            bm25_results=state["bm25_results"],
            top_k=self.rrf_top_k,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics = dict(state.get("metrics", {}))
        metrics["rrf_fusion_ms"] = elapsed_ms
        metrics["rrf_candidates"] = len(results)

        return {
            "fused_results": results,
            "metrics": metrics,
        }

    def reranking_node(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        results = self.reranking_agent.rerank(
            query=state["query"],
            results=state["fused_results"],
            top_k=self.rerank_top_k,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics = dict(state.get("metrics", {}))
        metrics["reranking_ms"] = elapsed_ms
        metrics["reranked_candidates"] = len(results)

        return {
            "reranked_results": results,
            "metrics": metrics,
        }

    def duplicate_filter_node(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        input_count = len(state["reranked_results"])

        results = self.duplicate_agent.filter_duplicates(
            results=state["reranked_results"],
            final_k=self.final_context_k,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        duplicate_count = sum(
            len(result.duplicate_sources)
            for result in results
        )

        metrics = dict(state.get("metrics", {}))
        metrics["duplicate_filter_ms"] = elapsed_ms
        metrics["duplicate_filter_input"] = input_count
        metrics["final_context_chunks"] = len(results)
        metrics["duplicates_grouped"] = duplicate_count

        return {
            "final_results": results,
            "metrics": metrics,
        }

    def answer_node(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        answer = self.answer_agent.generate_answer(
            query=state["query"],
            results=state["final_results"],
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics = dict(state.get("metrics", {}))
        metrics["answer_generation_ms"] = elapsed_ms

        retrieval_keys = [
            "vector_retrieval_ms",
            "bm25_retrieval_ms",
            "rrf_fusion_ms",
            "reranking_ms",
            "duplicate_filter_ms",
        ]

        metrics["total_retrieval_ms"] = sum(
            float(metrics.get(key, 0.0))
            for key in retrieval_keys
        )

        metrics["total_pipeline_ms"] = (
            metrics["total_retrieval_ms"]
            + metrics["answer_generation_ms"]
        )

        sources = build_sources(state["final_results"])

        return {
            "answer": answer,
            "sources": sources,
            "metrics": metrics,
        }