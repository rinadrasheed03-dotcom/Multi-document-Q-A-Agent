from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import RAGNodes
from src.graph.state import RAGState


def create_rag_workflow(nodes: RAGNodes):
    """Create and compile the LangGraph RAG workflow."""

    workflow = StateGraph(RAGState)

    workflow.add_node(
        "vector_retrieval_agent",
        nodes.vector_retrieval_node,
    )
    workflow.add_node(
        "bm25_retrieval_agent",
        nodes.bm25_retrieval_node,
    )
    workflow.add_node(
        "rrf_fusion_agent",
        nodes.rrf_fusion_node,
    )
    workflow.add_node(
        "reranking_agent",
        nodes.reranking_node,
    )
    workflow.add_node(
        "duplicate_aware_agent",
        nodes.duplicate_filter_node,
    )
    workflow.add_node(
        "answer_agent",
        nodes.answer_node,
    )

    workflow.add_edge(
        START,
        "vector_retrieval_agent",
    )
    workflow.add_edge(
        "vector_retrieval_agent",
        "bm25_retrieval_agent",
    )
    workflow.add_edge(
        "bm25_retrieval_agent",
        "rrf_fusion_agent",
    )
    workflow.add_edge(
        "rrf_fusion_agent",
        "reranking_agent",
    )
    workflow.add_edge(
        "reranking_agent",
        "duplicate_aware_agent",
    )
    workflow.add_edge(
        "duplicate_aware_agent",
        "answer_agent",
    )
    workflow.add_edge(
        "answer_agent",
        END,
    )

    return workflow.compile()