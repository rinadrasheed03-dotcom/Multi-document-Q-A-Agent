from __future__ import annotations

import streamlit as st

from src.agents.answer_agent import AnswerAgent
from src.agents.bm25_retrieval_agent import BM25RetrievalAgent
from src.agents.duplicate_aware_agent import DuplicateAwareAgent
from src.agents.pdf_processing_agent import PDFProcessingAgent
from src.agents.reranking_agent import RerankingAgent
from src.agents.rrf_fusion_agent import RRFFusionAgent
from src.agents.vector_retrieval_agent import (
    VectorRetrievalAgent,
)
from src.config import settings
from src.graph.nodes import RAGNodes
from src.graph.workflow import create_rag_workflow
from src.ingestion.embeddings import EmbeddingService
from src.utils.text_utils import truncate_text


st.set_page_config(
    page_title="Multi-PDF RAG Agent",
    page_icon="📚",
    layout="wide",
)


def initialize_session_state() -> None:
    defaults = {
        "workflow": None,
        "chunks": [],
        "indexed_files": [],
        "messages": [],
        "processing_complete": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_service() -> EmbeddingService:
    return EmbeddingService(settings.embedding_model)


@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranking_agent() -> RerankingAgent:
    return RerankingAgent(settings.reranker_model)


def build_rag_system(uploaded_files) -> tuple:
    pdf_agent = PDFProcessingAgent(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = pdf_agent.process_files(uploaded_files)

    if not chunks:
        raise ValueError(
            "No readable text was extracted from the PDFs."
        )

    embedding_service = load_embedding_service()

    vector_agent = VectorRetrievalAgent(
        embedding_service=embedding_service,
    )
    vector_agent.build_index(chunks)

    bm25_agent = BM25RetrievalAgent()
    bm25_agent.build_index(chunks)

    rrf_agent = RRFFusionAgent(
        constant=settings.rrf_constant,
    )

    reranking_agent = load_reranking_agent()

    duplicate_agent = DuplicateAwareAgent(
        embedding_service=embedding_service,
        similarity_threshold=settings.duplicate_threshold,
    )

    answer_agent = AnswerAgent(
        api_key=settings.nvidia_api_key,
        model_name=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    nodes = RAGNodes(
        vector_agent=vector_agent,
        bm25_agent=bm25_agent,
        rrf_agent=rrf_agent,
        reranking_agent=reranking_agent,
        duplicate_agent=duplicate_agent,
        answer_agent=answer_agent,
        vector_top_k=settings.vector_top_k,
        bm25_top_k=settings.bm25_top_k,
        rrf_top_k=settings.rrf_top_k,
        rerank_top_k=settings.rerank_top_k,
        final_context_k=settings.final_context_k,
    )

    workflow = create_rag_workflow(nodes)

    return workflow, chunks


def display_sidebar() -> None:
    with st.sidebar:
        st.header("Document Upload")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )

        process_clicked = st.button(
            "Process Documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        if process_clicked:
            try:
                with st.spinner(
                    "Processing PDFs and building indexes..."
                ):
                    workflow, chunks = build_rag_system(
                        uploaded_files
                    )

                st.session_state.workflow = workflow
                st.session_state.chunks = chunks
                st.session_state.indexed_files = [
                    uploaded_file.name
                    for uploaded_file in uploaded_files
                ]
                st.session_state.processing_complete = True
                st.session_state.messages = []

                st.success(
                    f"Processed {len(uploaded_files)} PDFs "
                    f"into {len(chunks)} chunks."
                )

            except Exception as exc:
                st.session_state.processing_complete = False
                st.error(str(exc))

        st.divider()

        st.subheader("Retrieval Pipeline")

        st.markdown(
            """
1. Vector Retrieval Agent  
2. BM25 Retrieval Agent  
3. RRF Fusion Agent  
4. Reranking Agent  
5. Duplicate-Aware Agent  
6. Answer Agent
"""
        )

        if st.session_state.indexed_files:
            st.divider()
            st.subheader("Indexed PDFs")

            for file_name in st.session_state.indexed_files:
                st.write(f"• {file_name}")

        if st.button(
            "Clear Session",
            use_container_width=True,
        ):
            st.session_state.workflow = None
            st.session_state.chunks = []
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.session_state.processing_complete = False
            st.rerun()


def display_metrics(metrics: dict) -> None:
    with st.expander("Retrieval metrics"):
        metric_columns = st.columns(4)

        metric_columns[0].metric(
            "Vector candidates",
            metrics.get("vector_candidates", 0),
        )

        metric_columns[1].metric(
            "BM25 candidates",
            metrics.get("bm25_candidates", 0),
        )

        metric_columns[2].metric(
            "RRF candidates",
            metrics.get("rrf_candidates", 0),
        )

        metric_columns[3].metric(
            "Final context",
            metrics.get("final_context_chunks", 0),
        )

        latency_columns = st.columns(4)

        latency_columns[0].metric(
            "Retrieval latency",
            f"{metrics.get('total_retrieval_ms', 0):.2f} ms",
        )

        latency_columns[1].metric(
            "Reranking latency",
            f"{metrics.get('reranking_ms', 0):.2f} ms",
        )

        latency_columns[2].metric(
            "LLM latency",
            f"{metrics.get('answer_generation_ms', 0):.2f} ms",
        )

        latency_columns[3].metric(
            "Duplicates grouped",
            metrics.get("duplicates_grouped", 0),
        )

        st.json(metrics)


def display_sources(sources: list[dict]) -> None:
    with st.expander("Sources"):
        if not sources:
            st.info("No sources were returned.")
            return

        for index, source in enumerate(sources, start=1):
            pdf_name = source["pdf_name"]
            page_number = source["page_number"]

            st.markdown(
                f"**{index}. {pdf_name} — Page {page_number}**"
            )

            if source.get("duplicate_of"):
                st.caption(
                    "This source contains content similar to "
                    "another selected evidence chunk."
                )


def display_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if message.get("sources"):
                    display_sources(message["sources"])

                if message.get("metrics"):
                    display_metrics(message["metrics"])


def main() -> None:
    initialize_session_state()

    st.title("📚 Multi-PDF RAG Agent")

    st.caption(
        "Hybrid retrieval using Vector Search, BM25, "
        "Reciprocal Rank Fusion, reranking and "
        "duplicate-aware context selection."
    )

    display_sidebar()

    if not settings.nvidia_api_key:
        st.warning(
            "NVIDIA_API_KEY is not configured. "
            "Add the key to your .env file before asking questions."
        )

    if not st.session_state.processing_complete:
        st.info(
            "Upload PDF documents from the sidebar and select "
            "'Process Documents' to begin."
        )
        return

    document_count = len(st.session_state.indexed_files)
    chunk_count = len(st.session_state.chunks)

    summary_columns = st.columns(2)

    summary_columns[0].metric(
        "Indexed PDFs",
        document_count,
    )

    summary_columns[1].metric(
        "Document chunks",
        chunk_count,
    )

    display_chat_history()

    query = st.chat_input(
        "Ask a question across the uploaded PDFs"
    )

    if not query:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Retrieving, reranking and generating answer..."
            ):
                result = st.session_state.workflow.invoke(
                    {
                        "query": query,
                        "metrics": {},
                    }
                )

            answer = result.get(
                "answer",
                "No answer was generated.",
            )
            sources = result.get("sources", [])
            metrics = result.get("metrics", {})

            st.markdown(answer)
            display_sources(sources)
            display_metrics(metrics)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metrics": metrics,
                }
            )

        except Exception as exc:
            error_message = f"RAG workflow failed: {exc}"
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )


if __name__ == "__main__":
    main()