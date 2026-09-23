from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from src.agents.pdf_processing_agent import PDFProcessingAgent
from src.agents.question_generation_agent import (
    QuestionGenerationAgent,
)
from src.agents.answer_validation_agent import (
    AnswerValidationAgent,
)
from src.config import settings


st.set_page_config(
    page_title="AI Question Generator",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:

    defaults = {
        "chunks": [],
        "indexed_files": [],
        "questions": [],
        "processing_complete": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_documents(uploaded_files):

    pdf_agent = PDFProcessingAgent(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = pdf_agent.process_files(uploaded_files)

    if not chunks:
        raise ValueError(
            "No readable text was extracted from the documents."
        )

    return chunks


# ============================================================
# QUESTION GENERATION
# ============================================================

def generate_questions(
    chunks,
    num_questions,
    question_types,
    difficulty,
):

    generator = QuestionGenerationAgent(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        model_name=settings.nvidia_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    validator = AnswerValidationAgent(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        model_name=settings.nvidia_model,
    )

    all_questions = []

    # ----------------------------------------------------------
    # TARGET
    # ----------------------------------------------------------

    target_questions = num_questions

    # Maximum number of generation rounds.
    # This prevents an infinite loop if the model keeps
    # producing invalid questions.
    max_rounds = 10

    # Number of chunks used in each generation request.
    batch_size = 5

    round_number = 0

    total_generated = 0
    total_validated = 0
    total_rejected = 0

    # ----------------------------------------------------------
    # Continue until target number is reached
    # ----------------------------------------------------------

    while (
        len(all_questions) < target_questions
        and round_number < max_rounds
    ):

        round_number += 1

        print("\n" + "=" * 80)
        print(
            f"QUESTION GENERATION ROUND "
            f"{round_number}"
        )
        print("=" * 80)

        # ------------------------------------------------------
        # Shuffle/rotate starting point so later rounds don't
        # repeatedly use exactly the same chunks.
        # ------------------------------------------------------

        start_offset = (
            (round_number - 1) * batch_size
        ) % max(
            len(chunks),
            1,
        )

        ordered_chunks = (
            chunks[start_offset:]
            + chunks[:start_offset]
        )

        # ------------------------------------------------------
        # Generate more than needed.
        #
        # Example:
        # Need 7 more → ask for up to 3 per batch.
        # ------------------------------------------------------

        remaining = (
            target_questions
            - len(all_questions)
        )

        questions_per_batch = min(
            3,
            max(
                1,
                remaining,
            ),
        )

        # ------------------------------------------------------
        # Process chunks
        # ------------------------------------------------------

        for batch_start in range(
            0,
            len(ordered_chunks),
            batch_size,
        ):

            if len(all_questions) >= target_questions:
                break

            batch = ordered_chunks[
                batch_start:
                batch_start + batch_size
            ]

            context_parts = []

            source_file = ""
            page_number = None

            # --------------------------------------------------
            # Build context
            # --------------------------------------------------

            for chunk in batch:

                text = getattr(
                    chunk,
                    "text",
                    None,
                )

                if text is None and isinstance(
                    chunk,
                    dict,
                ):
                    text = chunk.get(
                        "text",
                        chunk.get(
                            "content",
                            "",
                        ),
                    )

                if not text:
                    continue

                context_parts.append(
                    str(text)
                )

                # ----------------------------------------------
                # Source information
                # ----------------------------------------------

                if isinstance(
                    chunk,
                    dict,
                ):

                    source_file = chunk.get(
                        "source_file",
                        chunk.get(
                            "pdf_name",
                            source_file,
                        ),
                    )

                    page_number = chunk.get(
                        "page_number",
                        page_number,
                    )

                else:

                    source_file = getattr(
                        chunk,
                        "source_file",
                        source_file,
                    )

                    page_number = getattr(
                        chunk,
                        "page_number",
                        page_number,
                    )

            context = "\n\n".join(
                context_parts
            )

            if not context.strip():
                continue

            # --------------------------------------------------
            # Generate
            # --------------------------------------------------

            try:

                questions = generator.generate(
                    context=context,
                    num_questions=questions_per_batch,
                    question_types=question_types,
                    difficulty=difficulty,
                    source_file=source_file,
                    page_number=page_number,
                )

            except Exception as exc:

                print(
                    "\nQUESTION GENERATION ERROR:"
                )

                print(
                    type(exc).__name__,
                    str(exc),
                )

                continue

            total_generated += len(
                questions
            )

            print(
                f"Generated {len(questions)} "
                f"question(s)."
            )

            # --------------------------------------------------
            # Validate
            # --------------------------------------------------

            for question in questions:

                if len(all_questions) >= target_questions:
                    break

                try:

                    supported = (
                        validator.validate(
                            question=(
                                question.question
                            ),
                            answer=(
                                question.answer
                            ),
                            evidence=(
                                question.evidence
                            ),
                        )
                    )

                except Exception as exc:

                    print(
                        "VALIDATION ERROR:",
                        type(exc).__name__,
                        str(exc),
                    )

                    supported = False

                if supported:

                    total_validated += 1

                    all_questions.append(
                        question.to_dict()
                    )

                    print(
                        f"✓ Accepted "
                        f"({len(all_questions)}/"
                        f"{target_questions})"
                    )

                else:

                    total_rejected += 1

                    print(
                        "✗ Question rejected "
                        "by validation."
                    )

        # ------------------------------------------------------
        # Check whether target has been reached
        # ------------------------------------------------------

        if len(all_questions) >= target_questions:

            break

    # ----------------------------------------------------------
    # Final result
    # ----------------------------------------------------------

    print("\n" + "=" * 80)
    print("QUESTION GENERATION SUMMARY")
    print("=" * 80)

    print(
        "Requested:",
        target_questions,
    )

    print(
        "Generated:",
        total_generated,
    )

    print(
        "Validated:",
        total_validated,
    )

    print(
        "Rejected:",
        total_rejected,
    )

    print(
        "Final:",
        len(all_questions),
    )

    print("=" * 80)

    return all_questions[
        :target_questions
    ]


# ============================================================
# SIDEBAR
# ============================================================

def display_sidebar():

    with st.sidebar:

        st.header("📚 Document Upload")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[
                "pdf",
                "docx",
                "txt",
                "csv",
                "xlsx",
                "pptx",
            ],
            accept_multiple_files=True,
            help=(
                "Supported formats: PDF, DOCX, TXT, CSV, XLSX, PPTX. "
                "The system will extract text from these documents "
                "to generate questions."
            )
        )

        st.divider()

        st.subheader("Question Settings")

        num_questions = st.slider(
            "Number of questions",
            min_value=1,
            max_value=100,
            value=10,
        )

        question_types = ["brief_answer"]

        difficulty = st.selectbox(
            "Difficulty",
            [
                "easy",
                "medium",
                "hard",
                "mixed",
            ],
            index=3,
        )

        st.divider()

        process_clicked = st.button(
            "📄 Process Documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        if process_clicked:

            try:

                with st.spinner(
                    "Reading documents..."
                ):

                    chunks = process_documents(
                        uploaded_files
                    )

                st.session_state.chunks = chunks

                st.session_state.indexed_files = [
                    file.name
                    for file in uploaded_files
                ]

                st.session_state.processing_complete = True

                st.session_state.questions = []

                st.success(
                    f"Processed "
                    f"{len(uploaded_files)} document(s)."
                )

            except Exception as exc:

                st.session_state.processing_complete = False

                st.error(str(exc))

        if st.session_state.processing_complete:

            st.divider()

            generate_clicked = st.button(
                "🧠 Generate Questions",
                type="primary",
                use_container_width=True,
            )

            if generate_clicked:

                if not question_types:

                    st.warning(
                        "Select at least one question type."
                    )

                else:

                    try:

                        with st.spinner(
                            "Generating and validating questions..."
                        ):

                            questions = generate_questions(
                                chunks=st.session_state.chunks,
                                num_questions=num_questions,
                                question_types=question_types,
                                difficulty=difficulty,
                            )

                        st.session_state.questions = questions

                        if questions:

                            st.success(
                                f"Generated "
                                f"{len(questions)} questions."
                            )

                        else:

                            st.warning(
                                "No valid questions were generated."
                            )

                    except Exception as exc:

                        st.error(
                            f"Question generation failed: {exc}"
                        )

        st.divider()

        if st.button(
            "Clear Session",
            use_container_width=True,
        ):

            st.session_state.chunks = []
            st.session_state.indexed_files = []
            st.session_state.questions = []
            st.session_state.processing_complete = False

            st.rerun()


# ============================================================
# DISPLAY QUESTION
# ============================================================

def display_question(
    index: int,
    question: dict,
):

    st.markdown(
        f"### Q{index}. {question['question']}"
    )

    question_type = question.get(
        "question_type",
        "short_answer",
    )

    difficulty = question.get(
        "difficulty",
        "medium",
    )

    st.caption(
        f"Type: {question_type} | "
        f"Difficulty: {difficulty}"
    )

    options = question.get("options")

    if options:

        for option in options:

            st.write(
                f"☐ {option}"
            )

    with st.expander("Show Answer"):

        st.success(
            question["answer"]
        )

        if question.get("evidence"):

            st.markdown(
                "**Evidence from document:**"
            )

            st.info(
                question["evidence"]
            )

    source_file = question.get(
        "source_file",
        "",
    )

    page_number = question.get(
        "page_number",
        None,
    )

    if source_file:

        source_text = f"📄 {source_file}"

        if page_number:

            source_text += (
                f" — Page {page_number}"
            )

        st.caption(source_text)

    st.divider()


# ============================================================
# EXPORT
# ============================================================

def create_excel_download(
    questions: list[dict],
) -> bytes:

    dataframe = pd.DataFrame(
        questions
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Questions",
        )

    return output.getvalue()


def create_json_download(
    questions: list[dict],
) -> bytes:

    return json.dumps(
        questions,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")


# ============================================================
# MAIN
# ============================================================

def main():

    initialize_session_state()

    st.title(
        "🧠 AI Question & Answer Generator"
    )

    st.caption(
        "Generate grounded questions and answers "
        "directly from uploaded documents."
    )

    display_sidebar()

    if not st.session_state.processing_complete:

        st.info(
            "Upload a document from the sidebar "
            "to begin."
        )

        return

    # --------------------------------------------------------
    # DOCUMENT SUMMARY
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Documents",
            len(
                st.session_state.indexed_files
            ),
        )

    with col2:

        st.metric(
            "Document chunks",
            len(
                st.session_state.chunks
            ),
        )

    with col3:

        st.metric(
            "Questions",
            len(
                st.session_state.questions
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------------

    with st.expander(
        "📄 Uploaded Documents"
    ):

        for file_name in (
            st.session_state.indexed_files
        ):

            st.write(
                f"• {file_name}"
            )

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    questions = (
        st.session_state.questions
    )

    if not questions:

        st.info(
            "No questions generated yet. "
            "Use 'Generate Questions' in the sidebar."
        )

        return

    st.header(
        "Generated Questions & Answers"
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "🔎 Search generated questions"
    )

    filtered_questions = questions

    if search.strip():

        search_lower = search.lower()

        filtered_questions = [
            q
            for q in questions
            if (
                search_lower
                in q["question"].lower()
            )
            or (
                search_lower
                in q["answer"].lower()
            )
        ]

    st.caption(
        f"Showing "
        f"{len(filtered_questions)} "
        f"question(s)"
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    for index, question in enumerate(
        filtered_questions,
        start=1,
    ):

        display_question(
            index=index,
            question=question,
        )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    st.header(
        "⬇️ Export"
    )

    col1, col2 = st.columns(2)

    with col1:

        excel_data = create_excel_download(
            questions
        )

        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name="generated_questions.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    with col2:

        json_data = create_json_download(
            questions
        )

        st.download_button(
            label="🗂️ Download JSON",
            data=json_data,
            file_name="generated_questions.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()