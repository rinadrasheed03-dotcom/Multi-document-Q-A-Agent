from __future__ import annotations

import re
from typing import Any

from src.ingestion.document_loader import load_document
from src.utils.language_detector import LanguageDetector


class PDFProcessingAgent:
    """
    Multi-datatype document processing agent.

    Supported document types:

        PDF
        DOCX
        TXT
        CSV
        XLSX
        PPTX

    Main responsibilities:

        1. Load document content.
        2. Clean extracted text.
        3. Split text into useful chunks.
        4. Detect language for each chunk.
        5. Preserve source/provenance metadata.
        6. Generate unique chunk IDs.

    Each returned chunk contains:

        text
        source_file
        source_type
        language
        language_code
        language_confidence
        page_number
        slide_number
        sheet_name
        row_number
        paragraph_number
        line_start
        line_end
        chunk_id
        source_location
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Multilingual language detector.
        #
        # This uses lingua-language-detector and does not
        # require FastText or C++ compilation.
        self.language_detector = LanguageDetector()

    # ==========================================================
    # PROCESS FILES
    # ==========================================================

    def process_files(
        self,
        uploaded_files,
    ) -> list[dict[str, Any]]:

        """
        Process multiple uploaded documents.

        Parameters
        ----------
        uploaded_files:
            Streamlit UploadedFile objects.

        Returns
        -------
        list[dict]
            Source-aware document chunks.
        """

        all_chunks: list[dict[str, Any]] = []

        for uploaded_file in uploaded_files:

            try:

                file_bytes = uploaded_file.getvalue()

                file_name = uploaded_file.name

                documents = load_document(
                    file_bytes=file_bytes,
                    file_name=file_name,
                )

                file_chunks = self._create_chunks(
                    documents=documents,
                    file_name=file_name,
                )

                all_chunks.extend(
                    file_chunks
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Failed to process "
                    f"'{uploaded_file.name}': {exc}"
                ) from exc

        return all_chunks

    # ==========================================================
    # CREATE CHUNKS
    # ==========================================================

    def _create_chunks(
        self,
        documents: list[dict[str, Any]],
        file_name: str = "",
    ) -> list[dict[str, Any]]:

        chunks: list[dict[str, Any]] = []

        for document_index, document in enumerate(
            documents
        ):

            # --------------------------------------------------
            # Extract text
            # --------------------------------------------------

            original_text = str(
                document.get(
                    "text",
                    "",
                )
            )

            if not original_text.strip():
                continue

            text = self._clean_text(
                original_text
            )

            if not text:
                continue

            # --------------------------------------------------
            # Source metadata
            # --------------------------------------------------

            source_file = document.get(
                "source_file",
                file_name or "unknown",
            )

            source_type = document.get(
                "source_type",
                self._detect_source_type(
                    source_file
                ),
            )

            page_number = document.get(
                "page_number"
            )

            slide_number = document.get(
                "slide_number"
            )

            sheet_name = document.get(
                "sheet_name"
            )

            row_number = document.get(
                "row_number"
            )

            paragraph_number = document.get(
                "paragraph_number"
            )

            line_start = document.get(
                "line_start"
            )

            line_end = document.get(
                "line_end"
            )

            # --------------------------------------------------
            # Split text
            # --------------------------------------------------

            text_chunks = self._split_text(
                text
            )

            # --------------------------------------------------
            # Create source-aware chunks
            # --------------------------------------------------

            for chunk_index, chunk_text in enumerate(
                text_chunks
            ):

                if not chunk_text.strip():
                    continue

                # --------------------------------------------------
                # Language detection
                # --------------------------------------------------

                language_result = (
                    self.language_detector.detect(
                        chunk_text
                    )
                )

                # --------------------------------------------------
                # Unique chunk ID
                # --------------------------------------------------

                chunk_id = self._create_chunk_id(
                    source_file=source_file,
                    document_index=document_index,
                    chunk_index=chunk_index,
                )

                # --------------------------------------------------
                # Source location
                # --------------------------------------------------

                source_location = (
                    self._build_source_location(
                        source_type=source_type,
                        page_number=page_number,
                        slide_number=slide_number,
                        sheet_name=sheet_name,
                        row_number=row_number,
                        paragraph_number=paragraph_number,
                        line_start=line_start,
                        line_end=line_end,
                    )
                )

                # --------------------------------------------------
                # Final chunk
                # --------------------------------------------------

                chunk: dict[str, Any] = {
                    "text": chunk_text,

                    "source_file": source_file,

                    "source_type": source_type,

                    "chunk_id": chunk_id,

                    # Language
                    "language": language_result[
                        "language"
                    ],

                    "language_code": language_result[
                        "language_code"
                    ],

                    "language_confidence": language_result[
                        "confidence"
                    ],

                    # PDF
                    "page_number": page_number,

                    # PowerPoint
                    "slide_number": slide_number,

                    # Excel / CSV
                    "sheet_name": sheet_name,

                    "row_number": row_number,

                    # DOCX
                    "paragraph_number": paragraph_number,

                    # TXT
                    "line_start": line_start,

                    "line_end": line_end,

                    # Human-readable source
                    "source_location": source_location,
                }

                chunks.append(
                    chunk
                )

        return chunks

    # ==========================================================
    # CREATE CHUNK ID
    # ==========================================================

    @staticmethod
    def _create_chunk_id(
        source_file: str,
        document_index: int,
        chunk_index: int,
    ) -> str:

        safe_file_name = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "_",
            source_file,
        )

        return (
            f"{safe_file_name}_"
            f"{document_index}_"
            f"{chunk_index}"
        )

    # ==========================================================
    # DETECT FILE TYPE
    # ==========================================================

    @staticmethod
    def _detect_source_type(
        file_name: str,
    ) -> str:

        extension = (
            file_name
            .lower()
            .split(".")[-1]
        )

        mapping = {
            "pdf": "PDF",
            "docx": "DOCX",
            "txt": "TXT",
            "csv": "CSV",
            "xlsx": "XLSX",
            "xls": "XLSX",
            "pptx": "PPTX",
            "ppt": "PPTX",
        }

        return mapping.get(
            extension,
            extension.upper(),
        )

    # ==========================================================
    # BUILD SOURCE LOCATION
    # ==========================================================

    @staticmethod
    def _build_source_location(
        source_type: str,
        page_number=None,
        slide_number=None,
        sheet_name=None,
        row_number=None,
        paragraph_number=None,
        line_start=None,
        line_end=None,
    ) -> str:

        source_type = (
            source_type or ""
        ).upper()

        # ------------------------------------------------------
        # PDF
        # ------------------------------------------------------

        if source_type == "PDF":

            if page_number is not None:

                return (
                    f"Page {page_number}"
                )

            return "PDF document"

        # ------------------------------------------------------
        # PPTX
        # ------------------------------------------------------

        if source_type == "PPTX":

            if slide_number is not None:

                return (
                    f"Slide {slide_number}"
                )

            return "PowerPoint presentation"

        # ------------------------------------------------------
        # XLSX / CSV
        # ------------------------------------------------------

        if source_type in {
            "XLSX",
            "CSV",
        }:

            parts = []

            if sheet_name:
                parts.append(
                    f"Sheet: {sheet_name}"
                )

            if row_number is not None:
                parts.append(
                    f"Row: {row_number}"
                )

            if parts:
                return " | ".join(parts)

            return "Tabular data"

        # ------------------------------------------------------
        # DOCX
        # ------------------------------------------------------

        if source_type == "DOCX":

            if paragraph_number is not None:

                return (
                    f"Paragraph "
                    f"{paragraph_number}"
                )

            return "Word document"

        # ------------------------------------------------------
        # TXT
        # ------------------------------------------------------

        if source_type == "TXT":

            if (
                line_start is not None
                and line_end is not None
            ):

                return (
                    f"Lines "
                    f"{line_start}-{line_end}"
                )

            if line_start is not None:

                return (
                    f"Line {line_start}"
                )

            return "Text file"

        return "Document"

    # ==========================================================
    # CLEAN TEXT
    # ==========================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        # Normalize line endings.
        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # Normalize tabs.
        text = text.replace(
            "\t",
            " ",
        )

        # Remove excessive spaces.
        text = re.sub(
            r"[ ]{2,}",
            " ",
            text,
        )

        # Preserve paragraph boundaries.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ==========================================================
    # SMART TEXT SPLITTING
    # ==========================================================

    def _split_text(
        self,
        text: str,
    ) -> list[str]:

        # Small text does not need splitting.
        if len(text) <= self.chunk_size:

            return [text]

        # ------------------------------------------------------
        # First split by paragraphs.
        # ------------------------------------------------------

        paragraphs = re.split(
            r"\n\s*\n",
            text,
        )

        chunks: list[str] = []

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            candidate = (
                f"{current_chunk}\n\n{paragraph}"
                if current_chunk
                else paragraph
            )

            # --------------------------------------------------
            # Paragraph fits.
            # --------------------------------------------------

            if len(candidate) <= self.chunk_size:

                current_chunk = candidate

                continue

            # --------------------------------------------------
            # Store previous chunk.
            # --------------------------------------------------

            if current_chunk:

                chunks.append(
                    current_chunk.strip()
                )

            # --------------------------------------------------
            # Large paragraph.
            # --------------------------------------------------

            if len(paragraph) > self.chunk_size:

                paragraph_chunks = (
                    self._split_large_text(
                        paragraph
                    )
                )

                if paragraph_chunks:

                    chunks.extend(
                        paragraph_chunks[:-1]
                    )

                    current_chunk = (
                        paragraph_chunks[-1]
                    )

                else:

                    current_chunk = ""

            else:

                current_chunk = paragraph

        # ------------------------------------------------------
        # Store final chunk.
        # ------------------------------------------------------

        if current_chunk:

            chunks.append(
                current_chunk.strip()
            )

        # ------------------------------------------------------
        # Add overlap.
        # ------------------------------------------------------

        if self.chunk_overlap <= 0:

            return chunks

        return self._add_overlap(
            chunks
        )

    # ==========================================================
    # LARGE PARAGRAPH SPLITTER
    # ==========================================================

    def _split_large_text(
        self,
        text: str,
    ) -> list[str]:

        # Sentence splitting supports many languages
        # reasonably well, while keeping punctuation.
        sentences = re.split(
            r"(?<=[.!?。！？؟])\s+",
            text,
        )

        chunks: list[str] = []

        current = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            candidate = (
                f"{current} {sentence}"
                if current
                else sentence
            )

            if len(candidate) <= self.chunk_size:

                current = candidate

                continue

            # --------------------------------------------------
            # Store current chunk.
            # --------------------------------------------------

            if current:

                chunks.append(
                    current.strip()
                )

            # --------------------------------------------------
            # Sentence itself is too large.
            # --------------------------------------------------

            if len(sentence) > self.chunk_size:

                word_chunks = (
                    self._split_by_words(
                        sentence
                    )
                )

                if word_chunks:

                    chunks.extend(
                        word_chunks[:-1]
                    )

                    current = (
                        word_chunks[-1]
                    )

                else:

                    current = ""

            else:

                current = sentence

        if current:

            chunks.append(
                current.strip()
            )

        return chunks

    # ==========================================================
    # WORD-BASED FALLBACK
    # ==========================================================

    def _split_by_words(
        self,
        text: str,
    ) -> list[str]:

        words = text.split()

        chunks: list[str] = []

        current_words: list[str] = []

        current_length = 0

        for word in words:

            additional_length = (
                len(word)
                + (
                    1
                    if current_words
                    else 0
                )
            )

            if (
                current_length
                + additional_length
                <= self.chunk_size
            ):

                current_words.append(
                    word
                )

                current_length += (
                    additional_length
                )

            else:

                if current_words:

                    chunks.append(
                        " ".join(
                            current_words
                        )
                    )

                current_words = [
                    word
                ]

                current_length = len(word)

        if current_words:

            chunks.append(
                " ".join(
                    current_words
                )
            )

        return chunks

    # ==========================================================
    # ADD OVERLAP
    # ==========================================================

    def _add_overlap(
        self,
        chunks: list[str],
    ) -> list[str]:

        if len(chunks) <= 1:

            return chunks

        result = [
            chunks[0]
        ]

        for index in range(
            1,
            len(chunks),
        ):

            previous_chunk = chunks[
                index - 1
            ]

            current_chunk = chunks[
                index
            ]

            # Take overlap from the previous chunk.
            overlap_text = (
                previous_chunk[
                    -self.chunk_overlap:
                ]
            )

            combined = (
                overlap_text
                + "\n\n"
                + current_chunk
            )

            # Prevent oversized chunks.
            if len(combined) > (
                self.chunk_size
                + self.chunk_overlap
            ):

                combined = current_chunk

            result.append(
                combined
            )

        return result