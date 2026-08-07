from __future__ import annotations

from typing import Protocol

from src.ingestion.chunker import TextChunker
from src.ingestion.pdf_loader import PDFLoader
from src.models import DocumentChunk


class UploadedFileLike(Protocol):
    """Protocol representing a Streamlit uploaded file."""

    name: str

    def getvalue(self) -> bytes:
        ...


class PDFProcessingAgent:
    """Extract and chunk text from multiple uploaded PDFs."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.loader = PDFLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_files(
        self,
        uploaded_files: list[UploadedFileLike],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()

            pages = self.loader.load_pdf(
                file_name=uploaded_file.name,
                file_bytes=file_bytes,
            )

            file_chunks = self.chunker.split_pages(pages)
            chunks.extend(file_chunks)

        return chunks