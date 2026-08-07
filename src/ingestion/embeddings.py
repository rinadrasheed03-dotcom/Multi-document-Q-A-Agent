from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from src.ingestion.pdf_loader import PDFPage
from src.models import DocumentChunk


class TextChunker:
    """Split PDF pages into overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(
        self,
        pages: list[PDFPage],
    ) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []

        for page in pages:
            page_chunks = self._split_text(page.text)

            for chunk_index, text in enumerate(page_chunks):
                chunk_id = self._create_chunk_id(
                    pdf_name=page.pdf_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                )

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=text,
                        pdf_name=page.pdf_name,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        metadata={
                            "source": page.pdf_name,
                            "page": page.page_number,
                        },
                    )
                )

        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        cleaned_text = re.sub(r"\s+", " ", text).strip()

        if not cleaned_text:
            return []

        if len(cleaned_text) <= self.chunk_size:
            return [cleaned_text]

        chunks: list[str] = []
        start = 0
        text_length = len(cleaned_text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                sentence_end = max(
                    cleaned_text.rfind(". ", start, end),
                    cleaned_text.rfind("? ", start, end),
                    cleaned_text.rfind("! ", start, end),
                    cleaned_text.rfind("\n", start, end),
                )

                minimum_boundary = start + int(
                    self.chunk_size * 0.6
                )

                if sentence_end >= minimum_boundary:
                    end = sentence_end + 1

            chunk = cleaned_text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    @staticmethod
    def _create_chunk_id(
        pdf_name: str,
        page_number: int,
        chunk_index: int,
        text: str,
    ) -> str:
        raw_value = (
            f"{pdf_name}|{page_number}|{chunk_index}|{text}"
        )

        return hashlib.sha256(
            raw_value.encode("utf-8")
        ).hexdigest()[:20]


class EmbeddingService:
    """Dummy embedding service for demonstration purposes.
    
    In a real application, replace this with a proper embedding model
    such as sentence-transformers or OpenAI embeddings.
    """

    def __init__(
        self,
        model_name: str,
        embedding_size: int = 384,
    ) -> None:
        self.model_name = model_name
        self.embedding_size = embedding_size
        # Note: This is a deterministic hash-based embedding for demonstration.
        # Replace with actual model loading in production.

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            embedding = self._embed_text(text)
            embeddings.append(embedding)
        return np.vstack(embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        embedding = self._embed_text(text)
        # Return as 2D array with shape (1, embedding_size) for FAISS compatibility
        return embedding.reshape(1, -1)

    def _embed_text(self, text: str) -> np.ndarray:
        """Create a deterministic embedding from text using SHA256 hashing."""
        # Create a hash of the text
        hash_object = hashlib.sha256(text.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert hex digest to list of byte values (0-255)
        hash_bytes = [
            int(hash_hex[i:i+2], 16) 
            for i in range(0, len(hash_hex), 2)
        ]
        
        # Extend or truncate to desired embedding size
        if len(hash_bytes) < self.embedding_size:
            # Repeat the hash bytes to reach embedding_size
            repeated = (
                hash_bytes * 
                ((self.embedding_size // len(hash_bytes)) + 1)
            )[:self.embedding_size]
            embedding = np.array(repeated, dtype=np.float32)
        else:
            embedding = np.array(hash_bytes[:self.embedding_size], dtype=np.float32)
        
        # Normalize to unit vector for cosine similarity with inner product
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

import hashlib
import re

from src.ingestion.pdf_loader import PDFPage
from src.models import DocumentChunk


class TextChunker:
    """Split PDF pages into overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(
        self,
        pages: list[PDFPage],
    ) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []

        for page in pages:
            page_chunks = self._split_text(page.text)

            for chunk_index, text in enumerate(page_chunks):
                chunk_id = self._create_chunk_id(
                    pdf_name=page.pdf_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                )

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=text,
                        pdf_name=page.pdf_name,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        metadata={
                            "source": page.pdf_name,
                            "page": page.page_number,
                        },
                    )
                )

        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        cleaned_text = re.sub(r"\s+", " ", text).strip()

        if not cleaned_text:
            return []

        if len(cleaned_text) <= self.chunk_size:
            return [cleaned_text]

        chunks: list[str] = []
        start = 0
        text_length = len(cleaned_text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                sentence_end = max(
                    cleaned_text.rfind(". ", start, end),
                    cleaned_text.rfind("? ", start, end),
                    cleaned_text.rfind("! ", start, end),
                    cleaned_text.rfind("\n", start, end),
                )

                minimum_boundary = start + int(
                    self.chunk_size * 0.6
                )

                if sentence_end >= minimum_boundary:
                    end = sentence_end + 1

            chunk = cleaned_text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    @staticmethod
    def _create_chunk_id(
        pdf_name: str,
        page_number: int,
        chunk_index: int,
        text: str,
    ) -> str:
        raw_value = (
            f"{pdf_name}|{page_number}|{chunk_index}|{text}"
        )

        return hashlib.sha256(
            raw_value.encode("utf-8")
        ).hexdigest()[:20]