from __future__ import annotations

from dataclasses import dataclass

import fitz


@dataclass
class PDFPage:
    """Text extracted from one PDF page."""

    pdf_name: str
    page_number: int
    text: str


class PDFLoader:
    """Extract text from uploaded PDF files."""

    @staticmethod
    def load_pdf(
        file_name: str,
        file_bytes: bytes,
    ) -> list[PDFPage]:
        pages: list[PDFPage] = []

        try:
            document = fitz.open(
                stream=file_bytes,
                filetype="pdf",
            )

            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = page.get_text("text").strip()

                if not text:
                    continue

                pages.append(
                    PDFPage(
                        pdf_name=file_name,
                        page_number=page_index + 1,
                        text=text,
                    )
                )

            document.close()

        except Exception as exc:
            raise ValueError(
                f"Unable to process PDF '{file_name}': {exc}"
            ) from exc

        return pages