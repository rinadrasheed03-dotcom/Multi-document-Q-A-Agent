from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


def load_pdf(file_bytes: bytes, file_name: str) -> list[dict]:
    """
    Extract text from every PDF page.

    Returns:
        [
            {
                "text": "...",
                "source_file": "example.pdf",
                "page_number": 1
            }
        ]
    """

    reader = PdfReader(BytesIO(file_bytes))

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source_file": file_name,
                "page_number": page_number,
            }
        )

    return documents