from __future__ import annotations

from io import BytesIO

from docx import Document


def load_docx(
    file_bytes: bytes,
    file_name: str,
) -> list[dict]:

    document = Document(
        BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    if not paragraphs:
        return []

    return [
        {
            "text": "\n".join(paragraphs),
            "source_file": file_name,
            "page_number": None,
        }
    ]