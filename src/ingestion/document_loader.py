from __future__ import annotations

from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.docx_loader import load_docx
from src.ingestion.txt_loader import load_txt
from src.ingestion.csv_loader import load_csv
from src.ingestion.xlsx_loader import load_xlsx
from src.ingestion.pptx_loader import load_pptx


def load_document(
    file_bytes: bytes,
    file_name: str,
) -> list[dict]:

    extension = Path(
        file_name
    ).suffix.lower()

    if extension == ".pdf":

        return load_pdf(
            file_bytes,
            file_name,
        )

    if extension == ".docx":

        return load_docx(
            file_bytes,
            file_name,
        )

    if extension == ".txt":

        return load_txt(
            file_bytes,
            file_name,
        )

    if extension == ".csv":

        return load_csv(
            file_bytes,
            file_name,
        )

    if extension in {
        ".xlsx",
        ".xls",
    }:

        return load_xlsx(
            file_bytes,
            file_name,
        )

    if extension == ".pptx":

        return load_pptx(
            file_bytes,
            file_name,
        )

    raise ValueError(
        f"Unsupported file type: {extension}"
    )