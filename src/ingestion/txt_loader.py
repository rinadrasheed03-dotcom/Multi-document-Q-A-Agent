from __future__ import annotations


def load_txt(
    file_bytes: bytes,
    file_name: str,
) -> list[dict]:

    text = file_bytes.decode(
        "utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return []

    return [
        {
            "text": text,
            "source_file": file_name,
            "page_number": None,
        }
    ]