from __future__ import annotations

import hashlib
import re


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""

    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def text_hash(text: str) -> str:
    """Create a stable hash from normalized text."""

    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 retrieval."""

    normalized = normalize_text(text)

    return [
        token
        for token in normalized.split()
        if len(token) > 1
    ]


def truncate_text(
    text: str,
    maximum_length: int = 500,
) -> str:
    """Shorten text for UI previews."""

    if len(text) <= maximum_length:
        return text

    return text[:maximum_length].rstrip() + "..."