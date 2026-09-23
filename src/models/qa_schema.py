from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class QuestionAnswer:

    # ---------------------------------------------------------
    # Generated Q&A
    # ---------------------------------------------------------

    question: str
    answer: str

    # ---------------------------------------------------------
    # Question information
    # ---------------------------------------------------------

    question_type: str = "brief_answer" 
    difficulty: str = "medium"

    # ---------------------------------------------------------
    # Language
    # ---------------------------------------------------------

    language: str = "English"

    # ---------------------------------------------------------
    # Source information
    # ---------------------------------------------------------

    source_file: str = ""

    source_type: str = ""

    page_number: Optional[int] = None

    slide_number: Optional[int] = None

    sheet_name: Optional[str] = None

    row_number: Optional[int] = None

    paragraph_number: Optional[int] = None

    line_start: Optional[int] = None

    line_end: Optional[int] = None

    # ---------------------------------------------------------
    # Exact supporting evidence
    # ---------------------------------------------------------

    evidence: str = ""

    # ---------------------------------------------------------
    # Question options
    # ---------------------------------------------------------

    options: Optional[list[str]] = None

    correct_option: Optional[str] = None

    # ---------------------------------------------------------
    # Chunk identifier
    # ---------------------------------------------------------

    chunk_id: Optional[str] = None

    def to_dict(self) -> dict:

        return asdict(self)