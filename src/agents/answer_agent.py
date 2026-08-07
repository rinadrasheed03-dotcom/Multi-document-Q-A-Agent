from __future__ import annotations

import time

from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    RateLimitError,
)

from src.models import RetrievalResult
from src.utils.citation_utils import build_source_label


class AnswerAgent:
    """Generate grounded answers using the Groq API."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to the .env file."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self.client = Groq(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def generate_answer(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> str:
        if not results:
            return (
                "I could not find relevant information in the "
                "uploaded PDF documents."
            )

        context = self._build_context(results)

        system_prompt = """
You are a Multi-PDF RAG Answer Agent.

Answer the user's question only using the supplied PDF context.

Rules:
1. Do not invent information.
2. If the context is insufficient, clearly state that.
3. Add a citation after every important factual claim.
4. Use this citation format: [PDF name, page X].
5. When documents disagree, explain the disagreement.
6. Mention when similar evidence appears in multiple PDFs.
7. Keep the answer clear, focused and concise.
""".strip()

        user_prompt = f"""
USER QUESTION:
{query}

RETRIEVED PDF CONTEXT:
{context}

Generate a grounded answer using PDF and page citations.
""".strip()

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    stream=False,
                )

                content = response.choices[0].message.content

                if not content:
                    return "Groq returned an empty response."

                return content.strip()

            except RateLimitError as exc:
                last_error = exc

                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

            except (
                APITimeoutError,
                APIConnectionError,
            ) as exc:
                last_error = exc

                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

            except Exception as exc:
                raise RuntimeError(
                    f"Groq answer generation failed: {exc}"
                ) from exc

        raise RuntimeError(
            "Groq answer generation failed after "
            f"{self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _build_context(
        results: list[RetrievalResult],
    ) -> str:
        context_sections: list[str] = []

        for index, result in enumerate(results, start=1):
            citation = build_source_label(result)

            duplicate_note = ""

            if result.duplicate_sources:
                duplicate_labels = [
                    (
                        f"{source['pdf_name']}, "
                        f"page {source['page_number']}"
                    )
                    for source in result.duplicate_sources
                ]

                duplicate_note = (
                    "\nSimilar supporting content also appears in: "
                    + "; ".join(duplicate_labels)
                )

            section = f"""
Evidence {index}
Source: {citation}

Content:
{result.chunk.text}
{duplicate_note}
""".strip()

            context_sections.append(section)

        return "\n\n---\n\n".join(context_sections)