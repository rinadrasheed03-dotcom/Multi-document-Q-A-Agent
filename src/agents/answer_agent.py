from __future__ import annotations

import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src.models import RetrievalResult
from src.utils.citation_utils import build_source_label


class AnswerAgent:
    """Generate grounded answers using the NVIDIA NIM API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        max_retries: int = 3,
    ) -> None:

        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is missing. "
                "Add it to the .env file."
            )

        if not base_url:
            raise ValueError(
                "NVIDIA_BASE_URL is missing. "
                "Add it to the .env file."
            )

        if not model_name:
            raise ValueError(
                "NVIDIA_MODEL is missing. "
                "Add it to the .env file."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )

    # ==========================================================
    # GENERATE ANSWER
    # ==========================================================

    def generate_answer(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> str:

        if not results:
            return (
                "I could not find relevant information in the "
                "uploaded documents."
            )

        context = self._build_context(results)

        system_prompt = """
You are a Multi-Datatype RAG Answer Agent.

Answer the user's question ONLY using the supplied retrieved
document context.

Rules:

1. Do not invent information.

2. Do not use outside knowledge.

3. If the retrieved context is insufficient to answer the
   question, clearly state that the available document
   context is insufficient.

4. Add a citation after every important factual claim.

5. Use the source citation format supplied in the context.

6. When documents disagree, clearly explain the disagreement
   using only the supplied evidence.

7. Mention when similar supporting evidence appears in
   multiple source documents.

8. Keep the answer clear, focused and concise.

9. Answer in the SAME LANGUAGE as the user's question and
   the relevant source content whenever possible.

10. Do not claim information that cannot be supported by
    the retrieved document context.
""".strip()

        user_prompt = f"""
USER QUESTION:
{query}

RETRIEVED DOCUMENT CONTEXT:
{context}

Generate a grounded answer using only the retrieved
document evidence and include appropriate source citations.
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

                    max_tokens=self.max_tokens,

                    stream=False,
                )

                content = response.choices[0].message.content

                if not content:
                    return (
                        "NVIDIA NIM returned an empty response."
                    )

                return content.strip()

            # --------------------------------------------------
            # Rate limit
            # --------------------------------------------------

            except RateLimitError as exc:

                last_error = exc

                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

            # --------------------------------------------------
            # Connection / timeout
            # --------------------------------------------------

            except (
                APITimeoutError,
                APIConnectionError,
            ) as exc:

                last_error = exc

                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

            # --------------------------------------------------
            # Other API errors
            # --------------------------------------------------

            except Exception as exc:

                raise RuntimeError(
                    f"NVIDIA NIM answer generation failed: {exc}"
                ) from exc

        raise RuntimeError(
            "NVIDIA NIM answer generation failed after "
            f"{self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    # ==========================================================
    # BUILD RETRIEVAL CONTEXT
    # ==========================================================

    @staticmethod
    def _build_context(
        results: list[RetrievalResult],
    ) -> str:

        context_sections: list[str] = []

        for index, result in enumerate(
            results,
            start=1,
        ):

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

        return "\n\n---\n\n".join(
            context_sections
        )