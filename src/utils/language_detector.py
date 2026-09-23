from __future__ import annotations

from lingua import Language, LanguageDetectorBuilder


class LanguageDetector:
    """
    Multilingual language detector.

    Detects languages from document text without requiring
    FastText or native C++ compilation.
    """

    def __init__(self) -> None:

        self.detector = (
            LanguageDetectorBuilder
            .from_all_languages()
            .build()
        )

    def detect(
        self,
        text: str,
    ) -> dict:

        if not text or not text.strip():

            return {
                "language_code": "unknown",
                "language": "Unknown",
                "confidence": 0.0,
            }

        cleaned_text = self._clean_text(text)

        language = (
            self.detector
            .detect_language_of(cleaned_text)
        )

        if language is None:

            return {
                "language_code": "unknown",
                "language": "Unknown",
                "confidence": 0.0,
            }

        language_name = language.name.title()

        language_code = (
            language.iso_code_639_1.name.lower()
            if language.iso_code_639_1
            else "unknown"
        )

        confidence = (
            self._estimate_confidence(
                cleaned_text,
                language,
            )
        )

        return {
            "language_code": language_code,
            "language": language_name,
            "confidence": confidence,
        }

    # ==========================================================
    # TOP LANGUAGES
    # ==========================================================

    def detect_top_languages(
        self,
        text: str,
        top_k: int = 5,
    ) -> list[dict]:

        if not text or not text.strip():
            return []

        cleaned_text = self._clean_text(text)

        detected = (
            self.detector
            .compute_language_confidence_values(
                cleaned_text
            )
        )

        detected = sorted(
            detected,
            key=lambda x: x.value,
            reverse=True,
        )

        results = []

        for item in detected[:top_k]:

            language = item.key

            code = (
                language.iso_code_639_1.name.lower()
                if language.iso_code_639_1
                else "unknown"
            )

            results.append(
                {
                    "language_code": code,
                    "language": language.name.title(),
                    "confidence": round(
                        float(item.value),
                        4,
                    ),
                }
            )

        return results

    # ==========================================================
    # CLEAN TEXT
    # ==========================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        text = text.replace(
            "\n",
            " ",
        )

        text = text.replace(
            "\r",
            " ",
        )

        return " ".join(
            text.split()
        )

    # ==========================================================
    # CONFIDENCE
    # ==========================================================

    def _estimate_confidence(
        self,
        text: str,
        language: Language,
    ) -> float:

        try:

            confidence_values = (
                self.detector
                .compute_language_confidence_values(
                    text
                )
            )

            for item in confidence_values:

                if item.key == language:

                    return round(
                        float(item.value),
                        4,
                    )

        except Exception:
            pass

        return 0.0