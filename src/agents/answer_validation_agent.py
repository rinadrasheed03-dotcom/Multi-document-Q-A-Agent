from __future__ import annotations

import json
import re

from openai import OpenAI


class AnswerValidationAgent:
    """
    Validates whether a generated answer is supported
    by the supplied document evidence using NVIDIA NIM.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
    ):

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

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model_name = model_name

    # ==========================================================
    # VALIDATE
    # ==========================================================

    def validate(
        self,
        question: str,
        answer: str,
        evidence: str,
    ) -> bool:

        if not question.strip():
            return False

        if not answer.strip():
            return False

        if not evidence.strip():
            return False

        prompt = f"""
Determine whether the ANSWER is completely supported
by the EVIDENCE.

QUESTION:
{question}

ANSWER:
{answer}

EVIDENCE:
{evidence}

IMPORTANT:

You MUST return ONLY this JSON object.

If the answer is fully supported:

{{"supported": true}}

If the answer contains any unsupported important claim:

{{"supported": false}}

DO NOT explain your reasoning.
DO NOT write any text before the JSON.
DO NOT write any text after the JSON.
DO NOT use markdown.
DO NOT use outside knowledge.
""".strip()

        try:

            response = self.client.chat.completions.create(
                model=self.model_name,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON document-grounding "
                            "validator. Return ONLY valid JSON. "
                            "Never provide explanations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                temperature=0,

                # Increased because gpt-oss-120b may reason
                # before producing the final response.
                max_tokens=1000,

                # Ask NVIDIA to return JSON.
                response_format={
                    "type": "json_object"
                },
            )

            content = response.choices[0].message.content

            if not content:
                print(
                    "VALIDATION FAILED: empty NVIDIA response"
                )
                return False

            print("\n" + "-" * 70)
            print("NVIDIA VALIDATION RESPONSE")
            print("-" * 70)
            print(content)
            print("-" * 70)

            result = self._parse_json(content)

            supported = result.get("supported")

            if isinstance(supported, bool):

                print(
                    "VALIDATION RESULT:",
                    supported,
                )

                return supported

            if isinstance(supported, str):

                value = supported.strip().lower()

                if value == "true":
                    print("VALIDATION RESULT: True")
                    return True

                if value == "false":
                    print("VALIDATION RESULT: False")
                    return False

            print(
                "VALIDATION FAILED: invalid "
                "'supported' value."
            )

            return False

        except Exception as exc:

            print(
                "\nVALIDATION ERROR:",
                type(exc).__name__,
                str(exc),
            )

            return False

    # ==========================================================
    # JSON PARSER
    # ==========================================================

    @staticmethod
    def _parse_json(
        raw_response: str,
    ) -> dict:

        cleaned = raw_response.strip()

        # Remove markdown fences if present.
        cleaned = re.sub(
            r"^```json\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

        # ------------------------------------------------------
        # Direct JSON
        # ------------------------------------------------------

        try:

            result = json.loads(cleaned)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        # ------------------------------------------------------
        # Find JSON object inside response
        # ------------------------------------------------------

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1 and end > start:

            json_text = cleaned[
                start:end + 1
            ]

            try:

                result = json.loads(
                    json_text
                )

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                pass

        # ------------------------------------------------------
        # Last-resort parser
        #
        # This handles a reasoning response such as:
        #
        # "Thus answer is fully supported.
        #  supported: true"
        # ------------------------------------------------------

        lower = cleaned.lower()

        if (
            "supported: true" in lower
            or '"supported": true' in lower
        ):
            return {
                "supported": True
            }

        if (
            "supported: false" in lower
            or '"supported": false' in lower
        ):
            return {
                "supported": False
            }

        raise ValueError(
            "NVIDIA validation response "
            "was not valid JSON."
        )