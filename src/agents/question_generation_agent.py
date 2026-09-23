from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from src.models.qa_schema import QuestionAnswer


class QuestionGenerationAgent:
    """
    Generates detailed brief-answer questions and answers
    strictly from uploaded document content.

    Features:
        - Brief-answer questions only
        - Generalized, topic-focused questions
        - Entrepreneur/business-oriented perspective
        - No document/study/meta wording in questions
        - Same-language question and answer
        - Strictly document-grounded answers
        - Exact supporting evidence and source metadata
        - Per-question difficulty level
        - JSON structured output
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):

        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not configured."
            )

        if not base_url:
            raise ValueError(
                "NVIDIA_BASE_URL is not configured."
            )

        if not model_name:
            raise ValueError(
                "NVIDIA_MODEL is not configured."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ==========================================================
    # GENERATE QUESTIONS
    # ==========================================================

    def generate(
        self,
        context: str,
        num_questions: int,
        question_types: list[str] | None = None,
        difficulty: str = "medium",
        source_file: str = "",
        page_number=None,
        language: str = "English",
    ) -> list[QuestionAnswer]:
        """
        Generate detailed brief-answer questions.

        Parameters
        ----------
        context:
            Source document/chunk content.

        num_questions:
            Number of questions to generate.

        question_types:
            Kept for compatibility with the existing application.
            The system now generates ONLY brief-answer questions.

        difficulty:
            Question difficulty.

        source_file:
            Original document filename.

        page_number:
            PDF page number, when available.

        language:
            Detected language of the document/chunk.
        """

        if not context or not context.strip():
            return []

        # ------------------------------------------------------
        # Always use brief_answer.
        # ------------------------------------------------------

        question_types = [
            "brief_answer"
        ]

        # ------------------------------------------------------
        # Build prompt.
        # ------------------------------------------------------

        prompt = self._build_prompt(
            context=context,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
            language=language,
        )

        # ------------------------------------------------------
        # Call NVIDIA NIM.
        # ------------------------------------------------------

        response = self.client.chat.completions.create(
            model=self.model_name,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert educational "
                        "question-generation system. "
                        "You MUST use only the supplied "
                        "document content. "
                        "Never use outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=self.temperature,

            max_tokens=self.max_tokens,

            
        )

        # ------------------------------------------------------
        # Extract response.
        # ------------------------------------------------------

        raw_response = (
            response
            .choices[0]
            .message
            .content
        )

        # ------------------------------------------------------
        # Debug: show the exact NVIDIA model response.
        # ------------------------------------------------------

        print("\n" + "=" * 80)
        print("NVIDIA RAW RESPONSE")
        print("=" * 80)
        print(raw_response)
        print("=" * 80 + "\n")

        if not raw_response:
            return []

        # ------------------------------------------------------
        # Parse JSON.
        # ------------------------------------------------------

        data = self._parse_json(
            raw_response
        )

        print("NVIDIA PARSED TYPE:", type(data).__name__)
        print("NVIDIA PARSED KEYS:", list(data.keys()) if isinstance(data, dict) else "NOT A DICT")
        print(
            "NVIDIA QUESTION ITEMS:",
            len(data.get("questions", []))
            if isinstance(data, dict)
            and isinstance(data.get("questions", []), list)
            else 0,
        )

        results: list[QuestionAnswer] = []

        # ------------------------------------------------------
        # Convert generated questions.
        # ------------------------------------------------------

        question_items = data.get("questions", [])

        if not isinstance(question_items, list):
            raise ValueError(
                "NVIDIA returned a JSON object, but "
                "'questions' is not a list."
            )

        for index, item in enumerate(question_items, start=1):

            if not isinstance(item, dict):
                print(
                    f"Skipping NVIDIA question {index}: "
                    "item is not an object."
                )
                continue

            try:
                question = self._convert_to_question(
                    item=item,
                    source_file=source_file,
                    page_number=page_number,
                    language=language,
                )

            except Exception as exc:
                print(
                    f"QuestionAnswer conversion failed for "
                    f"item {index}: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

            if question is not None:
                results.append(question)
            else:
                print(
                    f"Skipping NVIDIA question {index}: "
                    "missing question, answer, or evidence."
                )

        print(
            "NVIDIA VALID QUESTION OBJECTS:",
            len(results),
        )

        return results

    # ==========================================================
    # PROMPT
    # ==========================================================

    @staticmethod
    def _build_prompt(
        context: str,
        num_questions: int,
        question_types: list[str],
        difficulty: str,
        language: str,
    ) -> str:

        return f"""
You are an expert question-generation system.

Generate questions ONLY from the SPECIFIC CONTENT in the supplied
text. The questions must be about the actual concepts, names, terms,
methods, technologies, systems, variables, measurements, processes,
findings, advantages, limitations, problems, solutions, and
relationships that are explicitly mentioned in the supplied content.

The user wants questions that are SPECIFIC to the content, not vague
or generic questions.

==================================================
MOST IMPORTANT RULE: SPECIFIC TOPIC QUESTIONS
==================================================

Use the actual terminology and specific things mentioned in the
supplied content.

For example, if the content mentions:
- a particular technology
- a particular algorithm
- a particular dataset
- a particular model
- a particular process
- particular variables
- particular performance values
- particular findings
- particular components

then use those specific things in the questions.

BAD:
"What factors affect system performance?"

GOOD:
"What factors affecting the performance of [the specific system
named in the supplied content] are identified, and how does each
factor influence its performance?"

BAD:
"Why is data analysis important?"

GOOD:
"How does [the specific analysis method named in the supplied
content] process the identified variables, and what outcome does
this process produce?"

BAD:
"What are the benefits of technology?"

GOOD:
"What advantages of [the specific technology named in the supplied
content] are identified, and what problem do those advantages help
address?"

The examples above are only style examples. Always replace the
bracketed concepts with REAL terms from the supplied content.

Do NOT invent a specific name that is not present in the supplied
content.

==================================================
NO DOCUMENT / STUDY / REPORT WORDING
==================================================

The question must sound like a normal question about the subject
itself.

NEVER start or phrase a question with:

- "According to the document..."
- "Based on the document..."
- "In the document..."
- "From the document..."
- "The document states..."
- "The document mentions..."
- "According to the study..."
- "Based on the study..."
- "In the study..."
- "From the study..."
- "The study states..."
- "The study mentions..."
- "According to the report..."
- "Based on the report..."
- "In the report..."
- "From the report..."
- "According to the paper..."
- "Based on the paper..."
- "In the paper..."
- "From the paper..."
- "According to the article..."
- "Based on the article..."
- "What does the document say..."
- "What does the study say..."
- "What is mentioned in the document..."
- "What is discussed in the study..."
- "Based on the provided text..."
- "From the provided content..."
- "From the given context..."

Do not mention the source, document, study, report, paper, article,
passage, or context in the question.

IMPORTANT:
A word such as "study", "system", "model", "method", or "dataset"
may be used if it is part of the ACTUAL SUBJECT MATTER.

GOOD:
"What limitations of the proposed model are identified?"

BAD:
"According to the study, what limitations does the model have?"

==================================================
ENTREPRENEUR / DECISION-MAKER STYLE
==================================================

Think like an entrepreneur, engineer, manager, product developer,
or technical decision-maker who wants to understand the SPECIFIC
subject matter well enough to make a practical decision.

Prefer questions about the specific items mentioned in the content:

- how a named technology works
- how a named method works
- why a named method is used
- the role of a named component
- relationships between named variables
- causes and effects explicitly discussed
- advantages of a named approach
- limitations of a named approach
- challenges of a named system
- performance of a named system
- comparison of named approaches
- processes and workflows explicitly described
- specific findings and their implications
- specific parameters and their effects
- how named components work together

Do not add business facts, market information, customers, costs,
competitors, regulations, or applications unless they are explicitly
present in the supplied content.

The entrepreneurial perspective changes the QUESTION STYLE only.
It must never introduce outside information.

==================================================
DOCUMENT LANGUAGE
==================================================

The detected language of the supplied content is:

{language}

Generate BOTH the question and answer in the SAME LANGUAGE as the
supplied content.

Do NOT translate the source into English.

Preserve important technical names and terminology from the source.

==================================================
QUESTION TYPE
==================================================

Generate ONLY brief-answer questions.

Do NOT generate:
- MCQs
- Multiple-choice questions
- True/False questions
- Yes/No questions
- Fill-in-the-blank questions
- One-word questions
- Questions answered by only a name
- Questions answered by only a number
- Questions answered by only a date

==================================================
QUESTION REQUIREMENTS
==================================================

Generate exactly {num_questions} candidate questions.

For every question:

1. It must be about a SPECIFIC concept, entity, method, process,
   finding, parameter, technology, system, or relationship explicitly
   present in the supplied content.

2. Use the actual names and terminology from the supplied content.

3. Do not make the question so broad that it could apply to almost
   any unrelated document.

4. The question must be standalone.

5. The question must NOT mention the source itself.

6. The question should test understanding, not simple copying.

7. Prefer "how", "why", "what role", "how does", "what relationship",
   "what difference", "what advantages", "what limitations",
   "what causes", or "what effects" when supported by the content.

8. Do not ask for information absent from the supplied content.

9. Avoid duplicate questions.

10. Normally use 15-35 words when the source supports that level
    of specificity.

11. Normally require a 2-5 sentence answer.

==================================================
ANSWER REQUIREMENTS
==================================================

For every question:

1. Answer the exact question asked.
2. Use ONLY information in the supplied content.
3. Use the source's specific terminology.
4. Normally provide 2-5 complete sentences.
5. Do not add outside facts.
6. Do not invent examples.
7. Do not make unsupported assumptions.
8. Every important claim must be supported by the evidence.
9. Answer in the SAME LANGUAGE as the source.

==================================================
EVIDENCE / PROVENANCE
==================================================

For every question provide "evidence".

The evidence MUST be the actual supporting passage from the supplied
content.

Use exact or nearly exact wording from the source whenever possible.

The evidence must directly support the answer.

Do NOT write:
"Page 4 discusses this."

Instead provide the actual passage that supports the answer.

The application separately attaches source filename and page metadata
when available.

==================================================
DIFFICULTY
==================================================

Requested difficulty:

{difficulty}

Each question MUST have one of:

- easy
- medium
- hard

EASY:
Direct understanding of a specific named concept, method, component,
process, parameter, or finding.

MEDIUM:
Explanation of relationships, processes, causes/effects,
advantages/limitations, or connected specific concepts.

HARD:
Deeper analysis, comparison, multi-step reasoning, or interpretation
of relationships among specific concepts explicitly present in the
content.

MIXED:
Use a meaningful mixture of easy, medium, and hard. Each individual
question MUST contain its actual level, not "mixed".

==================================================
FINAL QUALITY CHECK
==================================================

Before returning a question, silently check:

1. Does it contain or clearly target a SPECIFIC thing mentioned in
   the supplied content?
2. Could this question apply to many unrelated topics?
   If yes, rewrite it using the actual source terminology.
3. Does it avoid document/study/report/paper/source wording?
4. Does it use specific concepts rather than broad generic concepts?
5. Is the answer completely supported by the supplied evidence?
6. Is the evidence actually present in the supplied content?
7. Is the difficulty correct?
8. Is it different from the other questions?
9. Did you use outside knowledge?
10. Did you invent any name, number, result, or fact?

If any answer is NO, rewrite the question.

==================================================
OUTPUT FORMAT
==================================================

Return VALID JSON ONLY.

Do not include markdown.
Do not include ```json.
Do not include explanations before or after the JSON.

Use exactly:

{{
    "questions": [
        {{
            "question": "...",
            "answer": "...",
            "question_type": "brief_answer",
            "difficulty": "medium",
            "language": "{language}",
            "evidence": "..."
        }}
    ]
}}

The difficulty value MUST be exactly:
"easy", "medium", or "hard".

==================================================
SUPPLIED CONTENT
==================================================

{context}

==================================================
END SUPPLIED CONTENT
==================================================
"""
    # ==========================================================
    # JSON PARSER
    # ==========================================================

    @staticmethod
    def _parse_json(
        raw_response: str,
    ) -> dict[str, Any]:

        if not raw_response:
            raise ValueError(
                "The question generation model "
                "returned an empty response."
            )

        cleaned = raw_response.strip()

        # ------------------------------------------------------
        # Remove Markdown code fences if the model added them.
        # ------------------------------------------------------

        cleaned = re.sub(
            r"^```(?:json)?\s*",
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
        # First attempt: complete JSON response.
        # ------------------------------------------------------

        try:
            data = json.loads(cleaned)

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            pass

        # ------------------------------------------------------
        # Second attempt: find JSON object inside surrounding text.
        # ------------------------------------------------------

        start_index = cleaned.find("{")
        end_index = cleaned.rfind("}")

        if (
            start_index != -1
            and end_index != -1
            and end_index > start_index
        ):

            json_text = cleaned[
                start_index:end_index + 1
            ]

            try:
                data = json.loads(json_text)

                if isinstance(data, dict):
                    return data

            except json.JSONDecodeError:
                pass

        # ------------------------------------------------------
        # Third attempt: model returned only a questions array.
        # ------------------------------------------------------

        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")

        if (
            array_start != -1
            and array_end != -1
            and array_end > array_start
        ):

            array_text = cleaned[
                array_start:array_end + 1
            ]

            try:
                questions = json.loads(array_text)

                if isinstance(questions, list):
                    return {
                        "questions": questions
                    }

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "The question generation model "
            "did not return valid JSON."
        )

    # ==========================================================
    # CONVERT MODEL OUTPUT
    # ==========================================================

    @staticmethod
    def _convert_to_question(
        item: dict[str, Any],
        source_file: str,
        page_number,
        language: str,
    ) -> QuestionAnswer | None:

        # ------------------------------------------------------
        # Question
        # ------------------------------------------------------

        question_text = str(
            item.get(
                "question",
                "",
            )
        ).strip()

        # ------------------------------------------------------
        # Reject source-referential/meta questions.
        # ------------------------------------------------------

        forbidden_phrases = [
            "in the document",
            "in this document",
            "according to the document",
            "based on the document",
            "from the document",
            "the document says",
            "the document states",
            "the document mentions",

            "in the study",
            "in this study",
            "according to the study",
            "based on the study",
            "from the study",
            "the study says",
            "the study states",
            "the study mentions",

            "in the report",
            "according to the report",
            "based on the report",
            "from the report",
            "the report says",
            "the report states",
            "the report mentions",

            "in the paper",
            "according to the paper",
            "based on the paper",
            "from the paper",
            "the paper says",
            "the paper states",
            "the paper mentions",

            "in the article",
            "according to the article",
            "based on the article",
            "from the article",

            "from the provided text",
            "from the provided content",
            "from the given context",
            "according to the provided content",
            "based on the provided content",
        ]

        question_lower = question_text.lower()

        if any(
            phrase in question_lower
            for phrase in forbidden_phrases
        ):
            print(
                "Skipping source-referential question:",
                question_text,
            )
            return None

        # ------------------------------------------------------
        # Answer
        # ------------------------------------------------------

        answer_text = str(
            item.get(
                "answer",
                "",
            )
        ).strip()

        # ------------------------------------------------------
        # Evidence
        # ------------------------------------------------------

        evidence_text = str(
            item.get(
                "evidence",
                item.get(
                    "supporting_evidence",
                    item.get(
                        "source_evidence",
                        "",
                    ),
                ),
            )
        ).strip()

        # ------------------------------------------------------
        # Validate required fields.
        # ------------------------------------------------------

        if not question_text:
            return None

        if not answer_text:
            return None

        if not evidence_text:
            return None

        # ------------------------------------------------------
        # Always force brief_answer.
        # ------------------------------------------------------

        question_type = (
            "brief_answer"
        )

        # ------------------------------------------------------
        # Difficulty.
        # ------------------------------------------------------

        generated_difficulty = str(
            item.get(
                "difficulty",
                "medium",
            )
        ).strip().lower()

        if generated_difficulty not in {
            "easy",
            "medium",
            "hard",
        }:
            generated_difficulty = "medium"

        # ------------------------------------------------------
        # Create QuestionAnswer object.
        # ------------------------------------------------------

        return QuestionAnswer(
            question=question_text,

            answer=answer_text,

            question_type=question_type,

            difficulty=generated_difficulty,

            language=language,

            source_file=source_file,

            page_number=page_number,

            evidence=evidence_text,

            options=None,

            correct_option=None,
        )
        