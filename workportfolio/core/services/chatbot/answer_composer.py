from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.models import DocumentChunk
from core.services.chatbot.answer_service import AnswerService
from core.services.chatbot.extractors import try_extract_contact, try_extract_skills


@dataclass(frozen=True)
class ComposedAnswer:
    answer: str
    confidence: float
    mode: str  # "generic" | "contact_extractor" | "skills_extractor"


class AnswerComposer:
    """
    Enterprise approach:
    - Always retrieve evidence
    - Always answer from evidence
    - Optionally apply specialized extractors if they clearly fit
    (not restricting the bot; just improving accuracy/format).
    """

    @staticmethod
    def compose(question: str, chunks: List[DocumentChunk]) -> ComposedAnswer:
        # 1) Try extractors (optional enhancements)
        applied, answer, boost = try_extract_contact(question, chunks)
        if applied:
            return ComposedAnswer(answer=answer, confidence=min(0.9, 0.75 + boost), mode="contact_extractor")

        applied, answer, boost = try_extract_skills(question, chunks)
        if applied:
            return ComposedAnswer(answer=answer, confidence=min(0.9, 0.70 + boost), mode="skills_extractor")

        # 2) Default: generic grounded response
        result = AnswerService.build_response(
            user_question=question, chunks=chunks)
        return ComposedAnswer(answer=result["answer"], confidence=result["confidence"], mode="generic")
