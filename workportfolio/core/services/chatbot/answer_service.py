"""
Answer generation service (non-LLM v1).

Creates a clean, structured answer from retrieved chunks.
Later we will replace the summarization step with Gemini/Ollama,
but keep the same response schema.
"""

from typing import Dict, List
from core.models import DocumentChunk


class AnswerService:
    @staticmethod
    def _clean(text: str) -> str:
        text = (text or "").replace("\r", " ").replace("\n", " ").strip()
        # collapse multiple spaces
        while "  " in text:
            text = text.replace("  ", " ")
        return text

    @staticmethod
    def _make_snippet(text: str, max_len: int = 220) -> str:
        text = AnswerService._clean(text)
        return text if len(text) <= max_len else text[:max_len].rstrip() + "..."

    @staticmethod
    def build_response(user_question: str, chunks: List[DocumentChunk]) -> Dict:
        if not chunks:
            return {
                "answer": "I don’t have enough evidence in the uploaded documents to answer that.",
                "citations": [],
                "confidence": 0.0,
            }

        # Use the best 3 chunks for the visible answer
        top_chunks = chunks[:3]

        # Produce a cleaner “evidence summary” format
        evidence = [AnswerService._make_snippet(c.content) for c in top_chunks]

        answer_lines = [
            "Here’s what I found in the uploaded documents (grounded evidence):",
        ]
        for i, ev in enumerate(evidence, start=1):
            answer_lines.append(f"{i}. {ev}")

        citations = [
            {
                "document_id": str(c.document.id),
                "document_title": c.document.title,
                "chunk_id": str(c.id),
                "chunk_index": c.chunk_index,
                "section_title": c.section_title,
                "page_number": c.page_number,
            }
            for c in chunks
        ]

        confidence = min(0.55 + 0.1 * len(top_chunks), 0.9)

        return {
            "answer": "\n".join(answer_lines),
            "citations": citations,
            "confidence": confidence,
        }
