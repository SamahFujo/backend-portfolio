from __future__ import annotations

import json
from typing import List, Dict, Any
from django.conf import settings

from core.models import DocumentChunk
from core.services.llm.gemini_client import GeminiClient
from core.services.llm.gemini_safe import gemini_call_safe


class GeminiGroundedAnswerer:
    """
    Produces the final answer ONLY from evidence chunks.
    Returns verdict + used_chunk_indices so citations are precise.
    """

    @staticmethod
    def _snippet(text: str, max_len: int = 900) -> str:
        t = (text or "").replace("\r", " ").strip()
        t = " ".join(t.split())
        return t if len(t) <= max_len else t[:max_len].rstrip() + "..."

    @classmethod
    def answer(cls, question: str, evidence_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        if not evidence_chunks:
            return {
                "verdict": "not_enough_evidence",
                "answer": "I don’t have enough evidence in the uploaded documents to answer that.",
                "bullets": [],
                "used_chunk_indices": [],
            }

        evidence_lines = []
        for i, c in enumerate(evidence_chunks):
            evidence_lines.append(
                f"[{i}] {c.document.title} (chunk {c.chunk_index}): {cls._snippet(c.content)}"
            )
        evidence_text = "\n".join(evidence_lines)

        system_instruction = (
            "You answer questions about Samah using ONLY the provided evidence.\n"
            "Rules:\n"
            "1) Do NOT invent details.\n"
            "2) If evidence is insufficient, verdict MUST be not_enough_evidence.\n"
            "3) If question is yes/no, verdict MUST be yes or no.\n"
            "4) Return JSON only.\n"
        )

        prompt = (
        "Return JSON with keys:\n"
        "- verdict: one of ['yes','no','not_enough_evidence']\n"
        "- answer: concise answer (1-3 sentences)\n"
        "- bullets: 3-6 bullets that add NEW details not already stated in 'answer'\n"
        "- used_chunk_indices: array of integers referencing evidence chunks used\n\n"
        "Rules:\n"
        "A) Do NOT repeat the same idea in different words.\n"
        "B) Bullets must be non-overlapping and each bullet must be unique.\n"
        "C) If the answer is already complete, bullets can be an empty array.\n\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_text}\n"
        )

        def _call():
            client = GeminiClient.client()
            resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            return resp.text


        ok, text, err = gemini_call_safe(_call, max_retries=0)

        # If Gemini is down / quota exceeded / error -> return fallback WITHOUT crashing
        if not ok:
            # fallback: return an evidence-based response (no LLM)
            fallback_used = list(range(min(3, len(evidence_chunks))))
            fallback_lines = []
            for i in fallback_used:
                fallback_lines.append(f"- {cls._snippet(evidence_chunks[i].content, 220)}")

            return {
                "verdict": "not_enough_evidence",
                "answer": (
                    "Gemini is temporarily unavailable (quota/rate limit). "
                    "Here is the best evidence I found:\n" + "\n".join(fallback_lines)
                ),
                "bullets": [],
                "used_chunk_indices": fallback_used,
            }

        # Parse JSON from Gemini
        try:
            data = json.loads(text)
        except Exception:
            return {
                "verdict": "not_enough_evidence",
                "answer": "I couldn’t reliably format an answer. Please try again.",
                "bullets": [],
                "used_chunk_indices": [],
            }

        verdict = data.get("verdict")
        if verdict not in {"yes", "no", "not_enough_evidence"}:
            verdict = "not_enough_evidence"

        used = []
        for idx in data.get("used_chunk_indices", []) or []:
            if isinstance(idx, int) and 0 <= idx < len(evidence_chunks):
                used.append(idx)

        bullets = data.get("bullets", []) or []
        bullets = [str(b).strip() for b in bullets if str(b).strip()][:6]

        answer = (data.get("answer") or "").strip()
        if not answer:
            answer = "I don’t have enough evidence in the uploaded documents to answer that."
            verdict = "not_enough_evidence"

        return {
            "verdict": verdict,
            "answer": answer,
            "bullets": bullets,
            "used_chunk_indices": used,
        }
