from __future__ import annotations

import json
from typing import List, Dict, Any
from django.conf import settings

from core.models import DocumentChunk
from core.services.llm.gemini_router import GeminiRouter


class GeminiGroundedAnswerer:
    """
    Produces the final answer ONLY from evidence chunks.
    Returns verdict + used_chunk_indices so citations are precise.

    Verdict rules:
    - If question is yes/no -> verdict must be "yes" or "no"
    - Otherwise -> verdict is "supported" or "not_enough_evidence"
    """

    @staticmethod
    def _snippet(text: str, max_len: int = 900) -> str:
        t = (text or "").replace("\r", " ").strip()
        t = " ".join(t.split())
        return t if len(t) <= max_len else t[:max_len].rstrip() + "..."

    @staticmethod
    def _is_yes_no_question(question: str) -> bool:
        q = (question or "").strip().lower()
        # simple heuristic (good enough)
        return q.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "should ", "has ", "have ", "was ", "were "))

    @classmethod
    def answer(cls, question: str, evidence_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        if not evidence_chunks:
            return {
                "verdict": "not_enough_evidence",
                "answer": "I don’t have enough evidence in the uploaded documents to answer that.",
                "bullets": [],
                "used_chunk_indices": [],
                "meta": {"model_used": None, "tried_models": [], "error": "no_evidence"},
            }

        evidence_lines = []
        for i, c in enumerate(evidence_chunks):
            evidence_lines.append(
                f"[{i}] {c.document.title} (chunk {c.chunk_index}): {cls._snippet(c.content)}"
            )
        evidence_text = "\n".join(evidence_lines)

        is_yes_no = cls._is_yes_no_question(question)

        # ✅ System instruction: stricter grounding
        system_instruction = (
            "You answer questions about Samah using ONLY the provided evidence.\n"
            "Rules:\n"
            "1) Do NOT invent details.\n"
            "2) Do NOT use subjective praise unless it is explicitly stated in evidence.\n"
            "3) If evidence is insufficient, verdict MUST be not_enough_evidence.\n"
            "4) If the question is yes/no, verdict MUST be yes or no.\n"
            "5) If the question is NOT yes/no, verdict MUST be supported or not_enough_evidence.\n"
            "6) Use at most 6 evidence chunks.\n"
            "7) Return JSON only.\n"
        )

        # ✅ Prompt: include supported verdict
        prompt = (
            "Return JSON with keys:\n"
            "- verdict: for yes/no questions -> one of ['yes','no','not_enough_evidence']\n"
            "          for non-yes/no questions -> one of ['supported','not_enough_evidence']\n"
            "- answer: concise answer (1-3 sentences)\n"
            "- bullets: 0-6 bullets that add NEW details not already stated in 'answer'\n"
            "- used_chunk_indices: array of integers referencing evidence chunks used (max 6)\n\n"
            "Rules:\n"
            "A) Do NOT repeat the same idea in different words.\n"
            "B) Bullets must be non-overlapping and each bullet must be unique.\n"
            "C) If the answer is already complete, bullets can be an empty array.\n\n"
            f"Question (yes/no = {str(is_yes_no).lower()}): {question}\n\n"
            f"Evidence:\n{evidence_text}\n"
        )

        # Router chain: primary -> fallbacks
        chain = [settings.GEMINI_PRIMARY_MODEL] + \
            getattr(settings, "GEMINI_FALLBACK_MODELS", [])

        ok, text, meta = GeminiRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,
            model_chain=chain,
        )

        if not ok:
            fallback_used = list(range(min(3, len(evidence_chunks))))
            fallback_lines = [
                f"- {cls._snippet(evidence_chunks[i].content, 220)}" for i in fallback_used]
            return {
                "verdict": "not_enough_evidence",
                "answer": (
                    f"Gemini is temporarily unavailable ({meta.get('error')}). "
                    "Here is the best evidence I found:\n" +
                    "\n".join(fallback_lines)
                ),
                "bullets": [],
                "used_chunk_indices": fallback_used,
                "meta": meta,
            }

        # Parse JSON
        try:
            data = json.loads(text)
        except Exception:
            return {
                "verdict": "not_enough_evidence",
                "answer": "I couldn’t reliably format an answer. Please try again.",
                "bullets": [],
                "used_chunk_indices": [],
                "meta": meta,
            }

        verdict = data.get("verdict", "not_enough_evidence")

        # ✅ Validate verdict depending on question type
        if is_yes_no:
            allowed = {"yes", "no", "not_enough_evidence"}
        else:
            allowed = {"supported", "not_enough_evidence"}

        if verdict not in allowed:
            verdict = "not_enough_evidence"

        used = []
        for idx in data.get("used_chunk_indices", []) or []:
            if isinstance(idx, int) and 0 <= idx < len(evidence_chunks):
                used.append(idx)
            if len(used) >= 6:
                break

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
            "meta": meta,  # ✅ always returned now
        }
