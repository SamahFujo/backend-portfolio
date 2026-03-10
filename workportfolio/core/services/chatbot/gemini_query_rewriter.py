from __future__ import annotations

import json
from typing import Dict, Any
from django.conf import settings
from core.services.llm.gemini_client import GeminiClient


class GeminiQueryRewriter:
    """
    Rewrites the user query into a retrieval-optimized query:
    - fixes typos
    - expands abbreviations
    - adds synonyms / keywords
    - keeps the same intent
    """

    @classmethod
    def rewrite(cls, user_query: str) -> Dict[str, Any]:
        q = (user_query or "").strip()
        if not q:
            return {"rewritten_query": "", "notes": "empty"}

        system_instruction = (
            "You rewrite user queries to improve document retrieval.\n"
            "Rules:\n"
            "1) Keep the SAME meaning/intent.\n"
            "2) Fix typos.\n"
            "3) Add relevant keywords/synonyms.\n"
            "4) Output JSON only.\n"
        )

        prompt = (
            "Return JSON with keys:\n"
            "- rewritten_query: string (optimized for retrieval)\n"
            "- keywords: array of strings (optional)\n"
            "- notes: short string\n\n"
            f"User query: {q}\n"
        )

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

        try:
            data = json.loads(resp.text)
            rewritten = (data.get("rewritten_query") or "").strip()
            if not rewritten:
                rewritten = q
            return {
                "rewritten_query": rewritten,
                "keywords": data.get("keywords", []),
                "notes": data.get("notes", ""),
            }
        except Exception:
            # Fallback: use original query if parsing fails
            return {"rewritten_query": q, "keywords": [], "notes": "json_parse_failed"}