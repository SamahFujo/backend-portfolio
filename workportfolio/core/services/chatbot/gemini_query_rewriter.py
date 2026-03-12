from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, Any
from django.conf import settings

from core.services.llm.gemini_router import GeminiRouter


class GeminiQueryRewriter:
    @classmethod
    @lru_cache(maxsize=512)
    def rewrite_cached(cls, user_query: str) -> Dict[str, Any]:
        q = (user_query or "").strip()
        if not q:
            return {"rewritten_query": "", "notes": "empty"}

        # local fallback rewrite always available
        local = cls._local_rewrite(q)

        # if no key, return local
        if not getattr(settings, "GEMINI_API_KEY", None):
            return {"rewritten_query": local, "notes": "no_gemini_key"}

        system_instruction = (
            "You rewrite user queries to improve document retrieval.\n"
            "Rules:\n"
            "1) Keep the SAME meaning/intent.\n"
            "2) Fix typos.\n"
            "3) Add relevant keywords/synonyms.\n"
            "4) Return JSON only.\n"
        )

        prompt = (
            "Return JSON with keys:\n"
            "- rewritten_query: string\n"
            "- notes: short string\n\n"
            f"User query: {q}\n"
        )

        chain = [settings.GEMINI_REWRITE_PRIMARY] + \
            getattr(settings, "GEMINI_REWRITE_FALLBACKS", [])
        ok, text, meta = GeminiRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,
            model_chain=chain,
        )

        if not ok:
            return {"rewritten_query": local, "notes": f"rewrite_fallback:{meta.get('error')}", "meta": meta}

        try:
            data = json.loads(text)
            rewritten = (data.get("rewritten_query") or "").strip() or local
            return {"rewritten_query": rewritten, "notes": data.get("notes", "ok"), "meta": meta}
        except Exception:
            return {"rewritten_query": local, "notes": "rewrite_json_parse_failed", "meta": meta}

    @staticmethod
    def _local_rewrite(q: str) -> str:
        return (
            q.replace(" now ", " know ")
            .replace(" certficate", " certificate")
            .replace(" certifcate", " certificate")
            .strip()
        )
