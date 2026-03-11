from __future__ import annotations

import json
from functools import lru_cache
from django.conf import settings

from core.services.llm.gemini_client import GeminiClient
from core.services.llm.gemini_safe import gemini_call_safe


class GeminiQueryRewriter:
    @classmethod
    @lru_cache(maxsize=512)
    def rewrite_cached(cls, user_query: str) -> dict:
        q = (user_query or "").strip()
        if not q:
            return {"rewritten_query": "", "notes": "empty"}

        # If Gemini key missing, fallback immediately
        if not getattr(settings, "GEMINI_API_KEY", None):
            return {"rewritten_query": cls._local_rewrite(q), "notes": "no_gemini_key"}

        def _call():
            client = GeminiClient.client()
            resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=(
                    "Rewrite the query for retrieval: fix typos, keep meaning, add useful keywords.\n"
                    "Return JSON: {rewritten_query, notes}\n\n"
                    f"Query: {q}"
                ),
                config={"response_mime_type": "application/json",
                        "temperature": 0.1},
            )
            return resp.text

        ok, text, err = gemini_call_safe(_call, max_retries=0)
        if not ok:
            return {"rewritten_query": cls._local_rewrite(q), "notes": f"gemini_{err}_fallback"}

        try:
            data = json.loads(text)
            rewritten = (data.get("rewritten_query") or "").strip() or q
            return {"rewritten_query": rewritten, "notes": data.get("notes", "ok")}
        except Exception:
            return {"rewritten_query": cls._local_rewrite(q), "notes": "json_parse_failed_fallback"}

    @staticmethod
    def _local_rewrite(q: str) -> str:
        # tiny local typo fixes that help a lot
        return (
            q.replace(" now ", " know ")
            .replace(" certficate", " certificate")
            .replace(" certifcate", " certificate")
            .strip()
        )
