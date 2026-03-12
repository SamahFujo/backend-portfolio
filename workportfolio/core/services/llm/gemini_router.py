from __future__ import annotations

from typing import Any, Dict, List, Tuple
from django.conf import settings
from google.genai.errors import ClientError

from core.services.llm.gemini_client import GeminiClient


class GeminiRouter:
    """
    Tries a chain of Gemini models in order.
    On 429 quota/rate limit it tries the next model.
    """

    @staticmethod
    def _dedupe(models: List[str]) -> List[str]:
        seen = set()
        out = []
        for m in models:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    @classmethod
    def generate_json(
        cls,
        prompt: str,
        system_instruction: str,
        temperature: float,
        model_chain: List[str],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Returns:
        ok: bool
        text: response text (JSON string)
        meta: {model_used, error, tried_models}
        """
        client = GeminiClient.client()
        tried = []

        for model in cls._dedupe(model_chain):
            tried.append(model)
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        "response_mime_type": "application/json",
                        "temperature": temperature,
                    },
                )
                return True, resp.text, {"model_used": model, "error": None, "tried_models": tried}

            except ClientError as e:
                if getattr(e, "status_code", None) == 429:
                    # quota exceeded -> try next model
                    continue
                return False, "", {"model_used": model, "error": f"client_error:{e}", "tried_models": tried}

            except Exception as e:
                return False, "", {"model_used": model, "error": f"unknown_error:{e}", "tried_models": tried}

        return False, "", {"model_used": None, "error": "quota_exhausted_all_models", "tried_models": tried}
