from __future__ import annotations

from typing import Any, Dict, List, Tuple
from django.conf import settings

from core.services.llm.gemini_router import GeminiRouter
from core.services.llm.ollama_router import OllamaRouter


class LLMRouter:
    """
    Provider-agnostic router.

    Supported providers:
    - gemini
    - ollama
    """

    @classmethod
    def generate_json(
        cls,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.1,
        model_chain: List[str] | None = None,
        json_schema: Dict[str, Any] | None = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        provider = getattr(settings, "LLM_PROVIDER", "gemini").lower().strip()

        if provider == "gemini":
            ok, text, meta = GeminiRouter.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                model_chain=model_chain or [],
            )
            if meta is None:
                meta = {}
            meta["provider"] = "gemini"
            return ok, text, meta

        if provider == "ollama":
            return OllamaRouter.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                model_chain=model_chain,
                json_schema=json_schema,
            )

        return False, "", {
            "provider": provider,
            "model_used": None,
            "tried_models": [],
            "error": f"unsupported_provider:{provider}",
        }