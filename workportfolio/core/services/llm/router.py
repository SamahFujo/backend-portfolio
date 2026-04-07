from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from google import genai
from google.genai import types


class LLMRouter:
    """
    Gemini-only router.

    Routes requests by task:
    - rewrite         -> GEMINI_REWRITE_API_KEY
    - grounded_answer -> GEMINI_GROUNDED_API_KEY

    Keeps the same general generate_json(...) interface so the rest of the
    codebase needs only minimal changes.
    """

    TASK_REWRITE = "rewrite"
    TASK_GROUNDED_ANSWER = "grounded_answer"

    @classmethod
    def generate_json(
        cls,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.1,
        model_chain: Optional[List[str]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        task: str = TASK_GROUNDED_ANSWER,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Try Gemini models in order until one succeeds.

        Returns:
            (ok, text, meta)
        """
        chain = [m for m in (model_chain or []) if m]
        if not chain:
            return False, "", {
                "provider": "gemini",
                "model_used": None,
                "tried_models": [],
                "error": "empty_model_chain",
            }

        api_key = cls._get_api_key_for_task(task)
        if not api_key:
            return False, "", {
                "provider": "gemini",
                "model_used": None,
                "tried_models": chain,
                "error": f"missing_api_key_for_task:{task}",
            }

        tried_models: List[str] = []

        for model_name in chain:
            tried_models.append(model_name)

            ok, text, meta = cls._call_gemini_json(
                api_key=api_key,
                model_name=model_name,
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                json_schema=json_schema,
            )

            meta["tried_models"] = tried_models[:]
            meta["task"] = task

            if ok:
                return True, text, meta

        return False, "", {
            "provider": "gemini",
            "model_used": None,
            "tried_models": tried_models,
            "task": task,
            "error": "all_models_failed",
        }

    @classmethod
    def generate_text(
        cls,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
        model_chain: Optional[List[str]] = None,
        task: str = TASK_GROUNDED_ANSWER,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Optional plain-text generation helper for grounded answers or other
        places where you do not need JSON schema output.
        """
        chain = [m for m in (model_chain or []) if m]
        if not chain:
            return False, "", {
                "provider": "gemini",
                "model_used": None,
                "tried_models": [],
                "error": "empty_model_chain",
            }

        api_key = cls._get_api_key_for_task(task)
        if not api_key:
            return False, "", {
                "provider": "gemini",
                "model_used": None,
                "tried_models": chain,
                "error": f"missing_api_key_for_task:{task}",
            }

        tried_models: List[str] = []

        for model_name in chain:
            tried_models.append(model_name)

            ok, text, meta = cls._call_gemini_text(
                api_key=api_key,
                model_name=model_name,
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
            )

            meta["tried_models"] = tried_models[:]
            meta["task"] = task

            if ok:
                return True, text, meta

        return False, "", {
            "provider": "gemini",
            "model_used": None,
            "tried_models": tried_models,
            "task": task,
            "error": "all_models_failed",
        }

    @classmethod
    def _call_gemini_json(
        cls,
        api_key: str,
        model_name: str,
        prompt: str,
        system_instruction: str,
        temperature: float,
        json_schema: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        try:
            client = genai.Client(api_key=api_key)

            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction or None,
                response_mime_type="application/json",
                response_json_schema=json_schema or None,
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            text = (getattr(response, "text", None) or "").strip()
            if not text:
                return False, "", {
                    "provider": "gemini",
                    "model_used": model_name,
                    "error": "empty_response_text",
                }

            # Validate JSON shape early so callers receive predictable data.
            try:
                json.loads(text)
            except Exception as exc:
                return False, "", {
                    "provider": "gemini",
                    "model_used": model_name,
                    "error": f"invalid_json_response:{exc}",
                }

            return True, text, {
                "provider": "gemini",
                "model_used": model_name,
                "error": None,
            }

        except Exception as exc:
            return False, "", {
                "provider": "gemini",
                "model_used": model_name,
                "error": str(exc),
            }

    @classmethod
    def _call_gemini_text(
        cls,
        api_key: str,
        model_name: str,
        prompt: str,
        system_instruction: str,
        temperature: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        try:
            client = genai.Client(api_key=api_key)

            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction or None,
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            text = (getattr(response, "text", None) or "").strip()
            if not text:
                return False, "", {
                    "provider": "gemini",
                    "model_used": model_name,
                    "error": "empty_response_text",
                }

            return True, text, {
                "provider": "gemini",
                "model_used": model_name,
                "error": None,
            }

        except Exception as exc:
            return False, "", {
                "provider": "gemini",
                "model_used": model_name,
                "error": str(exc),
            }

    @classmethod
    def _get_api_key_for_task(cls, task: str) -> str:
        if task == cls.TASK_REWRITE:
            return getattr(settings, "GEMINI_REWRITE_API_KEY", "")

        if task == cls.TASK_GROUNDED_ANSWER:
            return getattr(settings, "GEMINI_GROUNDED_API_KEY", "")

        # Safe default: use grounded key for unknown tasks.
        return getattr(settings, "GEMINI_GROUNDED_API_KEY", "")
