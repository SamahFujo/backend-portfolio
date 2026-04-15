from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from google import genai
from google.genai import types
import requests


class LLMRouter:
    """
    Multi-provider router.

    Supported providers:
    - Gemini
    - DeepSeek

    Routes requests by task:
    - rewrite         -> GEMINI_REWRITE_API_KEY
    - grounded_answer -> GEMINI_GROUNDED_API_KEY / DEEPSEEK_API_KEY
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
        chain = [m for m in (model_chain or []) if m]
        if not chain:
            return False, "", {
                "provider": None,
                "model_used": None,
                "tried_models": [],
                "error": "empty_model_chain",
            }

        tried_models: List[str] = []
        last_meta: Dict[str, Any] = {
            "provider": None,
            "model_used": None,
            "tried_models": [],
            "task": task,
            "error": "all_models_failed",
        }

        for model_name in chain:
            tried_models.append(model_name)
            provider = cls._detect_provider(model_name)

            if provider == "gemini":
                api_key = cls._get_gemini_api_key_for_task(task)
                if not api_key:
                    last_meta = {
                        "provider": "gemini",
                        "model_used": model_name,
                        "tried_models": tried_models[:],
                        "task": task,
                        "error": f"missing_gemini_api_key_for_task:{task}",
                    }
                    continue

                ok, text, meta = cls._call_gemini_json(
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    json_schema=json_schema,
                )

            elif provider == "deepseek":
                api_key = cls._get_deepseek_api_key()
                if not api_key:
                    last_meta = {
                        "provider": "deepseek",
                        "model_used": model_name,
                        "tried_models": tried_models[:],
                        "task": task,
                        "error": "missing_deepseek_api_key",
                    }
                    continue

                ok, text, meta = cls._call_deepseek_json(
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )

            else:
                last_meta = {
                    "provider": "unknown",
                    "model_used": model_name,
                    "tried_models": tried_models[:],
                    "task": task,
                    "error": f"unsupported_provider_for_model:{model_name}",
                }
                continue

            meta["tried_models"] = tried_models[:]
            meta["task"] = task
            last_meta = meta

            if ok:
                return True, text, meta

        return False, "", last_meta


    @classmethod
    def generate_text(
        cls,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
        model_chain: Optional[List[str]] = None,
        task: str = TASK_GROUNDED_ANSWER,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        chain = [m for m in (model_chain or []) if m]
        if not chain:
            return False, "", {
                "provider": None,
                "model_used": None,
                "tried_models": [],
                "error": "empty_model_chain",
            }

        tried_models: List[str] = []
        last_meta: Dict[str, Any] = {
            "provider": None,
            "model_used": None,
            "tried_models": [],
            "task": task,
            "error": "all_models_failed",
        }

        for model_name in chain:
            tried_models.append(model_name)
            provider = cls._detect_provider(model_name)

            if provider == "gemini":
                api_key = cls._get_gemini_api_key_for_task(task)
                if not api_key:
                    last_meta = {
                        "provider": "gemini",
                        "model_used": model_name,
                        "tried_models": tried_models[:],
                        "task": task,
                        "error": f"missing_gemini_api_key_for_task:{task}",
                    }
                    continue

                ok, text, meta = cls._call_gemini_text(
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )

            elif provider == "deepseek":
                api_key = cls._get_deepseek_api_key()
                if not api_key:
                    last_meta = {
                        "provider": "deepseek",
                        "model_used": model_name,
                        "tried_models": tried_models[:],
                        "task": task,
                        "error": "missing_deepseek_api_key",
                    }
                    continue

                ok, text, meta = cls._call_deepseek_text(
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )

            else:
                last_meta = {
                    "provider": "unknown",
                    "model_used": model_name,
                    "tried_models": tried_models[:],
                    "task": task,
                    "error": f"unsupported_provider_for_model:{model_name}",
                }
                continue

            meta["tried_models"] = tried_models[:]
            meta["task"] = task
            last_meta = meta

            if ok:
                return True, text, meta

        return False, "", last_meta

    @staticmethod
    def _detect_provider(model_name: str) -> str:
        m = (model_name or "").strip().lower()

        if m.startswith("gemini"):
            return "gemini"

        if m.startswith("deepseek"):
            return "deepseek"

        return "unknown"

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
    def _call_deepseek_text(
        cls,
        api_key: str,
        model_name: str,
        prompt: str,
        system_instruction: str,
        temperature: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        try:
            url = getattr(settings, "DEEPSEEK_BASE_URL",
                          "https://api.deepseek.com/chat/completions")

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction or ""},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }

            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=getattr(settings, "LLM_HTTP_TIMEOUT", 60),
            )

            if response.status_code >= 400:
                return False, "", {
                    "provider": "deepseek",
                    "model_used": model_name,
                    "error": f"http_{response.status_code}:{response.text}",
                }

            data = response.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not text:
                return False, "", {
                    "provider": "deepseek",
                    "model_used": model_name,
                    "error": "empty_response_text",
                }

            return True, text, {
                "provider": "deepseek",
                "model_used": model_name,
                "error": None,
            }

        except Exception as exc:
            return False, "", {
                "provider": "deepseek",
                "model_used": model_name,
                "error": str(exc),
            }

    @classmethod
    def _call_deepseek_json(
        cls,
        api_key: str,
        model_name: str,
        prompt: str,
        system_instruction: str,
        temperature: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        ok, text, meta = cls._call_deepseek_text(
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )

        if not ok:
            return ok, text, meta

        try:
            json.loads(text)
        except Exception as exc:
            return False, "", {
                "provider": "deepseek",
                "model_used": model_name,
                "error": f"invalid_json_response:{exc}",
            }

        return True, text, meta

    @classmethod
    def _get_gemini_api_key_for_task(cls, task: str) -> str:
        if task == cls.TASK_REWRITE:
            return getattr(settings, "GEMINI_REWRITE_API_KEY", "")

        if task == cls.TASK_GROUNDED_ANSWER:
            return getattr(settings, "GEMINI_GROUNDED_API_KEY", "")

        return getattr(settings, "GEMINI_GROUNDED_API_KEY", "")

    @classmethod
    def _get_deepseek_api_key(cls) -> str:
        return getattr(settings, "DEEPSEEK_API_KEY", "")
