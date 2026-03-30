from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import requests
from django.conf import settings


class OllamaRouter:
    """
    Local Ollama JSON generator.

    Standard return:
        ok: bool
        text: str
        meta: dict
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
        models = model_chain or [getattr(settings, "OLLAMA_PRIMARY_MODEL", "qwen2.5:14b-instruct")]
        url = getattr(settings, "OLLAMA_API_URL", "http://localhost:11434/api/generate")

        tried_models: List[str] = []
        last_error = None

        for model in models:
            tried_models.append(model)

            try:
                payload = {
                    "model": model,
                    "prompt": f"{system_instruction}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                }

                # If schema is passed, ask Ollama to force structured JSON output
                if json_schema:
                    payload["format"] = json_schema
                else:
                    payload["format"] = "json"

                response = requests.post(url, json=payload, timeout=180)
                response.raise_for_status()

                data = response.json()
                text = (data.get("response") or "").strip()

                if not text:
                    last_error = f"empty_response_from_model:{model}"
                    continue

                # Validate JSON early so callers get predictable behavior
                json.loads(text)

                return True, text, {
                    "provider": "ollama",
                    "model_used": model,
                    "tried_models": tried_models,
                    "error": None,
                }

            except Exception as e:
                last_error = str(e)

        return False, "", {
            "provider": "ollama",
            "model_used": None,
            "tried_models": tried_models,
            "error": last_error or "unknown_ollama_error",
        }