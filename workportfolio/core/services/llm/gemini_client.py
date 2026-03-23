from __future__ import annotations

from django.conf import settings
from google import genai


class GeminiClient:
    """
    Centralized Gemini client factory.
    Reuses one configured client instance across the app.
    """

    _client = None

    @classmethod
    def client(cls):
        """
        Return a configured Gemini client instance.
        """
        if cls._client is None:
            api_key = getattr(settings, "GEMINI_API_KEY", None)

            if not api_key:
                raise ValueError("GEMINI_API_KEY is missing in Django settings.")

            cls._client = genai.Client(api_key=api_key)

        return cls._client