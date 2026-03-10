from django.conf import settings
from google import genai


class GeminiClient:
    @staticmethod
    def client() -> genai.Client:
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")
        return genai.Client(api_key=settings.GEMINI_API_KEY)