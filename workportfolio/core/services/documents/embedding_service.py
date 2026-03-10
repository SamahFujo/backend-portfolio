"""
Embedding service abstraction.

Currently uses Jina embeddings API.
Later we can extend this to support Gemini or local embeddings.
"""

from typing import List
from django.conf import settings
import requests


class EmbeddingService:
    """
    Service responsible for generating embeddings.
    """

    JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
    MODEL_NAME = "jina-embeddings-v3"

    @classmethod
    def generate_embedding(cls, text: str) -> List[float]:
        """
        Generate embedding for a single text input.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot generate embedding for empty text.")

        embeddings = cls.generate_embeddings([text])
        if not embeddings:
            raise ValueError("Embedding generation returned no result.")

        return embeddings[0]

    @classmethod
    def generate_embeddings(cls, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Jina API.
        """
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        if not cleaned_texts:
            return []

        if not settings.JINA_API_KEY:
            raise ValueError("JINA_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": cls.MODEL_NAME,
            "input": cleaned_texts,
        }

        response = requests.post(
            cls.JINA_EMBEDDINGS_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()

        data = result.get("data", [])
        if not data:
            raise ValueError("No embedding data returned from Jina API.")

        # Sort by index to preserve input order
        data = sorted(data, key=lambda item: item.get("index", 0))

        return [item["embedding"] for item in data]