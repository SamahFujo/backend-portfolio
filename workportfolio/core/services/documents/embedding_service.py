"""
Embedding service abstraction.

Currently uses Jina embeddings API.
Later we can extend this to support Gemini or local embeddings.
"""

from typing import List, Optional
from django.conf import settings
import requests


class EmbeddingService:
    """
    Service responsible for generating embeddings.
    """

    JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
    MODEL_NAME = "jina-embeddings-v3"

    # Safe defaults for production use
    DEFAULT_TIMEOUT = 60
    MAX_TEXT_LENGTH = 8000
    DEFAULT_BATCH_SIZE = 32

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """
        Normalize and truncate text before sending it to the embedding API.
        """
        text = (text or "").replace("\r", " ").replace("\n", " ").strip()
        text = " ".join(text.split())
        if not text:
            return ""
        if len(text) > cls.MAX_TEXT_LENGTH:
            text = text[:cls.MAX_TEXT_LENGTH].rstrip()
        return text

    @classmethod
    def generate_embedding(cls, text: str, task: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text input.

        Args:
            text (str): Input text
            task (Optional[str]): Optional embedding task hint, such as
                'retrieval.query' or 'retrieval.passage'

        Returns:
            List[float]: Embedding vector
        """
        cleaned = cls._clean_text(text)
        if not cleaned:
            raise ValueError("Cannot generate embedding for empty text.")

        embeddings = cls.generate_embeddings([cleaned], task=task)
        if not embeddings:
            raise ValueError("Embedding generation returned no result.")

        return embeddings[0]

    @classmethod
    def generate_embeddings(
        cls,
        texts: List[str],
        task: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Jina API.

        Args:
            texts (List[str]): Input texts
            task (Optional[str]): Optional embedding task hint
            batch_size (Optional[int]): Batch size for API calls

        Returns:
            List[List[float]]: List of embedding vectors
        """
        cleaned_texts = []

        for text in texts:
            cleaned = cls._clean_text(text)

            if not cleaned:
                raise ValueError("Cannot generate embeddings because one chunk is empty.")

            cleaned_texts.append(cleaned)
            
        if not cleaned_texts:
            return []

        if not settings.JINA_API_KEY:
            raise ValueError("JINA_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

        batch_size = batch_size or cls.DEFAULT_BATCH_SIZE
        all_embeddings: List[List[float]] = []

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i + batch_size]

            payload = {
                "model": cls.MODEL_NAME,
                "input": batch,
            }

            # Optional task hint for retrieval quality
            if task:
                payload["task"] = task

            try:
                response = requests.post(
                    cls.JINA_EMBEDDINGS_URL,
                    headers=headers,
                    json=payload,
                    timeout=cls.DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(
                    f"Jina embeddings request failed: {str(e)}") from e

            result = response.json()
            data = result.get("data", [])

            if not data:
                raise ValueError("No embedding data returned from Jina API.")

            # Preserve input order
            data = sorted(data, key=lambda item: item.get("index", 0))
            batch_embeddings = [item.get("embedding")
                                for item in data if item.get("embedding")]

            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    f"Embedding count mismatch. Sent {len(batch)} texts but received {len(batch_embeddings)} embeddings."
                )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings
