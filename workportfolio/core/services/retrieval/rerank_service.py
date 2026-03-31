from __future__ import annotations

from dataclasses import dataclass
from typing import List
from django.conf import settings
import requests


@dataclass(frozen=True)
class RerankItem:
    """
    A single rerank result item.

    index = index of the document in the input list
    score = relevance score assigned by the reranker
    """
    index: int
    score: float


class RerankService:
    """
    Jina reranking service.

    It reranks a list of candidate texts for a given query.
    """

    JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
    MODEL_NAME = "jina-reranker-v2-base-multilingual"
    DEFAULT_TIMEOUT = 60
    MAX_DOCUMENT_LENGTH = 4000

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """
        Normalize whitespace and truncate long candidate text.
        """
        text = (text or "").replace("\r", " ").replace("\n", " ").strip()
        text = " ".join(text.split())
        if len(text) > cls.MAX_DOCUMENT_LENGTH:
            text = text[:cls.MAX_DOCUMENT_LENGTH].rstrip()
        return text

    @classmethod
    def rerank(
        cls,
        query: str,
        documents: List[str],
        top_n: int = 5,
        timeout: int = 60,
    ) -> List[RerankItem]:
        query = (query or "").strip()
        if not query:
            return []

        if not documents:
            return []

        if not settings.JINA_API_KEY:
            raise ValueError("JINA_API_KEY is not configured.")

        cleaned_documents = [cls._clean_text(doc) for doc in documents]
        cleaned_documents = [doc for doc in cleaned_documents if doc]

        if not cleaned_documents:
            return []

        headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": cls.MODEL_NAME,
            "query": query,
            "documents": cleaned_documents,
            "top_n": min(top_n, len(cleaned_documents)),
        }

        try:
            r = requests.post(
                cls.JINA_RERANK_URL,
                headers=headers,
                json=payload,
                timeout=timeout or cls.DEFAULT_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException:
            # Return empty so caller can safely fallback to vector ranking
            return []

        results = data.get("results", []) or []
        items: List[RerankItem] = []
        seen_indexes = set()

        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score")

            if not isinstance(idx, int):
                continue
            if not isinstance(score, (int, float)):
                continue
            if idx < 0 or idx >= len(cleaned_documents):
                continue
            if idx in seen_indexes:
                continue

            seen_indexes.add(idx)
            items.append(RerankItem(index=idx, score=float(score)))

        return items
