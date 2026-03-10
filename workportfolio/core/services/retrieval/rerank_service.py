from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
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

    @classmethod
    def rerank(
        cls,
        query: str,
        documents: List[str],
        top_n: int = 5,
        timeout: int = 60
    ) -> List[RerankItem]:
        query = (query or "").strip()
        if not query:
            return []

        if not documents:
            return []

        if not settings.JINA_API_KEY:
            raise ValueError("JINA_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": cls.MODEL_NAME,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }

        r = requests.post(cls.JINA_RERANK_URL, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", []) or []
        items: List[RerankItem] = []

        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score")
            if isinstance(idx, int) and isinstance(score, (int, float)):
                items.append(RerankItem(index=idx, score=float(score)))

        # Keep them ordered as the API returns (typically best first).
        return items