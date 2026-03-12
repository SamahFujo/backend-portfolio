from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from django.conf import settings

from core.services.llm.gemini_router import GeminiRouter


@dataclass(frozen=True)
class DocTypeResult:
    doc_type: str
    confidence: float
    tags: List[str]
    source: str  # "rules" | "gemini" | "fallback"


class DocumentTypeClassifier:
    """
    Classifies an uploaded document into a document_type and tags.

    Strategy:
    1) Rules first (fast, free)
    2) Gemini fallback only if rules are uncertain
    """

    # --- Rule patterns ---
    CV_HINTS = [
        r"\bskills\b", r"\beducation\b", r"\bexperience\b", r"\bsummary\b",
        r"\blinkedin\b", r"\bphone\b", r"\bemail\b", r"\bgpa\b"
    ]

    CERT_HINTS = [
        r"\bcertificate\b", r"\bcertification\b", r"\bissued by\b",
        r"\bcredential\b", r"\bcompletion\b", r"\bachievement\b"
    ]

    PROJECT_HINTS = [
        r"\bproject\b", r"\bdashboard\b", r"\bchatbot\b", r"\barchitecture\b",
        r"\btech stack\b", r"\bdeployment\b", r"\bbackend\b", r"\bfrontend\b"
    ]

    RECOMMENDATION_HINTS = [
        r"\bto whom it may concern\b",
        r"\brecommendation\b",
        r"\breference letter\b",
        r"\bi recommend\b",
        r"\bhas been a\b",
        r"\bSincerely\b",
        r"\bRegards\b",
    ]

    EXPERIENCE_LETTER_HINTS = [
        r"\bexperience letter\b",
        r"\bemployment letter\b",
        r"\bto whom it may concern\b",
        r"\bthis is to certify\b",
        r"\bemployment verification\b",
        r"\bhas been employed\b",
        r"\bwas employed\b",
        r"\bjoining date\b",
        r"\bdate of joining\b",
        r"\blast working day\b",
        r"\bemployment period\b",
        r"\bposition\b",
        r"\bjob title\b",
        r"\bhr department\b",
    ]

    @classmethod
    def classify(cls, title: str, raw_text: str) -> DocTypeResult:
        """
        Return the predicted document_type + tags.
        """
        text = (raw_text or "").strip()
        t = (title or "").lower().strip()

        # If no text, we cannot classify properly (likely scanned PDF).
        if not text:
            # Still use title heuristics
            return cls._classify_from_title_only(t)

        # 1) Rule-based classification
        rule_result = cls._rule_based(text, t)
        if rule_result.confidence >= 0.75:
            return rule_result

        # 2) Gemini fallback (ONLY if key exists)
        if getattr(settings, "GEMINI_API_KEY", None):
            gemini_result = cls._gemini_classify(title=title, raw_text=text)
            if gemini_result:
                return gemini_result

        # 3) Final fallback
        return DocTypeResult(doc_type="other", confidence=0.4, tags=["unclassified"], source="fallback")

    @classmethod
    def _classify_from_title_only(cls, title_lower: str) -> DocTypeResult:
        tags: List[str] = []
        if "cv" in title_lower or "resume" in title_lower:
            return DocTypeResult("cv", 0.65, ["cv"], "rules")
        if "certificate" in title_lower or "cert" in title_lower:
            return DocTypeResult("certificates", 0.65, ["certificates"], "rules")
        if "project" in title_lower:
            return DocTypeResult("projects", 0.60, ["projects"], "rules")
        if "recommend" in title_lower or "reference" in title_lower:
            return DocTypeResult("recommendation", 0.60, ["recommendation"], "rules")
        if "experience" in title_lower or "employment" in title_lower:
            return DocTypeResult("experience_letter", 0.60, ["experience_letter", "employment"], "rules")
        return DocTypeResult("other", 0.4, ["unclassified"], "fallback")

    @classmethod
    def _score(cls, text_lower: str, patterns: List[str]) -> int:
        score = 0
        for p in patterns:
            if re.search(p, text_lower, re.I):
                score += 1
        return score

    @classmethod
    def _rule_based(cls, text: str, title_lower: str) -> DocTypeResult:
        text_lower = text.lower()

        cv_score = cls._score(text_lower, cls.CV_HINTS)
        cert_score = cls._score(text_lower, cls.CERT_HINTS)
        proj_score = cls._score(text_lower, cls.PROJECT_HINTS)
        rec_score = cls._score(text_lower, cls.RECOMMENDATION_HINTS)
        exp_score = cls._score(text_lower, cls.EXPERIENCE_LETTER_HINTS)

        # Boost from title hints
        if "cv" in title_lower or "resume" in title_lower:
            cv_score += 2
        if "certificate" in title_lower or "cert" in title_lower:
            cert_score += 2
        if "project" in title_lower:
            proj_score += 2
        if "recommend" in title_lower or "reference" in title_lower:
            rec_score += 2
        if "experience" in title_lower or "employment" in title_lower:
            exp_score += 2

        scores = {
            "cv": cv_score,
            "certificates": cert_score,
            "projects": proj_score,
            "recommendation": rec_score,
            "experience_letter": exp_score,
        }

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total = sum(scores.values()) or 1

        # Confidence heuristic
        confidence = min(0.95, 0.45 + (best_score / max(3, total)))

        tags = [best_type]
        if best_type == "projects":
            tags += ["portfolio", "solutions"]
        if best_type == "cv":
            tags += ["profile"]
        if best_type == "recommendation":
            tags += ["reference", "endorsement"]
        if best_type == "certificates":
            tags += ["credentials"]
        if best_type == "experience_letter":
            tags += ["employment", "verification", "hr"]

        return DocTypeResult(best_type, confidence, tags, "rules")

    @classmethod
    def _gemini_classify(cls, title: str, raw_text: str) -> Optional[DocTypeResult]:
        # Keep evidence short to reduce cost
        snippet = " ".join(raw_text.split())[:4000]

        system_instruction = (
            "Classify the document type.\n"
            "Allowed document_type values: cv, projects, certificates, recommendation, experience_letter, other.\n"
            "Return JSON only."
        )

        prompt = (
            "Return JSON:\n"
            "{"
            "\"document_type\":\"cv|projects|certificates|recommendation|experience_letter|other\","
            "\"confidence\":0.0-1.0,"
            "\"tags\":[\"...\"]"
            "}\n\n"
            f"Title: {title}\n\n"
            f"Content snippet:\n{snippet}\n"
        )

        # Use your REWRITE chain for this classification (cheap)
        chain = [settings.GEMINI_REWRITE_PRIMARY] + \
            getattr(settings, "GEMINI_REWRITE_FALLBACKS", [])
        ok, text, meta = GeminiRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.0,
            model_chain=chain,
        )
        if not ok:
            return None

        try:
            data = json.loads(text)
            doc_type = data.get("document_type", "other")
            conf = float(data.get("confidence", 0.6))
            tags = data.get("tags", []) or []
            if doc_type not in {"cv", "projects", "certificates", "recommendation", "experience_letter", "other"}:
                doc_type = "other"
            tags = [str(t).strip() for t in tags if str(t).strip()][:10]
            if doc_type not in tags:
                tags.insert(0, doc_type)
            return DocTypeResult(doc_type, max(0.0, min(1.0, conf)), tags, "gemini")
        except Exception:
            return None
