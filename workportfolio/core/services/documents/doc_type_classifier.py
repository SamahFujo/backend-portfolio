from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List, Optional
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
        r"\bsincerely\b",
        r"\bregards\b",
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

    CAPABILITIES_HINTS = [
        r"\bwhat i can help with\b",
        r"\bwhat i can do confidently\b",
        r"\btypes of projects i can build\b",
        r"\btechnologies i use professionally\b",
        r"\bcapabilities\b",
        r"\bservices\b",
        r"\bai and llm solutions\b",
        r"\bbackend development\b",
        r"\bfull-stack web applications\b",
    ]

    PREFERENCES_HINTS = [
        r"\bpreferences and work style\b",
        r"\bfavorite programming language\b",
        r"\bfavourite programming language\b",
        r"\bpreferred backend framework\b",
        r"\bpreferred frontend stack\b",
        r"\bpreferred type of work\b",
        r"\bworking style\b",
        r"\bwork style\b",
        r"\bbackend vs frontend vs ai\b",
    ]

    COMPENSATION_HINTS = [
        r"\bcompensation and availability\b",
        r"\bexpected salary range\b",
        r"\bpreferred work type\b",
        r"\bpreferred work arrangement\b",
        r"\bavailability for opportunities\b",
        r"\bfreelance and project-based work\b",
        r"\bcompensation discussion style\b",
        r"\bsalary\b",
        r"\bpayment range\b",
    ]

    FAQ_HINTS = [
        r"\bfaq\b",
        r"\bfrequently asked questions\b",
        r"\bwhat does samah do\b",
        r"\bwhat are samah’s strongest technical areas\b",
        r"\bwhat is samah’s favorite programming language\b",
        r"\bcan samah build\b",
        r"\bis samah open to\b",
    ]

    ACHIEVEMENTS_HINTS = [
        r"\bachievements and impact\b",
        r"\bimpact through ai and automation\b",
        r"\bleadership and ownership impact\b",
        r"\bcustomer and stakeholder impact\b",
        r"\btype of value i add to projects\b",
        r"\bwhy my work stands out\b",
        r"\bprofessional strength in delivery\b",
    ]

    CAREER_TIMELINE_HINTS = [
        r"\bcareer timeline\b",
        r"\bearly technical and customer-facing experience\b",
        r"\bgrowth into full-stack and backend development\b",
        r"\bgrowth into ai and applied machine learning work\b",
        r"\bleadership progression\b",
        r"\bcurrent professional profile\b",
        r"\bcareer direction\b",
    ]

    ALLOWED_TYPES = {
        "cv",
        "projects",
        "certificates",
        "recommendation",
        "experience_letter",
        "capabilities",
        "preferences",
        "compensation",
        "faq",
        "achievements",
        "career_timeline",
        "other",
    }

    @classmethod
    def classify(cls, title: str, raw_text: str) -> DocTypeResult:
        """
        Return the predicted document_type + tags.
        """
        text = (raw_text or "").strip()
        t = (title or "").lower().strip()

        if not text:
            return cls._classify_from_title_only(t)

        rule_result = cls._rule_based(text, t)
        if rule_result.confidence >= 0.75:
            return rule_result

        if getattr(settings, "GEMINI_API_KEY", None):
            gemini_result = cls._gemini_classify(title=title, raw_text=text)
            if gemini_result:
                return gemini_result

        return DocTypeResult(
            doc_type="other",
            confidence=0.4,
            tags=["unclassified"],
            source="fallback"
        )

    @classmethod
    def _classify_from_title_only(cls, title_lower: str) -> DocTypeResult:
        if "faq" in title_lower:
            return DocTypeResult("faq", 0.72, ["faq"], "rules")

        if "preference" in title_lower or "work style" in title_lower:
            return DocTypeResult("preferences", 0.72, ["preferences"], "rules")

        if "compensation" in title_lower or "availability" in title_lower:
            return DocTypeResult("compensation", 0.72, ["compensation"], "rules")

        if "capabilities" in title_lower or "what i can help with" in title_lower:
            return DocTypeResult("capabilities", 0.72, ["capabilities"], "rules")

        if "achievement" in title_lower or "impact" in title_lower:
            return DocTypeResult("achievements", 0.72, ["achievements"], "rules")

        if "career timeline" in title_lower or "timeline" in title_lower:
            return DocTypeResult("career_timeline", 0.72, ["career_timeline"], "rules")

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

        scores = {
            "cv": cls._score(text_lower, cls.CV_HINTS),
            "certificates": cls._score(text_lower, cls.CERT_HINTS),
            "projects": cls._score(text_lower, cls.PROJECT_HINTS),
            "recommendation": cls._score(text_lower, cls.RECOMMENDATION_HINTS),
            "experience_letter": cls._score(text_lower, cls.EXPERIENCE_LETTER_HINTS),
            "capabilities": cls._score(text_lower, cls.CAPABILITIES_HINTS),
            "preferences": cls._score(text_lower, cls.PREFERENCES_HINTS),
            "compensation": cls._score(text_lower, cls.COMPENSATION_HINTS),
            "faq": cls._score(text_lower, cls.FAQ_HINTS),
            "achievements": cls._score(text_lower, cls.ACHIEVEMENTS_HINTS),
            "career_timeline": cls._score(text_lower, cls.CAREER_TIMELINE_HINTS),
        }

        # Title boosts
        if "cv" in title_lower or "resume" in title_lower:
            scores["cv"] += 2
        if "certificate" in title_lower or "cert" in title_lower:
            scores["certificates"] += 2
        if "project" in title_lower:
            scores["projects"] += 2
        if "recommend" in title_lower or "reference" in title_lower:
            scores["recommendation"] += 2
        if "experience" in title_lower or "employment" in title_lower:
            scores["experience_letter"] += 2
        if "capabilities" in title_lower or "what i can help with" in title_lower:
            scores["capabilities"] += 2
        if "preference" in title_lower or "work style" in title_lower:
            scores["preferences"] += 2
        if "compensation" in title_lower or "availability" in title_lower:
            scores["compensation"] += 2
        if "faq" in title_lower:
            scores["faq"] += 2
        if "achievement" in title_lower or "impact" in title_lower:
            scores["achievements"] += 2
        if "career timeline" in title_lower or "timeline" in title_lower:
            scores["career_timeline"] += 2

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total = sum(scores.values()) or 1

        confidence = min(0.95, 0.45 + (best_score / max(3, total)))

        tags = [best_type]
        extra_tags = {
            "projects": ["portfolio", "solutions"],
            "cv": ["profile"],
            "recommendation": ["reference", "endorsement"],
            "certificates": ["credentials"],
            "experience_letter": ["employment", "verification", "hr"],
            "capabilities": ["services", "skills", "delivery"],
            "preferences": ["work_style", "favorites", "preferences"],
            "compensation": ["availability", "salary", "work_type"],
            "faq": ["questions", "answers"],
            "achievements": ["impact", "value", "delivery"],
            "career_timeline": ["career", "history", "progression"],
        }
        tags += extra_tags.get(best_type, [])

        return DocTypeResult(best_type, confidence, tags, "rules")

    @classmethod
    def _gemini_classify(cls, title: str, raw_text: str) -> Optional[DocTypeResult]:
        snippet = " ".join(raw_text.split())[:4000]

        allowed_types_text = ", ".join(sorted(cls.ALLOWED_TYPES))

        system_instruction = (
            "Classify the document type.\n"
            f"Allowed document_type values: {allowed_types_text}.\n"
            "Return JSON only."
        )

        prompt = (
            "Return JSON:\n"
            "{"
            "\"document_type\":\"one of the allowed values\","
            "\"confidence\":0.0-1.0,"
            "\"tags\":[\"...\"]"
            "}\n\n"
            f"Title: {title}\n\n"
            f"Content snippet:\n{snippet}\n"
        )

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

            if doc_type not in cls.ALLOWED_TYPES:
                doc_type = "other"

            tags = [str(t).strip() for t in tags if str(t).strip()][:10]
            if doc_type not in tags:
                tags.insert(0, doc_type)

            return DocTypeResult(
                doc_type,
                max(0.0, min(1.0, conf)),
                tags,
                "gemini",
            )
        except Exception:
            return None
