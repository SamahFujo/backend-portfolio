from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class QuestionContract:
    name: str
    preferred_document_types: tuple[str, ...]
    strict_retry_document_types: tuple[str, ...]
    required_terms: tuple[str, ...] = ()
    section_terms: tuple[str, ...] = ()
    forbidden_document_types: tuple[str, ...] = ()
    min_chunk_score: float = 0.55
    retrieval_suffix: str = ""


QUESTION_CONTRACTS: Dict[str, QuestionContract] = {
    "education": QuestionContract(
        name="education",
        preferred_document_types=("cv",),
        strict_retry_document_types=("cv",),
        required_terms=(
            "education",
            "degree",
            "bachelor",
            "master",
            "b.sc",
            "bsc",
            "university",
            "college",
            "gpa",
            "field of study",
            "major",
        ),
        section_terms=("education", "academic"),
        forbidden_document_types=("faq", "experience_letter"),
        min_chunk_score=0.62,
        retrieval_suffix="education degree university college GPA field of study major academic background CV",
    ),
    "contact": QuestionContract(
        name="contact",
        preferred_document_types=("cv",),
        strict_retry_document_types=("cv",),
        required_terms=(
            "email",
            "phone",
            "linkedin",
            "contact",
            "whatsapp",
            "@",
            "+",
        ),
        section_terms=("contact", "resume header"),
        min_chunk_score=0.58,
        retrieval_suffix="contact email phone linkedin whatsapp CV",
    ),
    "compensation": QuestionContract(
        name="compensation",
        preferred_document_types=("compensation",),
        strict_retry_document_types=("compensation",),
        required_terms=(
            "salary",
            "compensation",
            "rate",
            "hourly",
            "monthly",
            "availability",
            "remote",
            "freelance",
            "contract",
        ),
        forbidden_document_types=("faq",),
        min_chunk_score=0.60,
        retrieval_suffix="salary compensation hourly rate availability freelance contract work arrangement",
    ),
    "certificates": QuestionContract(
        name="certificates",
        preferred_document_types=("certificates", "cv"),
        strict_retry_document_types=("certificates", "cv"),
        required_terms=(
            "certificate",
            "certificates",
            "certification",
            "training",
            "course",
            "credential",
        ),
        section_terms=("certificates", "certifications", "training"),
        forbidden_document_types=("experience_letter",),
        retrieval_suffix="certificates certifications courses training credentials CV",
    ),
    "recommendation": QuestionContract(
        name="recommendation",
        preferred_document_types=("recommendation",),
        strict_retry_document_types=("recommendation",),
        required_terms=("recommendation", "reference", "professionalism", "endorsement"),
        section_terms=("recommendation", "reference"),
        min_chunk_score=0.60,
        retrieval_suffix="recommendation reference endorsement professionalism letter",
    ),
    "experience_letter": QuestionContract(
        name="experience_letter",
        preferred_document_types=("experience_letter", "cv"),
        strict_retry_document_types=("experience_letter", "cv"),
        required_terms=("experience letter", "employment", "worked", "role", "date"),
        section_terms=("experience", "employment"),
        retrieval_suffix="experience letter employment verification role dates CV",
    ),
    "projects": QuestionContract(
        name="projects",
        preferred_document_types=("projects", "capabilities", "cv"),
        strict_retry_document_types=("projects", "capabilities"),
        required_terms=("project", "projects", "built", "developed", "implemented", "solution"),
        section_terms=("projects", "portfolio"),
        retrieval_suffix="projects built developed implemented solutions portfolio",
    ),
}

QUESTION_MARKERS: Dict[str, tuple[str, ...]] = {
    "education": (
        "education",
        "degree",
        "degrees",
        "university",
        "college",
        "gpa",
        "field of study",
        "major",
        "studied",
    ),
    "contact": (
        "contact",
        "email",
        "phone",
        "linkedin",
        "whatsapp",
        "reach",
        "connect",
        "get in touch",
    ),
    "compensation": (
        "salary",
        "compensation",
        "rate",
        "hourly",
        "monthly",
        "availability",
        "freelance",
        "contract",
        "remote",
        "notice period",
    ),
    "certificates": (
        "certificate",
        "certificates",
        "certification",
        "certifications",
        "course",
        "training",
        "credential",
    ),
    "recommendation": ("recommendation", "reference letter", "endorsement", "referee"),
    "experience_letter": ("experience letter", "employment letter", "employment verification", "service letter"),
    "projects": ("project", "projects", "built", "developed", "implemented", "dashboard", "chatbot"),
}

ANSWER_TYPE_TO_CONTRACT = {
    "education": "education",
    "contact": "contact",
    "compensation": "compensation",
    "availability": "compensation",
    "certificates": "certificates",
    "recommendation": "recommendation",
    "experience_letter": "experience_letter",
    "projects": "projects",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "have",
    "her",
    "hers",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "samah",
    "she",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "what",
    "which",
    "who",
    "with",
}


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    low_text = text.lower()
    low_term = term.lower().strip()
    if " " in low_term or "." in low_term or "+" in low_term:
        return low_term in low_text
    return bool(re.search(rf"\b{re.escape(low_term)}\b", low_text))


def _tokenize(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9+.]+", (value or "").lower())
        if len(token) >= 3 and token not in STOPWORDS
    ]


def infer_question_contract(
    question: str,
    query_plan: Optional[Dict[str, Any]] = None,
    question_route: Optional[str] = None,
) -> Optional[QuestionContract]:
    query_plan = query_plan or {}
    answer_type = (query_plan.get("answer_type") or "").strip().lower()

    if answer_type in ANSWER_TYPE_TO_CONTRACT:
        return QUESTION_CONTRACTS[ANSWER_TYPE_TO_CONTRACT[answer_type]]

    low_question = (question or "").strip().lower()
    if not low_question:
        return None

    for contract_name, markers in QUESTION_MARKERS.items():
        if any(marker in low_question for marker in markers):
            return QUESTION_CONTRACTS[contract_name]

    if question_route == "profile_docs_question":
        if any(marker in low_question for marker in ("resume", "cv")):
            return QUESTION_CONTRACTS["contact"]

    return None


def evaluate_evidence(
    chunks: Iterable[Any],
    contract: Optional[QuestionContract],
    question: str = "",
    top_k: int = 4,
) -> Dict[str, Any]:
    if contract is None:
        return {
            "contract": None,
            "is_sufficient": True,
            "top_score": 1.0,
            "matched_chunk_ids": [],
            "chunk_scores": [],
            "reason": "no_contract",
        }

    question_tokens = set(_tokenize(question))
    chunk_scores: List[Dict[str, Any]] = []

    for index, chunk in enumerate(list(chunks)[:top_k]):
        document = getattr(chunk, "document", None)
        doc_type = getattr(document, "document_type", None)
        doc_title = getattr(document, "title", "")
        section_title = getattr(chunk, "section_title", "") or ""
        content = getattr(chunk, "content", "") or ""
        combined_text = f"{doc_title}\n{section_title}\n{content}".lower()

        score = 0.0
        reasons: List[str] = []

        if doc_type in contract.preferred_document_types:
            score += 0.42
            reasons.append("preferred_document_type")

        required_hits = [
            term for term in contract.required_terms if _contains_term(combined_text, term)
        ]
        if required_hits:
            score += min(0.35, 0.12 + (0.08 * len(required_hits)))
            reasons.append("required_terms")

        section_hits = [
            term for term in contract.section_terms if _contains_term(section_title, term)
        ]
        if section_hits:
            score += 0.18
            reasons.append("section_match")

        overlap_count = len(question_tokens.intersection(set(_tokenize(combined_text))))
        if overlap_count >= 2:
            score += 0.12
            reasons.append("question_overlap")
        elif overlap_count == 1:
            score += 0.06
            reasons.append("light_overlap")

        if doc_type in contract.forbidden_document_types:
            score -= 0.25
            reasons.append("forbidden_document_type")

        chunk_scores.append(
            {
                "chunk_id": str(getattr(chunk, "id", "")),
                "doc_type": doc_type,
                "doc_title": doc_title,
                "chunk_index": getattr(chunk, "chunk_index", None),
                "score": round(score, 4),
                "reasons": reasons,
            }
        )

    top_score = max((row["score"] for row in chunk_scores), default=0.0)
    matched_chunk_ids = [
        row["chunk_id"] for row in chunk_scores if row["score"] >= contract.min_chunk_score
    ]
    preferred_hits = [
        row for row in chunk_scores if row["doc_type"] in contract.preferred_document_types
    ]

    is_sufficient = bool(matched_chunk_ids)
    if not is_sufficient and top_score >= (contract.min_chunk_score - 0.08) and preferred_hits:
        is_sufficient = True

    return {
        "contract": contract.name,
        "is_sufficient": is_sufficient,
        "top_score": round(top_score, 4),
        "matched_chunk_ids": matched_chunk_ids,
        "chunk_scores": chunk_scores,
        "reason": "validated" if is_sufficient else "weak_or_mismatched_evidence",
    }


def build_retry_query(retrieval_query: str, contract: Optional[QuestionContract]) -> str:
    query = (retrieval_query or "").strip()
    if contract is None or not contract.retrieval_suffix:
        return query

    suffix = contract.retrieval_suffix.strip()
    low_query = query.lower()
    if suffix.lower() in low_query:
        return query

    return f"{query} {suffix}".strip()
