import re
from typing import List, Tuple
from core.models import DocumentChunk

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
LINKEDIN_RE = re.compile(
    r"(https?://(?:www\.)?linkedin\.com/[A-Za-z0-9_\-/%]+|linkedin\.com/[A-Za-z0-9_\-/%]+)",
    re.IGNORECASE,
)

KNOWN_SKILLS = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Django",
    "Django REST Framework",
    "FastAPI",
    "Flask",
    "React",
    "Next.js",
    "Tailwind CSS",
    "Bootstrap",
    "PostgreSQL",
    "MongoDB",
    "MySQL",
    "SQL Server",
    "Oracle",
    "Docker",
    "NGINX",
    "Gunicorn",
    "JWT",
    "RBAC",
    "Swagger",
    "OpenAPI",
    "Hugging Face",
    "BERT",
    "RoBERTa",
    "T5",
    "LangChain",
    "Langfuse",
    "Ollama",
    "Gemini",
    "Tesseract",
    "EasyOCR",
    "OpenCV",
    "RAG",
    "Vector Search",
]


def _joined_text(chunks: List[DocumentChunk]) -> str:
    """
    Join all retrieved chunk content into one text blob for lightweight extraction.
    """
    return "\n".join([(c.content or "") for c in chunks])


def _normalize_linkedin(url: str) -> str:
    """
    Normalize LinkedIn URLs so they always return with a scheme.
    """
    url = (url or "").strip()
    if not url:
        return url
    if not url.lower().startswith(("http://", "https://")):
        return f"https://{url}"
    return url


def try_extract_contact(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Applies only if the question is clearly about contacting / reaching Samah.
    If not relevant, returns (False, "", 0.0) so the chatbot continues normally.
    """
    q = (question or "").lower()
    keywords = ["contact", "reach", "email", "phone",
                "call", "whatsapp", "linkedin", "get in touch"]

    if not any(k in q for k in keywords):
        return False, "", 0.0

    text = _joined_text(chunks)

    emails = sorted(set(EMAIL_RE.findall(text)))
    phones = sorted(set([p.strip() for p in PHONE_RE.findall(text)]))
    linkedins = sorted(set(_normalize_linkedin(x)
    for x in LINKEDIN_RE.findall(text)))

    parts = []
    if phones:
        parts.append(f"Phone: {phones[0]}")
    if emails:
        parts.append(f"Email: {emails[0]}")
    if linkedins:
        parts.append(f"LinkedIn: {linkedins[0]}")

    if not parts:
        return True, "I couldn’t find contact details in the uploaded documents.", 0.0

    answer = "You can contact Samah using the details found in the uploaded documents:\n- " + \
        "\n- ".join(parts)

    # Slight confidence boost if concrete contact info was found
    boost = 0.15 if (emails or phones or linkedins) else 0.0
    return True, answer, boost


def try_extract_skills(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Applies only if the question is clearly about skills / technologies / tools.
    Tries section-based extraction first, then falls back to known skill matching.
    """
    q = (question or "").lower()
    keywords = [
        "skills",
        "technical",
        "tech stack",
        "tools",
        "frameworks",
        "technologies",
        "stack",
        "what does she know",
        "what can she use",
    ]

    if not any(k in q for k in keywords):
        return False, "", 0.0

    text = _joined_text(chunks).replace("\r", "\n")
    lower = text.lower()

    # Step 1: Try extracting from known headings
    heading_candidates = ["skills", "technical skills",
    "technologies", "tech stack", "tools"]
    section = ""

    for heading in heading_candidates:
        idx = lower.find(heading)
        if idx != -1:
            section = text[idx: idx + 3000]
            break

    extracted = []

    if section:
        lines = [ln.strip("•- \t") for ln in section.splitlines()]
        lines = [ln for ln in lines if ln and len(ln) < 80]
        lines = [
            ln for ln in lines
            if ln.lower() not in {
                "skills",
                "technical skills",
                "technologies",
                "tech stack",
                "tools",
                "about me",
                "education",
                "experience",
            }
        ]
        extracted.extend(lines)

    # Step 2: Fallback scan across all text for known technologies
    found_known = []
    for skill in KNOWN_SKILLS:
        if skill.lower() in lower:
            found_known.append(skill)

    # Step 3: Deduplicate while preserving order
    combined = []
    seen = set()

    for item in extracted + found_known:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            combined.append(item.strip())

    combined = combined[:25]

    if not combined:
        return True, "I couldn’t extract a clean skills list from the uploaded documents yet.", 0.0

    answer = "Samah’s technical skills (from the uploaded documents):\n- " + \
        "\n- ".join(combined)
    return True, answer, 0.10


def try_extract_preferences(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Applies only if the question is about preferences such as favorite language,
    preferred backend framework, frontend stack, or work style.
    """
    q = (question or "").lower()

    pref_keywords = [
        "favorite language",
        "favourite language",
        "preferred language",
        "favorite programming language",
        "favourite programming language",
        "prefer django",
        "django or fastapi",
        "preferred backend",
        "preferred frontend",
        "work style",
        "working style",
        "backend or frontend",
        "favorite framework",
        "favourite framework",
    ]

    if not any(k in q for k in pref_keywords):
        return False, "", 0.0

    text = _joined_text(chunks)
    lower = text.lower()

    findings = []

    if (
        "favorite programming language is python" in lower
        or "favourite programming language is python" in lower
        or "favorite programming language is **python**" in lower
    ):
        findings.append("Samah’s favorite programming language is Python.")

    if (
        "preferred backend framework is django" in lower
        or "preferred backend framework is **django**" in lower
    ):
        findings.append("Samah’s preferred backend framework is Django.")

    if (
        "preferred frontend stack is next.js with react and tailwind css" in lower
        or "preferred frontend stack is **next.js with react and tailwind css**" in lower
    ):
        findings.append(
            "Samah’s preferred frontend stack is Next.js with React and Tailwind CSS.")

    if "strongest in backend development and ai-focused work" in lower:
        findings.append(
            "Samah is strongest in backend development and AI-focused work.")

    if not findings:
        return False, "", 0.0

    answer = "\n".join(findings[:4])
    return True, answer, 0.15


def try_extract_availability(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Applies only if the question is about compensation, availability,
    work arrangement, freelance, contract, remote, or related topics.
    """
    q = (question or "").lower()

    keywords = [
        "salary",
        "payment",
        "compensation",
        "rate",
        "availability",
        "available",
        "remote",
        "hybrid",
        "on-site",
        "onsite",
        "freelance",
        "contract",
        "full-time",
        "open to work",
        "notice period",
    ]

    if not any(k in q for k in keywords):
        return False, "", 0.0

    text = _joined_text(chunks)
    lower = text.lower()
    findings = []

    if "open to full-time" in lower:
        findings.append("Samah is open to full-time opportunities.")

    if "open to freelance" in lower or "open to freelance or project-based work" in lower:
        findings.append("Samah is open to freelance and project-based work.")

    if "remote work" in lower or "hybrid work" in lower or "on-site work" in lower or "on-site" in lower:
        findings.append(
            "Samah is open to remote, hybrid, and suitable on-site opportunities.")

    if "dubai" in lower or "abu dhabi" in lower or "uae" in lower:
        findings.append(
            "Samah is especially open to opportunities in Dubai, Abu Dhabi, the UAE, and strong remote opportunities.")

    if "compensation" in q or "salary" in q or "payment" in q or "rate" in q:
        if (
            "fair and competitive compensation package" in lower
            or "compensation expectations should be discussed" in lower
            or "target compensation range should be discussed" in lower
        ):
            findings.append(
                "Compensation is intended to be discussed based on role scope, technical depth, leadership responsibility, and work arrangement."
            )

    if not findings:
        return False, "", 0.0

    answer = "\n".join(findings[:4])
    return True, answer, 0.10
