import re
from typing import List, Tuple
from core.models import DocumentChunk

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Broad candidate regex first, then we validate with helper functions.
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


def _is_date_like(value: str) -> bool:
    """
    Reject common date formats that can be falsely matched as phone numbers.
    """
    v = (value or "").strip()

    date_patterns = [
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",   # 12-08-2025 / 12/08/2025
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",     # 2025-08-12 / 2025/08/12
        r"^\d{1,2}\.\d{1,2}\.\d{2,4}$",       # 12.08.2025
    ]

    return any(re.match(p, v) for p in date_patterns)


def _normalize_phone(value: str) -> str:
    """
    Normalize phone candidate by trimming spaces and collapsing internal whitespace.
    """
    value = (value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _is_valid_phone(value: str) -> bool:
    """
    Keep only strings that look like real phone numbers.
    Reject dates and weak numeric strings.
    """
    v = _normalize_phone(value)
    if not v:
        return False

    if _is_date_like(v):
        return False

    digits_only = re.sub(r"\D", "", v)

    # Too short or too long to be a realistic phone number
    if len(digits_only) < 8 or len(digits_only) > 15:
        return False

    # Reject obvious year-like / date-like values
    if re.fullmatch(r"\d{8}", digits_only):
        # many 8-digit values are not phones in this context; keep conservative
        return False

    # Reject values that have only two separators and look like dd-mm-yyyy
    if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", v):
        return False

    return True


def try_extract_contact(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Applies only if the question is clearly about contacting / reaching Samah.
    Returns a more precise answer based on what the user asked for:
    - email question -> email only
    - phone question -> phone only
    - LinkedIn question -> LinkedIn only if present
    - general contact question -> all available contact details
    """
    q = (question or "").lower()

    contact_keywords = [
        "contact",
        "reach",
        "email",
        "phone",
        "call",
        "whatsapp",
        "linkedin",
        "get in touch",
        "connect",
        "contact details",
        "how can i contact",
        "how do i contact",
    ]

    if not any(k in q for k in contact_keywords):
        return False, "", 0.0

    text = _joined_text(chunks)

    emails = sorted(set(EMAIL_RE.findall(text)))

    raw_phones = PHONE_RE.findall(text)
    phones = sorted(
        set(
            _normalize_phone(p)
            for p in raw_phones
            if _is_valid_phone(p)
        )
    )

    linkedins = sorted(
        set(_normalize_linkedin(x) for x in LINKEDIN_RE.findall(text))
    )

    wants_email = "email" in q
    wants_phone = any(k in q for k in ["phone", "call", "whatsapp", "number"])
    wants_linkedin = "linkedin" in q
    wants_general_contact = any(
        k in q for k in ["contact", "reach", "get in touch", "connect", "contact details"]
    ) and not (wants_email or wants_phone or wants_linkedin)

    # Specific request: email only
    if wants_email:
        if emails:
            return True, f"Samah’s email is {emails[0]}.", 0.15
        return True, "I couldn’t find an email address in the uploaded documents.", 0.0

    # Specific request: phone only
    if wants_phone:
        if phones:
            return True, f"Samah’s phone number is {phones[0]}.", 0.15
        return True, "I couldn’t find a phone number in the uploaded documents.", 0.0

    # Specific request: LinkedIn only
    if wants_linkedin:
        if linkedins:
            return True, f"Samah’s LinkedIn is {linkedins[0]}.", 0.15
        return True, "I couldn’t find a LinkedIn link in the uploaded documents.", 0.0

    # General contact request: return all available details
    parts = []
    if emails:
        parts.append(f"Email: {emails[0]}")
    if linkedins:
        parts.append(f"LinkedIn: {linkedins[0]}")
    if phones:
        parts.append(f"Phone: {phones[0]}")

    if wants_general_contact or parts:
        if not parts:
            return True, "I couldn’t find contact details in the uploaded documents.", 0.0

        answer = "You can contact Samah using the details found in the uploaded documents:\n- " + \
            "\n- ".join(parts)
        return True, answer, 0.15

    return False, "", 0.0


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

    found_known = []
    for skill in KNOWN_SKILLS:
        if skill.lower() in lower:
            found_known.append(skill)

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
    Returns a narrower answer based on the actual question.
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

    if "favorite programming language" in q or "favourite programming language" in q:
        if (
            "favorite programming language is python" in lower
            or "favourite programming language is python" in lower
            or "favorite programming language is **python**" in lower
        ):
            return True, "Samah’s favorite programming language is Python.", 0.15

    if "django or fastapi" in q or "prefer django" in q:
        findings = []
        if (
            "preferred backend framework is django" in lower
            or "preferred backend framework is **django**" in lower
        ):
            findings.append("Samah’s preferred backend framework is Django.")
        if "fastapi" in lower:
            findings.append(
                "Samah also has experience with FastAPI, but Django is her stronger and more preferred framework.")
        if findings:
            return True, "\n".join(findings[:2]), 0.15

    if "preferred frontend" in q:
        if (
            "preferred frontend stack is next.js with react and tailwind css" in lower
            or "preferred frontend stack is **next.js with react and tailwind css**" in lower
        ):
            return True, "Samah’s preferred frontend stack is Next.js with React and Tailwind CSS.", 0.15

    if "backend or frontend" in q or "work style" in q or "working style" in q:
        findings = []
        if "strongest in backend development and ai-focused work" in lower:
            findings.append(
                "Samah is strongest in backend development and AI-focused work.")
        if findings:
            return True, "\n".join(findings[:2]), 0.10

    return False, "", 0.0


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
