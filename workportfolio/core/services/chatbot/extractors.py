import re
from typing import List, Tuple
from core.models import DocumentChunk

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
LINKEDIN_RE = re.compile(r"(linkedin\.com/[A-Za-z0-9_\-/%]+)", re.IGNORECASE)


def _joined_text(chunks: List[DocumentChunk]) -> str:
    return "\n".join([(c.content or "") for c in chunks])


def try_extract_contact(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Applies only if the question is clearly about contacting / reaching.
    If not, returns (False, "", 0.0) and the bot continues normally.
    """
    q = (question or "").lower()
    keywords = ["contact", "reach", "email", "phone",
                "call", "whatsapp", "linkedin", "get in touch"]
    if not any(k in q for k in keywords):
        return False, "", 0.0

    text = _joined_text(chunks)
    emails = sorted(set(EMAIL_RE.findall(text)))
    phones = sorted(set([p.strip() for p in PHONE_RE.findall(text)]))
    linkedins = sorted(set(LINKEDIN_RE.findall(text)))

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
    # boost confidence if we actually found concrete fields
    boost = 0.15 if (emails or phones) else 0.0
    return True, answer, boost


def try_extract_skills(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Applies only if the question is clearly about skills/tech stack/tools.
    Otherwise does nothing.
    """
    q = (question or "").lower()
    keywords = ["skills", "technical", "tech stack",
                "tools", "frameworks", "technologies"]
    if not any(k in q for k in keywords):
        return False, "", 0.0

    text = _joined_text(chunks).replace("\r", "\n")
    lower = text.lower()
    idx = lower.find("skills")
    if idx == -1:
        return True, "I found skills-related evidence, but couldn’t isolate a clean SKILLS section yet.", 0.0

    section = text[idx: idx + 2500]
    lines = [ln.strip("•- \t") for ln in section.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) < 60]
    lines = [ln for ln in lines if ln.lower(
    ) not in {"skills", "about me", "education", "experience"}]

    seen = set()
    skills = []
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            seen.add(key)
            skills.append(ln)

    skills = skills[:25]
    if not skills:
        return True, "I couldn’t extract a clean skills list from the uploaded documents yet.", 0.0

    answer = "Samah’s technical skills (from the uploaded documents):\n- " + \
        "\n- ".join(skills)
    return True, answer, 0.10
