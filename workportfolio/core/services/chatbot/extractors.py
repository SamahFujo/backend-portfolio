import re
from typing import List, Tuple, Optional
from core.models import DocumentChunk
from datetime import datetime

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
    "LangChain",
]

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_text_date(text: str):
    """Try to parse dates in common formats like '03-Oct-2021', '03/Oct/2021', '03 Oct 2021'."""
    if not text:
        return None

    text = text.strip().lower()

    # Match: 03-Oct-2021 or 03/Oct/2021 or 03 Oct 2021
    m = re.match(r"(\d{1,2})[-/\s]+([a-zA-Z]+)[-/\s]+(\d{4})", text)
    if not m:
        return None

    day = int(m.group(1))
    month_name = m.group(2).lower()
    year = int(m.group(3))

    month = MONTH_MAP.get(month_name)
    if not month:
        return None

    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def _format_duration(start_dt: datetime, end_dt: datetime) -> str:
    """Format the duration between two datetime objects.

    Args:
        start_dt (datetime): The start datetime.
        end_dt (datetime): The end datetime.

    Returns:
        str: The formatted duration string.
    """
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    months = (end_dt.year - start_dt.year) * \
        12 + (end_dt.month - start_dt.month)

    if end_dt.day < start_dt.day:
        months -= 1

    years = months // 12
    rem_months = months % 12

    if years > 0 and rem_months > 0:
        return f"{years} years and {rem_months} months"
    if years > 0:
        return f"{years} years"
    return f"{rem_months} months"


def try_extract_experience_duration(question: str, chunks):
    """
    Extract experience duration for:
    1) general total experience questions
    2) employer-specific duration questions

    General questions should use all documented roles from the CV when possible.
    Specific employer questions should only use matching roles/docs.
    """
    q = (question or "").strip().lower()

    direct_markers = [
        "how many years of experience",
        "how much experience",
        "how long has she worked",
        "what is her total experience",
        "how many years has she worked",
        "years of experience",
        "total experience",
        "how long did she work",
        "how long did samah work",
    ]

    has_direct_marker = any(marker in q for marker in direct_markers)

    has_years = "year" in q or "years" in q
    has_experience_like = any(
        word in q for word in ["experience", "expirience", "experiance", "exp"]
    )
    has_work_like = any(
        phrase in q for phrase in ["she have", "she has", "she worked", "she work", "worked"]
    )

    if not (has_direct_marker or (has_years and has_experience_like) or (has_experience_like and has_work_like)):
        return False, "", 0.0

    target_employer = _extract_target_employer(question)

    # ------------------------------------------------------------
    # 1) Try CV-based role ranges first
    # ------------------------------------------------------------
    cv_ranges = _extract_cv_experience_ranges(chunks)

    if cv_ranges:
        if target_employer:
            filtered = []
            for item in cv_ranges:
                org_low = item["organization"].lower()

                if target_employer == "nasser" and any(x in org_low for x in ["nasser", "nairdc"]):
                    filtered.append(item)
                elif target_employer == "asu" and "applied science university" in org_low:
                    filtered.append(item)
                elif target_employer == "mohammed bin khalifa cardiac centre" and (
                    "mohammed bin khalifa" in org_low or "cardiac centre" in org_low or "cardiac center" in org_low
                ):
                    filtered.append(item)

            cv_ranges = filtered

        if cv_ranges:
            starts = [item["start"] for item in cv_ranges]
            ends = [item["end"] for item in cv_ranges]

            start_dt = min(starts)
            end_dt = max(ends)
            duration = _format_duration(start_dt, end_dt)

            organizations = []
            for item in cv_ranges:
                org = item["organization"].strip()
                if org not in organizations:
                    organizations.append(org)

            ORG_DISPLAY_MAP = {
                "department of nasser centre for science and technology": "Nasser Artificial Intelligence Research & Development Centre (NAIRDC)",
                "nasser artificial intelligence research & development centre (nairdc)": "Nasser Artificial Intelligence Research & Development Centre (NAIRDC)",
                "applied science university (asu)": "Applied Science University",
                "applied science university": "Applied Science University",
                "mohammed bin khalifa cardiac centre": "Mohammed Bin Khalifa Cardiac Centre",
            }

            clean_orgs = []
            seen = set()

            for org in organizations:
                norm = " ".join(org.lower().split())
                display = ORG_DISPLAY_MAP.get(norm, org.strip())

                if display not in seen:
                    seen.add(display)
                    clean_orgs.append(display)

            if target_employer:
                display_org = clean_orgs[0] if clean_orgs else organizations[0]
                answer = (
                    f"Based on the available documents, Samah appears to have about {duration} "
                    f"of documented experience with {display_org}, spanning from "
                    f"{start_dt.strftime('%b %Y')} to {end_dt.strftime('%b %Y')}."
                )
            else:
                answer = (
                    f"Based on the available CV, Samah appears to have about {duration} of documented professional experience, "
                    f"spanning from {start_dt.strftime('%b %Y')} to {end_dt.strftime('%b %Y')}."
                )

                if clean_orgs:
                    answer += " This includes roles across: " + \
                        "; ".join(clean_orgs[:5]) + "."

                # Add current status note
                answer += (
                    " Although she is not currently in a full-time role, she continues to improve her skills "
                    "through self-initiated learning and by building projects."
                )

            return True, answer, 0.38

    # ------------------------------------------------------------
    # 2) Fallback to explicit date ranges from other docs
    # ------------------------------------------------------------
    joined = "\n".join((c.content or "") for c in chunks)

    explicit_duration = re.search(
        r"(\d+)\s+years?(?:\s+and\s+(\d+)\s+months?)?",
        joined,
        flags=re.IGNORECASE,
    )
    if explicit_duration:
        years = explicit_duration.group(1)
        months = explicit_duration.group(2)

        if months:
            answer = f"Based on the available documents, Samah has about {years} years and {months} months of experience."
        else:
            answer = f"Based on the available documents, Samah has about {years} years of experience."

        return True, answer, 0.35

    date_matches = re.findall(
        r"\b\d{1,2}[-/\s]+[A-Za-z]+[-/\s]+\d{4}\b",
        joined,
        flags=re.IGNORECASE,
    )

    parsed_dates = []
    for raw in date_matches:
        dt = _parse_text_date(raw)
        if dt:
            parsed_dates.append(dt)

    if len(parsed_dates) >= 2:
        parsed_dates.sort()
        start_dt = parsed_dates[0]
        end_dt = parsed_dates[-1]
        duration = _format_duration(start_dt, end_dt)

        answer = (
            f"Based on the available documents, Samah appears to have about {duration} of experience. "
            f"The clearest documented range found is from {start_dt.strftime('%d-%b-%Y')} "
            f"to {end_dt.strftime('%d-%b-%Y')}."
        )
        return True, answer, 0.30

    return False, "", 0.0


def _extract_cv_experience_ranges(chunks):
    text = "\n".join((c.content or "") for c in chunks if getattr(
        c.document, "document_type", None) == "cv")

    patterns = [
        # Examples:
        # Applied Science University / Bahrain/Jan 2020 - Dec 2021
        # Nasser Centre ... / Bahrain/Jan 2022 - Aug 2025
        r"([A-Za-z][A-Za-z &()/\-]+?)\s*/\s*Bahrain\s*/\s*([A-Za-z]{3,9})\s+(\d{4})\s*-\s*([A-Za-z]{3,9})\s+(\d{4})",
    ]

    results = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            org = m.group(1).strip()
            start_month = m.group(2).strip()
            start_year = int(m.group(3))
            end_month = m.group(4).strip()
            end_year = int(m.group(5))

            start_dt = _parse_month_year(start_month, start_year)
            end_dt = _parse_month_year(end_month, end_year, end_of_month=True)

            if start_dt and end_dt:
                results.append({
                    "organization": org,
                    "start": start_dt,
                    "end": end_dt,
                })

    return results


def _parse_month_year(month_name: str, year: int, end_of_month: bool = False):
    month = MONTH_MAP.get((month_name or "").strip().lower())
    if not month:
        return None

    if end_of_month:
        # simple safe end-of-month handling
        if month == 12:
            return datetime(year, 12, 31)
        next_month = datetime(year, month, 1).replace(day=28)
        while True:
            try:
                next_month = next_month.replace(day=next_month.day + 1)
            except ValueError:
                break
        return next_month

    return datetime(year, month, 1)


def _extract_target_employer(question: str) -> str | None:
    q = (question or "").lower()

    employer_aliases = {
        "nasser": ["nasser", "nairdc", "nasser centre", "nasser center"],
        "asu": ["applied science university", "asu"],
        "mohammed bin khalifa cardiac centre": [
            "mohammed bin khalifa cardiac centre",
            "mohammed bin khalifa cardiac center",
            "cardiac centre",
            "cardiac center",
        ],
    }

    for canonical, aliases in employer_aliases.items():
        if any(alias in q for alias in aliases):
            return canonical

    return None


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
        return True, "I couldn’t find a phone number in the available information.", 0.0

    # Specific request: LinkedIn only
    if wants_linkedin:
        if linkedins:
            return True, f"Samah’s LinkedIn is {linkedins[0]}.", 0.15
        return True, "I couldn’t find a LinkedIn link in the available information, but you can likely find it in the footer of this webpage", 0.0

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

        answer = "You can contact Samah using the contact details:\n- " + \
            "\n- ".join(parts)
        return True, answer, 0.15

    return False, "", 0.0


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


def _normalize_skill_line(line: str) -> str:
    """
    Clean bullet prefixes / numbering / extra spaces.
    """
    line = (line or "").strip()
    line = re.sub(r"^[\-\u2022•*\d.)\s]+", "", line).strip()
    line = re.sub(r"\s+", " ", line)
    return line


def _looks_like_noise_line(line: str) -> bool:
    """
    Reject FAQ-style questions, labels, broken fragments, and metadata lines.
    """
    if not line:
        return True

    lower = line.lower().strip()

    blocked_exact = {
        "skills",
        "technical skills",
        "technologies",
        "tech stack",
        "tools",
        "summary",
        "about me",
        "education",
        "experience",
        "projects",
        "types of projects i can build",
        "programming languages",
        "backend frameworks and tools",
        "frontend technologies",
        "database technologies",
        "devops and deployment",
        "ai and llm technologies",
        "document and ocr technologies",
    }

    blocked_starts = (
        "what ",
        "does ",
        "can ",
        "is ",
        "are ",
        "has ",
        "why ",
        "how ",
        "when ",
        "where ",
        "who ",
        "prepared on",
        "status",
        "technology stack",
        "key points",
    )

    if lower in blocked_exact:
        return True

    if lower.startswith(blocked_starts):
        return True

    if line.endswith("?"):
        return True

    # Reject lines that look like prose / metadata rather than skills
    if ":" in line and lower not in {
        "django rest framework",
        "sql server",
    }:
        return True

    # Reject obvious broken fragments
    if len(line) < 2:
        return True

    if len(line) > 60:
        return True

    # Reject lines with too many words unless they are known valid multi-word skills
    words = line.split()
    if len(words) > 5:
        return True

    # Reject lines that look like sentence fragments
    if lower.endswith("."):
        return True

    return False


def try_extract_capability_with_tool(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Handle capability questions that mention a specific tool.
    Example:
    - Can Samah help in a dashboard with Power BI tool?
    - Can she work with Tableau?
    """
    q = (question or "").strip().lower()

    capability_markers = [
        "can samah help",
        "can she help",
        "can samah build",
        "can she build",
        "can samah work",
        "can she work",
        "help with",
    ]

    if not any(marker in q for marker in capability_markers):
        return False, "", 0.0

    known_tools = [
        "power bi",
        "tableau",
        "qlik",
        "looker",
        "metabase",
        "superset",
    ]

    requested_tool = None
    for tool in known_tools:
        if tool in q:
            requested_tool = tool
            break

    if not requested_tool:
        return False, "", 0.0

    text = "\n".join((c.content or "") for c in chunks).lower()

    dashboard_signals = [
        "dashboard",
        "dashboards",
        "analytics",
        "visualization",
        "interactive dashboards",
        "decision support",
        "monitoring",
        "reporting",
    ]

    has_dashboard_capability = any(
        signal in text for signal in dashboard_signals)
    has_tool_evidence = requested_tool in text

    if has_dashboard_capability and has_tool_evidence:
        answer = (
            f"Yes, Samah can help with dashboard-related work, and the uploaded documents "
            f"also contain evidence related to {requested_tool.title()}."
        )
        return True, answer, 0.80

    if has_dashboard_capability and not has_tool_evidence:
        answer = (
            f"Samah can help with dashboard and analytics solutions, but I do not see "
            f"explicit evidence in the uploaded documents that she specifically works with {requested_tool.title()}."
        )
        return True, answer, 0.75

    if not has_dashboard_capability and has_tool_evidence:
        answer = (
            f"I found mention of {requested_tool.title()}, but I do not have enough evidence "
            f"to confirm dashboard-related capability from the uploaded documents."
        )
        return True, answer, 0.40

    return True, (
        f"I do not have enough evidence in the uploaded documents to confirm that Samah "
        f"works with {requested_tool.title()} specifically."
    ), 0.0


def _extract_requested_technology(question: str) -> str | None:
    """
    Extract the requested technology from the question.

    Args:
        question (str): The user's question.

    Returns:
        str | None: _description_
    """
    q = (question or "").lower()

    tech_aliases = {
        "django": ["django", "django rest framework", "drf"],
        "react": ["react", "react.js", "reactjs"],
        "next.js": ["next.js", "nextjs", "next js"],
        "python": ["python"],
        "oracle": ["oracle", "oracle db", "oracle database"],
        "bert": ["bert"],
        "roberta": ["roberta"],
        "langchain": ["langchain"],
        "gemini": ["gemini"],
        "ollama": ["ollama"],
    }

    for canonical, aliases in tech_aliases.items():
        if any(alias in q for alias in aliases):
            return canonical

    return None


def try_extract_projects_by_technology(question: str, chunks):
    q = (question or "").strip().lower()

    project_markers = [
        "which projects used",
        "what projects used",
        "which project used",
        "what project used",
        "projects with",
        "projects using",
        "project using",
    ]

    TECH_DISPLAY_NAMES = {
        "django": "Django",
        "react": "React",
        "next.js": "Next.js",
        "python": "Python",
        "oracle": "Oracle",
        "bert": "BERT",
        "roberta": "RoBERTa",
        "langchain": "LangChain",
        "gemini": "Gemini",
        "ollama": "Ollama",
    }

    if not any(marker in q for marker in project_markers):
        return False, "", 0.0

    requested_tech = _extract_requested_technology(question)
    if not requested_tech:
        return False, "", 0.0

    matched_projects = []
    seen = set()

    for c in chunks:
        text = (c.content or "")
        low = text.lower()

        if requested_tech not in low:
            continue

        project_match = re.search(
            r"Project:\s*(.+)", text, flags=re.IGNORECASE)
        if not project_match:
            continue

        project_name = project_match.group(1).strip()
        if project_name and project_name not in seen:
            seen.add(project_name)
            matched_projects.append(project_name)

    if not matched_projects:
        return True, (
            f"I could not find any project in the current documents that explicitly mentions {requested_tech.title()}."
        ), 0.20

    answer = "The projects that explicitly mention " + TECH_DISPLAY_NAMES.get(requested_tech, requested_tech) + \
        " are:\n- "
    answer += "\n- ".join(matched_projects)

    return True, answer, 0.35


def try_extract_project_list(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Extract a distinct list of project names when the user asks broadly
    about which projects Samah worked on.

    Important:
    For broad project-list questions, do NOT rely only on retrieved chunks,
    because vector retrieval may return only one project. Instead, fetch all
    active project chunks from the projects document.
    """
    q = (question or "").strip().lower()

    list_markers = [
        "what projects",
        "which projects",
        "list projects",
        "projects samah worked",
        "what project samah worked",
        "what projects did she work on",
        "projects has she worked on",
        "what are the projects",
        "projects that samah work",
        "projects that samah worked",
        "projects she worked",
        "worked on",
    ]

    if not any(marker in q for marker in list_markers):
        return False, "", 0.0

    # ------------------------------------------------------------
    # For broad project-list questions, fetch all active project chunks.
    # This prevents returning only the top retrieved project.
    # ------------------------------------------------------------
    project_chunks = (
        DocumentChunk.objects
        .filter(
            document__document_type="projects",
            document__is_active=True,
        )
        .order_by("chunk_index")
    )

    # Fallback to retrieved chunks if DB lookup returns nothing
    source_chunks = list(project_chunks) or chunks

    project_names = []
    seen = set()

    for chunk in source_chunks:
        text = (chunk.content or "").strip()

        for line in text.splitlines():
            line = line.strip()

            if line.lower().startswith("project:"):
                name = line.split(":", 1)[1].strip()
                key = name.lower()

                if name and key not in seen:
                    seen.add(key)
                    project_names.append(name)

    if not project_names:
        return True, "I couldn’t extract a clean list of projects from the uploaded documents.", 0.0

    answer = "Samah has worked on the following projects:\n" + "\n".join(
        f"- {name}" for name in project_names
    )

    return True, answer, 0.85


def _extract_structured_skill_lines(text: str) -> List[str]:
    """
    Try to extract clean skill-like lines from structured sections only.
    Conservative by design.
    """
    lines = text.replace("\r", "\n").splitlines()
    cleaned = []

    for raw_line in lines:
        line = _normalize_skill_line(raw_line)
        if not line:
            continue
        if _looks_like_noise_line(line):
            continue
        cleaned.append(line)

    return cleaned


def _find_known_skills(text: str) -> List[str]:
    """
    Match known skills directly from retrieved text.
    Sort by first appearance for more natural ordering.
    """
    lower = text.lower()
    found = []

    for skill in KNOWN_SKILLS:
        if skill.lower() in lower:
            found.append((lower.find(skill.lower()), skill))

    found.sort(key=lambda x: x[0])
    return [skill for _, skill in found]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output = []

    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            output.append(item.strip())

    return output


def _group_skills(skills: List[str]) -> str:
    """
    Format the final answer in grouped categories.
    """
    categories = {
        "Programming Languages": ["Python", "JavaScript", "TypeScript"],
        "Backend": ["Django", "Django REST Framework", "FastAPI", "Flask"],
        "Frontend": ["React", "Next.js", "Tailwind CSS", "Bootstrap"],
        "Databases": ["PostgreSQL", "MongoDB", "MySQL", "SQL Server", "Oracle"],
        "DevOps / API / Security": ["Docker", "NGINX", "Gunicorn", "JWT", "RBAC", "Swagger", "OpenAPI"],
        "AI / LLM": ["Hugging Face", "BERT", "RoBERTa", "T5", "LangChain", "Langfuse", "Ollama", "Gemini", "RAG", "Vector Search"],
        "OCR / Document Processing": ["Tesseract", "EasyOCR", "OpenCV"],
    }

    grouped_output = []

    for category, category_skills in categories.items():
        matched = [skill for skill in category_skills if skill in skills]
        if matched:
            grouped_output.append(f"- {category}: {', '.join(matched)}")

    if not grouped_output:
        return ""

    return "Samah’s technical skills:\n" + "\n".join(grouped_output)


def try_extract_project_fit(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Applies when the user is asking whether Samah can help with, build,
    or be a fit for a project/solution/use case.

    This extractor is generic:
    - it does not hardcode any one technology
    - it tries to detect the requested project/topic from the question
    - it answers as a capability/fit assessment, not as a raw skill list
    """
    q = (question or "").strip()
    low = q.lower()

    project_fit_markers = [
        "can you develop something like that",
        "can she develop something like that",
        "can samah develop something like that",
        "can she build something like that",
        "can samah build something like that",
        "can she help with this",
        "can samah help with this",
        "is she a fit for this",
        "is samah a fit for this",
        "can she handle this project",
        "can samah handle this project",
        "can she handle this kind of project",
        "can samah handle this kind of project",
        "can she work on this project",
        "can samah work on this project",
        "can she do this kind of project",
        "can samah do this kind of project",
        "can she build this kind of solution",
        "can samah build this kind of solution",
        "someone with",
        "this kind of project",
    ]

    if not any(marker in low for marker in project_fit_markers):
        return False, "", 0.0

    text = _joined_text(chunks)
    lower_text = text.lower()

    requested_topic = _extract_requested_topic_from_question(q)

    capability_signals = []
    if any(x in lower_text for x in ["dashboard", "dashboards", "analytics", "visualization", "reporting"]):
        capability_signals.append(
            "dashboard and analytics solution development")
    if any(x in lower_text for x in ["django", "api", "backend", "rest framework", "fastapi", "flask"]):
        capability_signals.append("backend and API development")
    if any(x in lower_text for x in ["react", "next.js", "frontend", "web platform", "full-stack"]):
        capability_signals.append("full-stack web application delivery")
    if any(x in lower_text for x in ["ai", "llm", "rag", "automation", "intelligent workflow"]):
        capability_signals.append(
            "AI-enabled and automation-focused solutions")
    if any(x in lower_text for x in ["data", "business systems", "interactive dashboards", "business dashboards"]):
        capability_signals.append("data-driven business system implementation")

    # Deduplicate while preserving order
    seen = set()
    capability_signals = [
        x for x in capability_signals
        if not (x in seen or seen.add(x))
    ]

    # Conservative direct-topic detection:
    # only claim direct experience if the exact topic appears in retrieved evidence
    direct_topic_match = False
    if requested_topic:
        direct_topic_match = requested_topic.lower() in lower_text

    if requested_topic:
        if direct_topic_match:
            answer = (
                f"Yes — Samah appears to have relevant experience related to {requested_topic}. "
                f"Based on what I found, she seems to be a good fit for this type of project."
            )
        else:
            if capability_signals:
                answer = (
                    f"I could not confirm direct experience specifically with {requested_topic}. "
                    f"However, Samah does have strong related experience in "
                    f"{', '.join(capability_signals[:4])}. "
                    f"So she may still be a good fit if the project involves similar dashboard, backend, data, or solution-delivery work."
                )
            else:
                answer = (
                    f"I could not confirm direct experience specifically with {requested_topic}, "
                    f"and I also do not see enough closely related background yet to assess the fit with confidence."
                )
    else:
        if capability_signals:
            answer = (
                "Samah seems like a reasonable fit for this kind of project based on her experience in "
                f"{', '.join(capability_signals[:4])}. "
                "The final fit would depend on the required tools, project scope, and whether it needs a highly specialized technology."
            )
        else:
            answer = (
                "I’m not able to judge the fit confidently yet based on what I found."
            )

    return True, answer, 0.18


def _extract_requested_topic_from_question(question: str) -> str:
    """
    Extract the requested technology / platform / topic from broad project-fit questions
    without swallowing trailing conversational phrases.
    """
    q = (question or "").strip()
    if not q:
        return ""

    patterns = [
        r"skills?\s+(?:of|in|for)\s+([A-Za-z0-9.+#\-/ ]+?)(?:\s+can\s+|[?.!,]|$)",
        r"experience\s+(?:in|with)\s+([A-Za-z0-9.+#\-/ ]+?)(?:\s+can\s+|[?.!,]|$)",
        r"project\s+(?:using|with|in)\s+([A-Za-z0-9.+#\-/ ]+?)(?:\s+can\s+|[?.!,]|$)",
        r"something\s+(?:like|with)\s+([A-Za-z0-9.+#\-/ ]+?)(?:\s+can\s+|[?.!,]|$)",
        r"build\s+(?:with|in|using)\s+([A-Za-z0-9.+#\-/ ]+?)(?:\s+can\s+|[?.!,]|$)",
        r"need[s]?\s+someone\s+with\s+([A-Za-z0-9.+#\-/ ]+?)\s+experience(?:\s+|[?.!,]|$)",
        r"need[s]?\s+([A-Za-z0-9.+#\-/ ]+?)\s+skills?(?:\s+can\s+|[?.!,]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            topic = match.group(1).strip(" ?.!،,")
            topic = re.sub(r"\s+", " ", topic).strip()
            if topic:
                return topic

    return ""


def try_extract_skills(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    """
    Handles only broad/general skills questions.
    It should NOT answer project-specific tool/technology questions.
    """
    q = (question or "").strip().lower()

    general_patterns = [
        r"\bwhat skills\b",
        r"\btechnical skills\b",
        r"\bwhat technologies does (she|samah) know\b",
        r"\bwhat tools does (she|samah) know\b",
        r"\btech stack\b",
        r"\btechnology stack\b",
        r"\bframeworks (does she use|used by samah)?\b",
        r"\bwhat can (she|samah) use\b",
        r"\bwhat does (she|samah) know\b",
    ]

    # If question looks project-specific, do NOT let this extractor handle it
    project_specific_markers = [
        "build",
        "used in",
        "used for",
        "used to build",
        "for the dashboard",
        "for this project",
        "in this project",
        "in the project",
        "for spend analysis",
        "for property chatbot",
        "for payroll",
        "for electricity",
        "dashboard",
        "project",
        "chatbot",
        "system",
        "platform",
        "solution",
    ]

    is_general = any(re.search(p, q) for p in general_patterns)
    is_project_specific = any(
        marker in q for marker in project_specific_markers)

    if not is_general or is_project_specific:
        return False, "", 0.0

    text = _joined_text(chunks)

    found_known = _find_known_skills(text)
    structured_lines = _extract_structured_skill_lines(text)

    known_skill_lookup = {skill.lower(): skill for skill in KNOWN_SKILLS}
    structured_as_known = []

    for line in structured_lines:
        normalized = line.strip().lower()
        if normalized in known_skill_lookup:
            structured_as_known.append(known_skill_lookup[normalized])

    combined = _dedupe_preserve_order(found_known + structured_as_known)

    if not combined:
        return True, "I couldn’t extract a clean general skills list from the uploaded documents yet.", 0.0

    answer = _group_skills(combined)
    if not answer:
        answer = "Samah’s technical skills (from the uploaded documents):\n- " + \
            "\n- ".join(combined[:20])

    return True, answer, 0.45


def try_extract_strengths(question: str, chunks: List[DocumentChunk]) -> Tuple[bool, str, float]:
    q = (question or "").lower()

    keywords = [
        "strongest technical areas",
        "strongest areas",
        "technical strengths",
        "core strengths",
        "main strengths",
        "strongest in",
    ]

    if not any(k in q for k in keywords):
        return False, "", 0.0

    text = _joined_text(chunks).lower()

    findings = []

    if any(x in text for x in ["backend development", "django", "api"]):
        findings.append(
            "Backend development, especially with Django and API architecture")

    if any(x in text for x in ["ai", "llm", "rag", "langchain", "ollama", "gemini"]):
        findings.append("AI / LLM integration and intelligent workflow design")

    if any(x in text for x in ["automation", "workflow"]):
        findings.append("Workflow automation and business process improvement")

    if any(x in text for x in ["document", "ocr", "tesseract", "easyocr", "opencv"]):
        findings.append(
            "Document intelligence and OCR-based processing pipelines")

    if any(x in text for x in ["react", "next.js", "full-stack"]):
        findings.append("Full-stack solution delivery using Next.js and React")

    if not findings:
        return False, "", 0.0

    findings = findings[:5]

    answer = (
        "Samah’s strongest technical areas include:\n- " +
        "\n- ".join(findings)
    )
    return True, answer, 0.20


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

    Behavior:
    - freelance question -> only freelance answer
    - remote/hybrid/on-site question -> only work arrangement answer
    - location question -> only location answer
    - salary/payment question -> only compensation answer
    - full-time question -> only full-time answer
    - broad availability question -> combined summary
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
        "location",
        "locations",
        "dubai",
        "abu dhabi",
        "uae",
    ]

    if not any(k in q for k in keywords):
        return False, "", 0.0

    text = _joined_text(chunks)
    lower = text.lower()

    has_full_time = "open to full-time" in lower
    has_freelance = (
        "open to freelance" in lower
        or "open to freelance or project-based work" in lower
    )
    has_work_mode = (
        "remote work" in lower
        or "hybrid work" in lower
        or "on-site work" in lower
        or "on-site" in lower
    )
    has_locations = any(x in lower for x in ["dubai", "abu dhabi", "uae"])
    has_compensation_note = (
        "fair and competitive compensation package" in lower
        or "compensation expectations should be discussed" in lower
        or "target compensation range should be discussed" in lower
    )

    asks_freelance = "freelance" in q or "contract" in q or "project-based" in q
    asks_remote = any(
        x in q for x in ["remote", "hybrid", "on-site", "onsite"])
    asks_location = any(
        x in q for x in ["location", "locations", "dubai", "abu dhabi", "uae"])
    asks_compensation = any(
        x in q for x in ["salary", "payment", "compensation", "rate"])
    asks_full_time = "full-time" in q
    asks_broad_availability = any(x in q for x in ["availability", "available", "open to work"]) and not (
        asks_freelance or asks_remote or asks_location or asks_compensation or asks_full_time
    )

    # Specific: freelance
    if asks_freelance:
        if has_freelance:
            return True, "Samah is open to freelance and project-based work.", 0.10
        return True, "I couldn’t find clear evidence about freelance work in the uploaded documents.", 0.0

    # Specific: remote / hybrid / on-site
    if asks_remote:
        if has_work_mode:
            return True, "Samah is open to remote, hybrid, and suitable on-site opportunities.", 0.10
        return True, "I couldn’t find clear evidence about work arrangement preferences in the uploaded documents.", 0.0

    # Specific: locations
    if asks_location:
        if has_locations:
            return True, "Samah is especially open to opportunities in Dubai, Abu Dhabi, the UAE, and strong remote opportunities.", 0.10
        return True, "I couldn’t find location preferences in the uploaded documents.", 0.0

    # Specific: compensation
    if asks_compensation:
        if has_compensation_note:
            return True, (
                "Compensation is intended to be discussed based on role scope, technical depth, "
                "leadership responsibility, and work arrangement."
            ), 0.10
        return True, "I couldn’t find a fixed compensation range in the uploaded documents.", 0.0

    # Specific: full-time
    if asks_full_time:
        if has_full_time:
            return True, "Samah is open to full-time opportunities.", 0.10
        return True, "I couldn’t find clear evidence about full-time availability in the uploaded documents.", 0.0

    # Broad availability question
    if asks_broad_availability:
        findings = []

        if has_full_time:
            findings.append("Samah is open to full-time opportunities.")
        if has_freelance:
            findings.append(
                "Samah is open to freelance and project-based work.")
        if has_work_mode:
            findings.append(
                "Samah is open to remote, hybrid, and suitable on-site opportunities.")
        if has_locations:
            findings.append(
                "Samah is especially open to opportunities in Dubai, Abu Dhabi, the UAE, and strong remote opportunities.")
        if has_compensation_note:
            findings.append(
                "Compensation is intended to be discussed based on role scope, technical depth, leadership responsibility, and work arrangement."
            )

        if not findings:
            return False, "", 0.0

        return True, "\n".join(findings[:5]), 0.10
