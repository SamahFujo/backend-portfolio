import re


class ScopeResolver:
    CERT = re.compile(
        r"\b(cert|certificate|certificates|certification|certifications|credential|credentials|badge|badges)\b",
        re.I,
    )

    PROJECT = re.compile(
        r"\b(project|projects|portfolio|dashboard|chatbot|solution|platform|system)\b",
        re.I,
    )

    RECOMMEND = re.compile(
        r"\b(recommendation|reference letter|referee|endorsement)\b",
        re.I,
    )

    EXPERIENCE = re.compile(
        r"\b(experience letter|employment letter|employment verification|service letter)\b",
        re.I,
    )

    CV = re.compile(
        r"\b(cv|resume)\b",
        re.I,
    )

    CONTACT = re.compile(
        r"\b(contact|email|phone|whatsapp|linkedin|get in touch|reach|connect|"
        r"communicate|talk to|speak with|contact details)\b",
        re.I,
    )

    PREFERENCES = re.compile(
        r"\b("
        r"favorite|favourite|prefer|preferred|preference|preferences|"
        r"favorite language|favourite language|preferred language|"
        r"preferred backend|preferred frontend|work style|working style|"
        r"django or fastapi|backend or frontend|favorite framework|favourite framework"
        r")\b",
        re.I,
    )

    COMPENSATION = re.compile(
        r"\b("
        r"salary|payment|compensation|compansation|expected salary|salary range|pay range|"
        r"hourly rate|daily rate|monthly rate|freelance rate|consulting rate|project rate|package|"
        r"cost|price|pricing|budget|quote|quotation|"
        r"availability|available|remote|hybrid|on-site|onsite|"
        r"freelance|contract|full-time|open to work|notice period|"
        r"work arrangement|preferred location|work location"
        r")\b",
        re.I,
    )

    TECH_EXPERIENCE = re.compile(
        r"\b("
        r"oracle|postgresql|mysql|sql server|mongodb|mongo|database|databases|"
        r"django|django rest framework|drf|fastapi|flask|react|next\.?js|tailwind|"
        r"python|javascript|typescript|bootstrap|langchain|ollama|gemini|"
        r"tool|tools|technology|technologies|framework|frameworks|stack|tech stack|"
        r"worked with|experience with|used|use|built with|used to build|frontend|backend"
        r")\b",
        re.I,
    )

    SKILL_RATING = re.compile(
        r"("
        r"\brate\s+(samah|her|she)\s+(in|on|for)\s+"
        r"|"
        r"\b(score|evaluate)\s+(samah|her|she)\b"
        r"|"
        r"\b(1\s*-\s*10|1\s+to\s+10|out of 10|/10)\b"
        r")",
        re.I,
    )

    CAPABILITIES = re.compile(
        r"\b("
        r"can you do|can samah do|can she do|help with|capabilities|services|"
        r"what can you do|what can samah do|able to|"
        r"can build|build this|handle this project|fit for|suitable for|"
        r"can she help|can samah help|support this|support that"
        r")\b",
        re.I,
    )

    FAQ = re.compile(
        r"\b("
        r"faq|frequently asked questions|about samah|who is samah|"
        r"what does samah do|strongest technical areas|what is samah"
        r")\b",
        re.I,
    )

    ACHIEVEMENTS = re.compile(
        r"\b("
        r"achievement|achievements|impact|value|strengths|why should we hire|"
        r"why hire|what did she achieve|what impact|professional strength"
        r")\b",
        re.I,
    )

    CAREER_TIMELINE = re.compile(
        r"\b("
        r"career timeline|career path|career progression|timeline|"
        r"years of experience|background|career history|previous role|"
        r"before becoming|role progression"
        r")\b",
        re.I,
    )

    FOLLOWUP_PROJECT_HINT = re.compile(
        r"\b(frontend|backend|which project|which projects|what else|more details|expand)\b",
        re.I,
    )

    WORK_HISTORY = re.compile(
        r"\b("
        r"company|companies|organization|organisation|employer|employers|"
        r"worked with|worked at|work with|work at|"
        r"which company|what company|"
        r"where did she work|where she worked|"
        r"employment history|work history|professional history|"
        r"previous company|current company|previous employer|current employer|"
        r"who did she work for|who has she worked for"
        r")\b",
        re.I,
    )

    @classmethod
    def _soft_scope(cls, preferred_document_types: list[str]):
        """
        Preferred docs guide retrieval, but should not permanently trap it.
        Retrieval can retry broadly if preferred docs are weak.
        """
        return {
            "only_active_docs": True,
            "preferred_document_types": preferred_document_types,
            "allow_fallback_to_all_docs": True,
        }

    @classmethod
    def _is_project_tech_question(cls, msg: str) -> bool:
        return bool(cls.PROJECT.search(msg)) and bool(cls.TECH_EXPERIENCE.search(msg))

    @classmethod
    def _is_skill_rating_question(cls, msg: str) -> bool:
        """
        Detect skill evaluation questions such as:
        - rate Samah in Python from 1-10
        - score her Django skill out of 10
        - evaluate Samah in React
        """
        return bool(cls.SKILL_RATING.search(msg)) and bool(cls.TECH_EXPERIENCE.search(msg))

    @classmethod
    def resolve_filters(cls, message: str, route: str | None = None):
        """
        Keep ScopeResolver conservative.

        Most profile questions should search across all active documents.
        Query planning + document-type boosting will guide ranking later.

        Hard filters are kept only for direct document-specific questions
        where the intent is very clear.
        """
        msg = (message or "").strip()
        if not msg:
            return {"only_active_docs": True}

        # Direct contact/CV questions are safe to route to CV.
        if cls.CONTACT.search(msg) or cls.CV.search(msg):
            return {
                "document_type": "cv",
                "only_active_docs": True,
            }

        # Direct recommendation-letter questions.
        if cls.RECOMMEND.search(msg):
            return {
                "document_type": "recommendation",
                "only_active_docs": True,
            }

        # Direct experience-letter questions.
        if cls.EXPERIENCE.search(msg):
            return {
                "document_type": "experience_letter",
                "only_active_docs": True,
            }

        # Direct certificate questions.
        if cls.CERT.search(msg):
            return {
                "document_type": "certificates",
                "only_active_docs": True,
            }

        # Direct compensation / availability questions.
        # This remains safe because "rate Samah in Python" is handled by the query plan
        # and should avoid compensation.
        if cls.COMPENSATION.search(msg):
            return {
                "document_type": "compensation",
                "only_active_docs": True,
            }

        # Default: broad retrieval.
        return {
            "only_active_docs": True,
        }