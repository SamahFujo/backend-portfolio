from __future__ import annotations
from spellchecker import SpellChecker

import json
import re
from typing import Dict, Any
from django.conf import settings

from core.services.llm.router import LLMRouter


class GeminiQueryRewriter:
    TYPO_MAP = {
        "certficate": "certificate",
        "certifcate": "certificate",
        "certfication": "certification",
        "certifcation": "certification",
        "certfications": "certifications",
        "cirtifications": "certifications",
        "experiance": "experience",
        "machien": "machine",
        "backgroud": "background",
        "projcts": "projects",
        "skilss": "skills",
        "dashbords": "dashboards",
        "postgress": "postgresql",
    }

    TERM_MAP = {
        "next js": "Next.js",
        "react js": "React",
        "drf": "Django REST Framework",
        "llms": "LLMs",
        "ocr": "OCR",
        "api": "API",
        "ai": "AI",
        "nlp": "NLP",
        "rag": "RAG",
        "bert": "BERT",
        "roberta": "RoBERTa",
        "langchain": "LangChain",
        "ollama": "Ollama",
        "hugging face": "Hugging Face",
        "gemini api": "Gemini API",
        "django": "Django",
        "mongodb": "MongoDB",
        "nginx": "Nginx",
        "postgresql": "PostgreSQL",
        "python": "Python",
        "react": "React",
        "docker": "Docker",
        "oracle database": "Oracle Database",
        "chatgpt": "ChatGPT",
        "claude": "Claude",
    }

    YES_NO_STARTS = (
        "is ", "are ", "do ", "does ", "did ", "can ", "could ",
        "should ", "has ", "have ", "was ", "were ",
        "what ", "which ", "who ", "where ", "when ", "why ", "how ",
        "ما ", "هل ", "من ", "متى ", "أين ", "لماذا ", "كيف "
    )

    SPELLCHECK_ENABLED = True

    SPELL_PROTECTED_TERMS = {
        # AI / ML / NLP
        "ai", "ml", "nlp", "llm", "llms", "rag", "ocr",
        "bert", "roberta", "t5", "huggingface", "transformers",
        "langchain", "langfuse", "ollama", "gemini", "openwebui",

        # backend / frontend / infra
        "django", "drf", "fastapi", "flask", "react", "next", "nextjs", "next.js",
        "typescript", "javascript", "tailwind", "bootstrap",
        "docker", "nginx", "gunicorn", "postman",

        # databases / platforms
        "postgres", "postgresql", "mongodb", "mysql", "oracle", "sql", "api", "apis",
        "azure", "aws", "jwt", "rbac",

        # project / profile-specific
        "samah", "jina", "unspsc", "coursiv", "claude", "chatgpt",

        # common tech terms that might be misspelled but are important to preserve
        "roberta", "pytorch", "tensorflow", "scikit", "sklearn", "pandas", "numpy", "redis",
        "qdrant", "chroma", "pgvector", "jinja", "jinja2", "jupyter", "cuda", "vue", "node", "nodejs", "rest", "restful",

        "chatbot", "chatbots",
        "deepseek",
        "samah.ai",
        "next.js",

    }

    SPELL_FORCE_MAP = {
        "cirtifications": "certifications",
        "certfications": "certifications",
        "certfication": "certification",
        "certifcation": "certification",
        "certficate": "certificate",
        "certifcate": "certificate",
        "experiance": "experience",
        "backgroud": "background",
        "projcts": "projects",
        "skilss": "skills",
        "dashbords": "dashboards",
        "postgress": "postgresql",
    }

    LOW_RISK_PRESERVE_PATTERNS = [
        "can this chatbot",
        "can the chatbot",
        "can this project",
        "can the project",
        "can this system",
        "can the system",
        "can she build",
        "can samah build",
        "can she develop",
        "can samah develop",
        "is she fit",
        "is samah fit",
        "is this suitable",
        "can it support",
        "could it support",
        "would it support",
    ]

    _spellchecker = None

    ENGLISH_ONLY_MESSAGE = (
        "Thank you for your message. The chatbot currently supports English only. "
        "Please type your query in English so I can assist you.\n\n"
        "شكراً لرسالتك. حالياً يدعم الشات بوت اللغة الإنجليزية فقط. "
        "يرجى كتابة استفسارك باللغة الإنجليزية حتى أتمكن من مساعدتك."
    )

    ALLOWED_DOCUMENT_TYPES = {
        "cv",
        "career_timeline",
        "experience_letter",
        "recommendation",
        "projects",
        "certificates",
        "achievements",
        "compensation",
        "preferences",
        "faq",
        "capabilities",
        "security_deployment",
    }

    ALLOWED_ANSWER_TYPES = {
        "profile_overview",
        "company_history",
        "work_history",
        "experience_duration",
        "technical_skills",
        "security_deployment",
        "skill_evaluation",
        "projects",
        "capabilities",
        "achievements",
        "leadership",
        "stakeholder_client_work",
        "certificates",
        "education",
        "contact",
        "compensation",
        "availability",
        "work_style",
        "preferences",
        "recommendation",
        "experience_letter",
        "general_profile",

    }

    ALLOWED_QUESTION_SHAPES = {
        "fact",
        "list",
        "timeline",
        "summary",
        "comparison",
        "explanation",
    }

    ANSWER_TYPE_TO_PREFERRED_DOCS = {
        "security_deployment": ["security_deployment", "projects", "cv"],
        "profile_overview": ["cv", "career_timeline", "faq"],
        "company_history": ["cv", "career_timeline", "experience_letter"],
        "work_history": ["cv", "career_timeline", "experience_letter"],
        "experience_duration": ["cv", "career_timeline", "experience_letter"],
        "technical_skills": ["cv", "capabilities", "projects", "security_deployment", "faq"],
        "skill_evaluation": ["cv", "capabilities", "projects", "security_deployment", "faq"],
        "projects": ["projects", "capabilities", "faq"],
        "capabilities": ["capabilities", "projects", "cv", "faq"],
        "achievements": ["achievements", "recommendation", "projects", "cv"],
        "leadership": ["cv", "career_timeline", "achievements", "recommendation"],
        "stakeholder_client_work": ["cv", "career_timeline", "achievements", "recommendation"],
        "certificates": ["certificates", "cv"],
        "education": ["cv"],
        "contact": ["cv"],
        "compensation": ["compensation"],
        "availability": ["compensation", "faq"],
        "work_style": ["preferences", "recommendation", "achievements"],
        "preferences": ["preferences", "faq"],
        "recommendation": ["recommendation"],
        "experience_letter": ["experience_letter"],
        "general_profile": ["cv", "career_timeline", "projects", "capabilities", "faq"],
    }
    
    
    @staticmethod
    def _extract_json_text(text: str) -> str:
        text = (text or "").strip()

        if not text:
            return ""

        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

            if text.lower().startswith("json"):
                text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()

        return text.strip()

    @classmethod
    def _parse_json_safely(cls, text: str) -> Dict[str, Any]:
        cleaned = cls._extract_json_text(text)

        if not cleaned:
            raise ValueError("empty_json_after_cleaning")

        try:
            return json.loads(cleaned)
        except Exception as exc:
            raise ValueError(f"invalid_json_after_cleaning:{exc}") from exc

    @staticmethod
    def _arabic_char_count(text: str) -> int:
        return len(re.findall(r"[\u0600-\u06FF]", text or ""))

    @staticmethod
    def _english_char_count(text: str) -> int:
        return len(re.findall(r"[A-Za-z]", text or ""))

    @classmethod
    def is_fully_arabic_query(cls, text: str, min_arabic_chars: int = 3) -> bool:
        if not text or not text.strip():
            return False

        arabic_count = cls._arabic_char_count(text)
        english_count = cls._english_char_count(text)

        if arabic_count < min_arabic_chars:
            return False

        if english_count > 0:
            return False

        return True

    @classmethod
    def _get_spellchecker(cls) -> SpellChecker | None:
        """
        Lazy-load spellchecker once.
        """
        if not cls.SPELLCHECK_ENABLED:
            return None

        if cls._spellchecker is None:
            cls._spellchecker = SpellChecker(language="en")

            # Protect known technical/domain words by teaching them to the dictionary
            cls._spellchecker.word_frequency.load_words(
                cls.SPELL_PROTECTED_TERMS)
            cls._spellchecker.word_frequency.load_words(
                {term.lower() for term in cls.TERM_MAP.values()}
            )

        return cls._spellchecker

    @staticmethod
    def _is_arabic_token(token: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", token or ""))

    @staticmethod
    def _is_english_token(token: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]+", token or ""))

    @staticmethod
    def _tokenize_with_punctuation(text: str) -> list[str]:
        """
        Tokenize while preserving Arabic, English, technical terms, and punctuation.
        """
        pattern = r"""
            [\u0600-\u06FF]+              |   # Arabic words
            [A-Za-z]+(?:[.+#_-][A-Za-z0-9]+)* |   # English / technical terms
            \d+                          |   # numbers
            [^\w\s]                          # punctuation
        """
        return re.findall(pattern, text, re.UNICODE | re.VERBOSE)

    @classmethod
    def _is_spellcheck_candidate(cls, token: str) -> bool:
        """
        Decide whether a token is safe to spell-correct.
        Spell-correct only plain English words.
        """
        if not token:
            return False

        # Never touch Arabic or mixed-script tokens
        if cls._is_arabic_token(token):
            return False

        # Only plain English-like words
        if not cls._is_english_token(token):
            return False

        # Skip short words
        if len(token) <= 3:
            return False

        # Skip ALL CAPS / likely acronyms
        if token.isupper():
            return False

        return True

    @staticmethod
    def _normalize_token_for_protection(token: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\.\+#]", "", (token or "").lower())

    @classmethod
    def _safe_spell_correct_query(cls, text: str) -> str:
        """
        Conservative spell-correction for natural-language retrieval queries.
        Supports mixed Arabic-English input by correcting only eligible English tokens.
        """
        if not text:
            return text

        spell = cls._get_spellchecker()
        if not spell:
            return text

        tokens = cls._tokenize_with_punctuation(text)
        corrected_tokens: list[str] = []

        for token in tokens:
            low = token.lower()
            protected_low = cls._normalize_token_for_protection(token)

            # Keep punctuation and symbols
            if not re.search(r"\w", token):
                corrected_tokens.append(token)
                continue

            # Never touch Arabic tokens
            if cls._is_arabic_token(token):
                corrected_tokens.append(token)
                continue

            # Only spell-correct safe English words
            if not cls._is_spellcheck_candidate(token):
                corrected_tokens.append(token)
                continue

            # Force-map known recurring mistakes first
            if low in cls.SPELL_FORCE_MAP:
                replacement = cls.SPELL_FORCE_MAP[low]
                corrected_tokens.append(
                    replacement.capitalize(
                    ) if token[:1].isupper() else replacement
                )
                continue

            # Protect technical/domain words
            if low in cls.SPELL_PROTECTED_TERMS or protected_low in cls.SPELL_PROTECTED_TERMS:
                corrected_tokens.append(token)
                continue

            # If known word, keep it
            if low not in spell.unknown([low]):
                corrected_tokens.append(token)
                continue

            suggestion = spell.correction(low)

            # Conservative replacement
            if suggestion and suggestion != low:
                corrected_tokens.append(
                    suggestion.capitalize(
                    ) if token[:1].isupper() else suggestion
                )
            else:
                corrected_tokens.append(token)

        rebuilt = " ".join(corrected_tokens)
        rebuilt = re.sub(r"\s+([?.!,;:؟،])", r"\1", rebuilt)
        rebuilt = re.sub(r"\(\s+", "(", rebuilt)
        rebuilt = re.sub(r"\s+\)", ")", rebuilt)
        rebuilt = re.sub(r"\s+", " ", rebuilt).strip()

        return rebuilt

    @classmethod
    def _infer_question_shape(cls, query: str, answer_type: str = "") -> str:
        q = (query or "").strip().lower()
        answer_type = (answer_type or "").strip().lower()

        if any(marker in q for marker in [
            "compare", "comparison", "difference", "versus", "vs ",
        ]):
            return "comparison"

        if answer_type in {"company_history", "work_history"} or any(marker in q for marker in [
            "timeline", "history", "career path", "career progression",
            "employment history", "roles over time",
        ]):
            return "timeline"

        if any(marker in q for marker in [
            "which", "what are", "list", "all ", "companies", "projects",
            "certificates", "certifications", "technologies",
        ]):
            return "list"

        if any(marker in q for marker in [
            "summarize", "summary", "overview", "background", "tell me about",
        ]):
            return "summary"

        if any(marker in q for marker in [
            "why", "how", "explain",
        ]):
            return "explanation"

        return "fact"

    @classmethod
    def _default_query_plan(cls, query: str, notes: str = "default") -> Dict[str, Any]:
        """
        Safe fallback query plan used when LLM rewrite fails.
        Uses deterministic local routing so FAQ does not dominate every broad query.
        """
        clean = (query or "").strip()
        q = clean.lower()

        def make_plan(
            *,
            answer_type: str,
            preferred: list[str],
            suffix: str,
            avoid: list[str] | None = None,
            plan_notes: str | None = None,
        ) -> Dict[str, Any]:
            return {
                "rewritten_query": clean,
                "retrieval_query": f"{clean} {suffix}".strip(),
                "answer_type": answer_type,
                "question_shape": cls._infer_question_shape(clean, answer_type),
                "preferred_document_types": preferred,
                "avoid_document_types": avoid or [],
                "needs_document_retrieval": True,
                "notes": plan_notes or notes,
            }

        if any(word in q for word in [
            "project", "projects", "built", "dashboard", "chatbot",
            "system", "application", "app", "portfolio website",
        ]):
            return make_plan(
                answer_type="projects",
                preferred=["projects", "capabilities"],
                suffix="Samah projects built systems dashboards chatbots applications portfolio technical outcomes",
                avoid=["compensation"],
                plan_notes="local_rule_projects",
            )

        if any(word in q for word in [
            "certificate", "certificates", "certification", "certifications",
            "certified", "course", "training",
        ]):
            return make_plan(
                answer_type="certificates",
                preferred=["certificates", "cv"],
                suffix="Samah certificates certifications professional training courses certificate names issuers",
                avoid=["faq", "compensation"],
                plan_notes="local_rule_certificates",
            )

        if any(word in q for word in [
            "career timeline", "timeline", "career history", "work history",
            "career path", "employment history", "roles", "previous role",
        ]):
            return make_plan(
                answer_type="work_history",
                preferred=["career_timeline", "cv", "experience_letter"],
                suffix="Samah career timeline work history roles companies dates professional experience CV",
                avoid=["faq", "compensation"],
                plan_notes="local_rule_career_timeline",
            )

        if any(word in q for word in [
            "achievement", "achievements", "impact", "strength", "strengths",
            "strong points", "value", "accomplishment", "accomplishments",
            "contribution", "contributions", "leadership", "stakeholder",
        ]):
            return make_plan(
                answer_type="achievements",
                preferred=["achievements", "recommendation",
                           "experience_letter", "cv"],
                suffix="Samah achievements impact strengths accomplishments leadership stakeholder contribution recommendation CV",
                avoid=["faq", "compensation"],
                plan_notes="local_rule_achievements",
            )

        if any(word in q for word in [
            "work style", "working style", "preferences", "prefer",
            "communication", "collaboration", "team style", "environment",
        ]):
            return make_plan(
                answer_type="work_style",
                preferred=["preferences", "recommendation"],
                suffix="Samah work style preferences communication collaboration teamwork working environment",
                avoid=["compensation"],
                plan_notes="local_rule_work_style",
            )

        if any(word in q for word in [
            "deployment", "deploy", "aws", "cloud", "ecs", "ecr", "rds",
            "s3", "docker", "security", "production", "devops",
            "ci/cd", "cicd", "github actions",
        ]):
            return make_plan(
                answer_type="security_deployment",
                preferred=["security_deployment", "projects", "cv"],
                suffix="Samah deployment skills AWS Docker ECS ECR RDS S3 security production CI CD DevOps",
                avoid=["faq", "compensation"],
                plan_notes="local_rule_security_deployment",
            )

        if any(word in q for word in [
            "salary", "compensation", "rate", "availability", "available",
            "notice period", "join", "joining", "expected salary",
            "freelance", "contract",
        ]):
            return make_plan(
                answer_type="compensation",
                preferred=["compensation"],
                suffix="Samah compensation availability notice period salary joining freelance contract",
                avoid=["faq"],
                plan_notes="local_rule_compensation",
            )

        if any(word in q for word in [
            "what can samah help", "can help", "services", "capabilities",
            "what can she do", "what does she do",
        ]):
            return make_plan(
                answer_type="capabilities",
                preferred=["capabilities", "projects", "cv"],
                suffix="Samah capabilities services AI backend automation full stack projects what I can help with",
                avoid=["compensation"],
                plan_notes="local_rule_capabilities",
            )

        if any(word in q for word in [
            "cv", "resume", "experience", "background", "profile",
            "skills", "technology", "technologies", "tech stack",
        ]):
            return make_plan(
                answer_type="general_profile",
                preferred=["cv", "career_timeline",
                           "projects", "capabilities"],
                suffix="Samah CV resume background experience skills technologies career projects capabilities",
                avoid=["compensation"],
                plan_notes="local_rule_profile",
            )

        return make_plan(
            answer_type="general_profile",
            preferred=["cv", "career_timeline",
                       "projects", "capabilities", "faq"],
            suffix="Samah professional profile CV career projects capabilities experience skills background",
            avoid=["compensation"],
            plan_notes=notes,
        )

    @classmethod
    def _normalize_query_plan(cls, data: Dict[str, Any], fallback_query: str) -> Dict[str, Any]:
        """
        Validate and normalize LLM query-plan output.
        This protects the rest of the chatbot from malformed LLM JSON.
        """
        fallback = cls._default_query_plan(fallback_query)

        rewritten_query = (data.get("rewritten_query")
                           or fallback_query or "").strip()
        retrieval_query = (data.get(
            "retrieval_query") or rewritten_query or fallback["retrieval_query"]).strip()

        answer_type = (data.get("answer_type") or "general_profile").strip()
        if answer_type not in cls.ALLOWED_ANSWER_TYPES:
            answer_type = "general_profile"

        question_shape = (data.get("question_shape") or "").strip().lower()
        if question_shape not in cls.ALLOWED_QUESTION_SHAPES:
            question_shape = cls._infer_question_shape(
                data.get("rewritten_query") or fallback_query,
                answer_type,
            )

        preferred = data.get("preferred_document_types") or []
        if not isinstance(preferred, list):
            preferred = []

        preferred = [
            str(item).strip()
            for item in preferred
            if str(item).strip() in cls.ALLOWED_DOCUMENT_TYPES
        ]

        if not preferred:
            preferred = cls.ANSWER_TYPE_TO_PREFERRED_DOCS.get(answer_type, [])

        effective_query = (rewritten_query or retrieval_query or fallback_query or "").lower()
        asks_for_projects = any(marker in effective_query for marker in [
            "project", "projects", "built", "developed", "implemented",
        ])

        if asks_for_projects and "projects" not in preferred:
            preferred = ["projects"] + preferred

        if asks_for_projects and answer_type == "security_deployment":
            answer_type = "projects"

        avoid = data.get("avoid_document_types") or []
        if not isinstance(avoid, list):
            avoid = []

        avoid = [
            str(item).strip()
            for item in avoid
            if str(item).strip() in cls.ALLOWED_DOCUMENT_TYPES
        ]

        return {
            "rewritten_query": rewritten_query,
            "retrieval_query": retrieval_query,
            "answer_type": answer_type,
            "question_shape": question_shape,
            "preferred_document_types": preferred,
            "avoid_document_types": avoid,
            "needs_document_retrieval": bool(data.get("needs_document_retrieval", True)),
            "notes": data.get("notes", fallback.get("notes", "ok")),
        }

    @classmethod
    def rewrite_cached(
        cls,
        user_query: str,
        history: list[dict] | None = None,
    ) -> Dict[str, Any]:
        q = (user_query or "").strip()
        if not q:
            plan = cls._default_query_plan("", notes="empty")
            return {
                **plan,
                "meta": {
                    "provider": "local_guard",
                    "model_used": None,
                    "tried_models": [],
                    "error": None,
                },
            }

        low_q = q.lower()
        if any(pattern in low_q for pattern in cls.LOW_RISK_PRESERVE_PATTERNS):
            plan = cls._default_query_plan(
                q, notes="preserved_capability_pattern")
            plan["answer_type"] = "capabilities"
            plan["preferred_document_types"] = cls.ANSWER_TYPE_TO_PREFERRED_DOCS["capabilities"]

            return {
                **plan,
                "meta": {
                    "provider": "local_guard",
                    "model_used": None,
                    "tried_models": [],
                    "error": None,
                },
            }

        local = cls._local_rewrite(q)

        rule_based = cls._rule_based_followup_rewrite(q, history)
        if rule_based:
            plan = cls._default_query_plan(
                rule_based, notes="rule_based_followup")

            return {
                **plan,
                "meta": {
                    "provider": "local_rule",
                    "model_used": None,
                    "tried_models": [],
                    "error": None,
                },
            }

        # Keep only truly safe bypasses
        if cls.is_fully_arabic_query(q):
            plan = cls._default_query_plan(q, notes="arabic_only_no_rewrite")

            return {
                **plan,
                "meta": {
                    "provider": "local_guard",
                    "model_used": None,
                    "tried_models": [],
                    "error": None,
                },
            }

        history_text = cls._format_history_for_prompt(history)

        system_instruction = (
            "You rewrite user queries and create a retrieval plan for Samah.ai's portfolio chatbot.\n"
            "The chatbot answers questions about Samah's professional profile using uploaded documents.\n\n"
            "Available document types:\n"
            "- security_deployment: deployment, AWS, Docker, ECS, ECR, RDS, S3, CI/CD, security, production setup\n"
            "- cv: resume, contact, education, professional experience, company names, core skills\n"
            "- career_timeline: career progression, role development, leadership journey, work history\n"
            "- experience_letter: formal employment confirmation and employment dates\n"
            "- recommendation: recommendation letter, professionalism, HR endorsement, interpersonal and organizational strengths\n"
            "- projects: project portfolio, delivered systems, technical contributions, outcomes\n"
            "- certificates: professional certificates and training\n"
            "- achievements: impact, strengths, value, leadership, stakeholder contribution\n"
            "- compensation: salary expectations, availability, freelance, contract, work arrangement, location preference\n"
            "- preferences: favorite technologies, work style, preferred stack, collaboration style\n"
            "- faq: common direct questions and concise answers\n"
            "- capabilities: services, what Samah can build, AI/backend/full-stack capabilities\n\n"
            "Allowed answer_type values:\n"
            "profile_overview, company_history, work_history, experience_duration, technical_skills, skill_evaluation, "
            "projects, capabilities, achievements, leadership, stakeholder_client_work, certificates, education, contact, "
            "compensation, availability, work_style, preferences, recommendation, experience_letter, general_profile.\n\n"
            "Rules:\n"
            "1) Keep the exact user intent.\n"
            "2) Fix grammar and spelling when helpful.\n"
            "3) Do not confuse payment rate with skill rating.\n"
            "4) 'rate Samah in Python from 1-10' = skill_evaluation, avoid compensation.\n"
            "5) 'which company did she work with/for' = company_history, prefer cv, career_timeline, experience_letter.\n"
            "6) Contact questions should prefer cv.\n"
            "7) Salary, hourly rate, freelance, contract, availability, remote/hybrid/on-site questions should prefer compensation.\n"
            "8) Favorite language/framework/work style questions should prefer preferences and faq.\n"
            "9) Project questions should prefer projects, capabilities, and faq.\n"
            "10) Achievement, strength, impact, leadership, stakeholder questions should prefer achievements, cv, recommendation, and career_timeline.\n"
            "11) Return JSON only."
        )

        prompt = (
            "Return JSON exactly with these keys:\n"
            "{"
            "\"rewritten_query\":\"clean standalone user question\","
            "\"retrieval_query\":\"optimized semantic retrieval query\","
            "\"answer_type\":\"one allowed answer_type\","
            "\"question_shape\":\"fact|list|timeline|summary|comparison|explanation\","
            "\"preferred_document_types\":[\"cv\"],"
            "\"avoid_document_types\":[\"compensation\"],"
            "\"needs_document_retrieval\":true,"
            "\"notes\":\"short explanation\""
            "}\n\n"
            "Important:\n"
            "- retrieval_query should include useful keywords likely to appear in the right documents.\n"
            "- preferred_document_types should use only available document types.\n"
            "- avoid_document_types should be used only when a document type is likely misleading.\n"
            "- Do not invent facts about Samah.\n"
            "- The current user query is the main intent.\n"
            "- Recent conversation is only for resolving vague references.\n\n"
            f"Recent conversation:\n{history_text or 'None'}\n\n"
            f"Current user query: {q}\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "rewritten_query": {"type": "string"},
                "retrieval_query": {"type": "string"},
                "answer_type": {"type": "string"},
                "question_shape": {"type": "string"},
                "preferred_document_types": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "avoid_document_types": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "needs_document_retrieval": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            "required": [
                "rewritten_query",
                "retrieval_query",
                "answer_type",
                "question_shape",
                "preferred_document_types",
                "avoid_document_types",
                "needs_document_retrieval",
                "notes",
            ],
            "additionalProperties": False,
        }

        chain = [getattr(settings, "REWRITE_PRIMARY_MODEL", "deepseek-chat")] + \
            getattr(settings, "REWRITE_FALLBACK_MODELS", ["deepseek-chat"])

        ok, text, meta = LLMRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,
            model_chain=chain,
            json_schema=schema,
            task=LLMRouter.TASK_REWRITE,
        )

        if not ok:
            plan = cls._default_query_plan(
                local,
                notes="rewrite_json_parse_failed",
            )

            return {
                **plan,
                "meta": meta,
            }

        try:
            data = cls._parse_json_safely(text)

            plan = cls._normalize_query_plan(
                data=data,
                fallback_query=local,
            )

            plan["rewritten_query"] = cls._local_rewrite(
                plan["rewritten_query"])
            plan["retrieval_query"] = cls._local_rewrite(
                plan["retrieval_query"])

            return {
                **plan,
                "meta": meta,
            }
        except Exception:
            plan = cls._default_query_plan(
                local,
                notes=f"rewrite_fallback:{meta.get('error')}",
            )

            return {
                **plan,
                "meta": meta,
            }

    @classmethod
    def _format_history_for_prompt(cls, history: list[dict] | None) -> str:
        if not history:
            return ""

        lines = []
        for item in history[-6:]:
            role = (item.get("role") or "").strip().lower()
            content = (item.get("content") or "").strip()
            if not content:
                continue

            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")

        return "\n".join(lines)

    @classmethod
    def _rule_based_followup_rewrite(
        cls,
        q: str,
        history: list[dict] | None = None,
    ) -> str | None:
        text = (q or "").strip()
        low = text.lower()

        # 1) Explicit current-turn compensation intent should remain unchanged
        if any(
            phrase in low
            for phrase in [
                "hourly rate",
                "daily rate",
                "rate per hour",
                "compensation",
                "salary",
                "payment",
            ]
        ):
            return "What is Samah’s hourly rate?" if "hourly rate" in low else text

        # 2) Generic experience questions about any technology/tool/platform
        experience_patterns = [
            r"(?:didn['’]?t have experience\s+(?:in|with)\s+)(.+)$",
            r"(?:doesn['’]?t have experience\s+(?:in|with)\s+)(.+)$",
            r"(?:experience\s+(?:in|with)\s+)(.+)$",
            r"(?:have experience\s+(?:in|with)\s+)(.+)$",
        ]

        for pattern in experience_patterns:
            match = re.search(pattern, low, flags=re.IGNORECASE)
            if match:
                raw_topic = match.group(1).strip(" ?.!،,")
                topic = cls._recover_original_span(text, raw_topic)
                if topic:
                    return f"Does Samah have experience with {topic}?"

        # 3) Generic contact/discussion follow-up
        if any(
            phrase in low
            for phrase in [
                "discuss it",
                "with who",
                "who should i contact",
                "how can i discuss this",
                "how can i reach her",
            ]
        ):
            return "How can I contact Samah to discuss this project?"

        return None

    @classmethod
    def _looks_context_dependent(cls, q: str) -> bool:
        low = (q or "").strip().lower()

        context_markers = [
            "it",
            "that",
            "this project",
            "again",
            "with who",
            "discuss it",
            "so she",
            "then can she",
        ]

        return any(marker in low for marker in context_markers)

    @staticmethod
    def _recover_original_span(original_text: str, normalized_fragment: str) -> str:
        """
        Try to recover a cleaner topic span from the original query text
        while preserving user casing where possible.
        """
        if not original_text or not normalized_fragment:
            return normalized_fragment

        pattern = re.compile(re.escape(normalized_fragment), re.IGNORECASE)
        match = pattern.search(original_text)
        if match:
            return original_text[match.start():match.end()].strip(" ?.!،,")

        return normalized_fragment.strip(" ?.!،,")

    @classmethod
    def _local_rewrite(cls, q: str) -> str:
        text = (q or "").strip()
        if not text:
            return ""

        original = text

        # Existing deterministic typo fixes first
        text = re.sub(r"\bdo you no\b", "do you know",
                      text, flags=re.IGNORECASE)

        for wrong, correct in cls.TYPO_MAP.items():
            text = re.sub(
                rf"\b{re.escape(wrong)}\b",
                correct,
                text,
                flags=re.IGNORECASE,
            )

        for wrong, correct in cls.TERM_MAP.items():
            text = re.sub(
                rf"\b{re.escape(wrong)}\b",
                correct,
                text,
                flags=re.IGNORECASE,
            )

        # Project/domain-specific cleanup
        text = re.sub(r"\bBERT\s+RoBERTa\b", "BERT and RoBERTa", text)
        text = re.sub(r"\bbert\s+roberta\b", "BERT and RoBERTa",
                      text, flags=re.IGNORECASE)

        # New guarded spellchecking layer
        text = cls._safe_spell_correct_query(text)

        # Final cleanup of spacing and punctuation
        text = re.sub(r"\bsamah\b", "Samah", text, flags=re.IGNORECASE)

        # Final normalization
        text = re.sub(r"\s+", " ", text).strip()

        if text and cls._looks_mostly_english(text) and len(text.split()) > 1:
            text = cls._capitalize_first(text)

        if text and text[-1] not in ".?!؟":
            low = text.lower()
            word_count = len(text.split())
            if word_count > 3 and low.startswith(cls.YES_NO_STARTS):
                text += "?" if not cls._contains_arabic(text) else "؟"

        return text or original

    @classmethod
    def _should_skip_llm(cls, original: str, local: str) -> bool:
        q = (original or "").strip()
        if not q:
            return True

        word_count = len(q.split())
        has_arabic = cls._contains_arabic(q)
        has_english = cls._contains_english(q)
        mixed_language = has_arabic and has_english

        complex_markers = (
            "compare ",
            "across ",
            "based on ",
            "combination of ",
            "how does ",
            "relationship between ",
            "together",
        )

        low_q = q.lower()
        if any(marker in low_q for marker in complex_markers):
            return False

        if word_count <= 2:
            return True

        if word_count <= 8 and not mixed_language:
            return True

        if word_count <= 12 and cls._is_small_change(q, local):
            return True

        if mixed_language and word_count <= 6 and cls._is_small_change(q, local):
            return True

        return False

    @staticmethod
    def _capitalize_first(text: str) -> str:
        return text[0].upper() + text[1:] if text else text

    @staticmethod
    def _contains_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text or ""))

    @staticmethod
    def _contains_english(text: str) -> bool:
        return bool(re.search(r"[A-Za-z]", text or ""))

    @classmethod
    def _looks_mostly_english(cls, text: str) -> bool:
        return cls._contains_english(text) and not cls._contains_arabic(text)

    @staticmethod
    def _normalize_for_compare(text: str) -> str:
        t = (text or "").strip().lower()
        t = re.sub(r"[?.!؟]", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    @classmethod
    def _is_small_change(cls, original: str, rewritten: str) -> bool:
        o = cls._normalize_for_compare(original)
        r = cls._normalize_for_compare(rewritten)
        if o == r:
            return True
        return abs(len(o) - len(r)) <= 12
