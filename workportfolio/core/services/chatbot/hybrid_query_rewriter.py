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

    _spellchecker = None
    
    ENGLISH_ONLY_MESSAGE = (
        "Thank you for your message. The chatbot currently supports English only. "
        "Please type your query in English so I can assist you.\n\n"
        "شكراً لرسالتك. حالياً يدعم الشات بوت اللغة الإنجليزية فقط. "
        "يرجى كتابة استفسارك باللغة الإنجليزية حتى أتمكن من مساعدتك."
    )

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
                    replacement.capitalize() if token[:1].isupper() else replacement
                )
                continue

            # Protect technical/domain words
            if low in cls.SPELL_PROTECTED_TERMS:
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
                    suggestion.capitalize() if token[:1].isupper() else suggestion
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
    def rewrite_cached(
        cls,
        user_query: str,
        history: list[dict] | None = None,
    ) -> Dict[str, Any]:
        q = (user_query or "").strip()
        if not q:
            return {"rewritten_query": "", "notes": "empty"}

        local = cls._local_rewrite(q)

        rule_based = cls._rule_based_followup_rewrite(q, history)
        if rule_based:
            return {
                "rewritten_query": rule_based,
                "notes": "rule_based_followup",
                "meta": {
                    "provider": "local_rule",
                    "model_used": None,
                    "tried_models": [],
                    "error": None,
                },
            }

        if cls._should_skip_llm(q, local) and not cls._looks_context_dependent(q):
            return {
                "rewritten_query": local,
                "notes": "local_fast_path",
                "meta": {
                    "provider": "local",
                    "model_used": None,
                    "tried_models": [],
                    "error": None,
                },
            }

        history_text = cls._format_history_for_prompt(history)

        system_instruction = (
            "You rewrite user queries to improve document retrieval.\n"
            "Rules:\n"
            "1) Keep the EXACT same meaning and intent.\n"
            "2) The CURRENT user query is always the primary intent.\n"
            "3) Use recent conversation only to resolve vague references like 'it', 'that', 'she', 'her', 'this project', or 'again'.\n"
            "4) Never replace the main topic of the current query with a previous topic unless the current query clearly depends on it.\n"
            "5) If the current query explicitly asks about experience, skills, contact, or compensation, preserve that exact intent.\n"
            "6) Do NOT reinterpret an experience question into a compensation question.\n"
            "7) Do NOT reinterpret a contact/discussion question into a compensation question unless the user explicitly asks about compensation.\n"
            "8) Fix typos, spacing, capitalization, and minor grammar only when helpful.\n"
            "9) Preserve person perspective exactly. Never change 'you' to 'I', 'your' to 'my', or names to pronouns.\n"
            "10) Preserve the original language of the query. If the query is Arabic, keep it Arabic. If mixed, keep the same mixed style unless a tiny correction is needed.\n"
            "11) Preserve proper nouns, acronyms, project names, and domain-specific terms exactly when possible.\n"
            "12) Do NOT replace specific terms with broad synonyms if that could hurt retrieval.\n"
            "13) If the current query is context-dependent, rewrite it into a standalone retrieval query using only the provided recent conversation.\n"
            "14) Return JSON only.\n"
        )

        prompt = (
            "Return JSON with keys:\n"
            "- rewritten_query: string\n"
            "- notes: short string\n\n"
            "Important:\n"
            "- Keep the same person perspective.\n"
            "- Keep the same language as the original query.\n"
            "- Preserve project names and technical terms.\n"
            "- The current user query is the main source of intent.\n"
            "- Use recent conversation only to resolve vague references.\n"
            "- Do not let a previous compensation question override a current experience or contact question.\n"
            "- Do not invent a new interpretation of a vague query.\n"
            "- If the original query is already usable, make only minimal edits.\n"
            "- If the current query is a follow-up that depends on recent conversation context,\n"
            "  rewrite it into a standalone retrieval-friendly query using that context.\n\n"
            f"Recent conversation:\n{history_text or 'None'}\n\n"
            f"Current user query: {q}\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "rewritten_query": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["rewritten_query", "notes"],
            "additionalProperties": False,
        }

        chain = [getattr(settings, "REWRITE_PRIMARY_MODEL", "gemini-2.5-flash-lite")] + \
            getattr(settings, "REWRITE_FALLBACK_MODELS", ["gemini-2.5-flash"])

        ok, text, meta = LLMRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,
            model_chain=chain,
            json_schema=schema,
            task=LLMRouter.TASK_REWRITE,
        )

        if not ok:
            return {
                "rewritten_query": local,
                "notes": f"rewrite_fallback:{meta.get('error')}",
                "meta": meta,
            }

        try:
            data = json.loads(text)
            rewritten = (data.get("rewritten_query") or "").strip() or local
            rewritten = cls._local_rewrite(rewritten)

            return {
                "rewritten_query": rewritten,
                "notes": data.get("notes", "ok"),
                "meta": meta,
            }
        except Exception:
            return {
                "rewritten_query": local,
                "notes": "rewrite_json_parse_failed",
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
