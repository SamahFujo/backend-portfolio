# --------------------------------------------------------------------------------
# GeminiQueryRewriter: Hybrid local + LLM query rewriting for better retrieval
# --------------------------------------------------------------------------------
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, Any
from django.conf import settings

from core.services.llm.router import LLMRouter


class GeminiQueryRewriter:
    """
    Hybrid rewrite strategy:
    1) Fast local normalization first
    2) Skip LLM for simple/clear queries
    3) Use tiny local Ollama model only for harder cases
    """

    TYPO_MAP = {
        "certficate": "certificate",
        "certifcate": "certificate",
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
        "i ": "I "
    }

    YES_NO_STARTS = (
        "is ", "are ", "do ", "does ", "did ", "can ", "could ",
        "should ", "has ", "have ", "was ", "were ",
        "what ", "which ", "who ", "where ", "when ", "why ", "how ",
        "ما ", "هل ", "من ", "متى ", "أين ", "لماذا ", "كيف "
    )

    @classmethod
    @lru_cache(maxsize=512)
    def rewrite_cached(cls, user_query: str) -> Dict[str, Any]:
        q = (user_query or "").strip()
        if not q:
            return {"rewritten_query": "", "notes": "empty"}

        local = cls._local_rewrite(q)

        # Fast path: if the query is simple enough, skip the model
        if cls._should_skip_llm(q, local):
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

        system_instruction = (
            "You rewrite user queries to improve document retrieval.\n"
            "Rules:\n"
            "1) Keep the EXACT same meaning and intent.\n"
            "2) Fix typos, spacing, capitalization, and minor grammar only when helpful.\n"
            "3) Preserve person perspective exactly. Never change 'you' to 'I', 'your' to 'my', or names to pronouns.\n"
            "4) Preserve the original language of the query. If the query is Arabic, keep it Arabic. If mixed, keep the same mixed style unless a tiny correction is needed.\n"
            "5) Preserve proper nouns, acronyms, project names, and domain-specific terms exactly when possible.\n"
            "6) Do NOT reinterpret the question into a different intent.\n"
            "7) Do NOT replace specific terms with broad synonyms if that could hurt retrieval.\n"
            "8) If the query is already clear, keep it very close to the original.\n"
            "9) Return JSON only.\n"
        )

        prompt = (
            "Return JSON with keys:\n"
            "- rewritten_query: string\n"
            "- notes: short string\n\n"
            "Important:\n"
            "- Keep the same person perspective.\n"
            "- Keep the same language as the original query.\n"
            "- Preserve project names and technical terms.\n"
            "- Do not invent a new interpretation of a vague query.\n"
            "- If the original query is already usable, make only minimal edits.\n\n"
            f"User query: {q}\n"
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

        chain = [getattr(settings, "OLLAMA_REWRITE_PRIMARY", "gemma3:1b")] + \
            getattr(settings, "OLLAMA_REWRITE_FALLBACKS", [])

        ok, text, meta = LLMRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,
            model_chain=chain,
            json_schema=schema,
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
    def _local_rewrite(cls, q: str) -> str:
        text = (q or "").strip()

        if not text:
            return ""

        original = text

        # typo normalization
        for wrong, correct in cls.TYPO_MAP.items():
            # handle common OCR-like typos and spacing issues
            text = re.sub(r"\bdo you no\b", "do you know",
            text, flags=re.IGNORECASE)
            text = re.sub(rf"\b{re.escape(wrong)}\b",
            correct, text, flags=re.IGNORECASE)

        # term normalization
        for wrong, correct in cls.TERM_MAP.items():
            text = re.sub(rf"\b{re.escape(wrong)}\b",
            correct, text, flags=re.IGNORECASE)
            
        # handle specific multi-term patterns that could be misinterpreted as separate entities
        text = re.sub(r"\bBERT\s+RoBERTa\b", "BERT and RoBERTa", text)
        text = re.sub(r"\bbert\s+roberta\b", "BERT and RoBERTa", text, flags=re.IGNORECASE)
        
        # collapse spaces
        text = re.sub(r"\s+", " ", text).strip()
        

        # normalize basic question capitalization for English-like queries
        if text and cls._looks_mostly_english(text) and len(text.split()) > 1:
            text = cls._capitalize_first(text)

        # add question mark for obvious question-like inputs
        if text and text[-1] not in ".?!؟":
            low = text.lower()
            word_count = len(text.split())
            if word_count > 3 and low.startswith(cls.YES_NO_STARTS):
                text += "?" if not cls._contains_arabic(text) else "؟"

        return text or original

    @classmethod
    def _should_skip_llm(cls, original: str, local: str) -> bool:
        """
        Skip LLM when:
        - query is short/simple
        - local rewrite is enough
        - no strong ambiguity
        """
        q = (original or "").strip()
        if not q:
            return True

        word_count = len(q.split())
        has_arabic = cls._contains_arabic(q)
        has_english = cls._contains_english(q)
        mixed_language = has_arabic and has_english
        
        # certain complex markers likely indicate a more complex query that could benefit from LLM rewrite, even if short
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
        # if query contains strong indicators of complexity, skip LLM only if it's very short (<=4 words) to allow for simple questions but still catch complex ones
        if any(marker in low_q for marker in complex_markers):
            return False

        # empty / ultra-short queries should not hit LLM
        if word_count <= 2:
            return True

        # simple short queries: local path is enough
        if word_count <= 8 and not mixed_language:
            return True

        # if local rewrite changed only formatting/typos and query is modest size
        if word_count <= 12 and cls._is_small_change(q, local):
            return True

        # mixed-language queries can still skip if they are simple and local rewrite already handled term normalization
        if mixed_language and word_count <= 6 and cls._is_small_change(q, local):
            return True

        return False

    @staticmethod
    def _capitalize_first(text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:]

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

        # small difference tolerance
        return abs(len(o) - len(r)) <= 12

# ---------------------------------------------Ollama----------------------------------------------
# from __future__ import annotations

# import json
# from functools import lru_cache
# from typing import Dict, Any
# from django.conf import settings

# from core.services.llm.router import LLMRouter


# class GeminiQueryRewriter:
#     @classmethod
#     @lru_cache(maxsize=512)
#     def rewrite_cached(cls, user_query: str) -> Dict[str, Any]:
#         q = (user_query or "").strip()
#         if not q:
#             return {"rewritten_query": "", "notes": "empty"}

#         # Local fallback rewrite always available
#         local = cls._local_rewrite(q)


#         system_instruction = (
#             "You rewrite user queries to improve document retrieval.\n"
#             "Rules:\n"
#             "1) Keep the EXACT same meaning and intent.\n"
#             "2) Fix typos, spacing, capitalization, and minor grammar only when helpful.\n"
#             "3) Preserve person perspective exactly. Never change 'you' to 'I', 'your' to 'my', or names to pronouns.\n"
#             "4) Preserve the original language of the query. If the query is Arabic, keep it Arabic. If mixed, keep the same mixed style unless a tiny correction is needed.\n"
#             "5) Preserve proper nouns, acronyms, project names, and domain-specific terms exactly when possible.\n"
#             "6) Do NOT reinterpret the question into a different intent.\n"
#             "7) Do NOT replace specific terms with broad synonyms if that could hurt retrieval.\n"
#             "8) If the query is already clear, keep it very close to the original.\n"
#             "9) Return JSON only.\n"
#         )
#         prompt = (
#             "Return JSON with keys:\n"
#             "- rewritten_query: string\n"
#             "- notes: short string\n\n"
#             "Important:\n"
#             "- Keep the same person perspective.\n"
#             "- Keep the same language as the original query.\n"
#             "- Preserve project names and technical terms.\n"
#             "- Do not invent a new interpretation of a vague query.\n"
#             "- If the original query is already usable, make only minimal edits.\n\n"
#             f"User query: {q}\n"
#         )

#         schema = {
#             "type": "object",
#             "properties": {
#                 "rewritten_query": {"type": "string"},
#                 "notes": {"type": "string"},
#             },
#             "required": ["rewritten_query", "notes"],
#             "additionalProperties": False,
#         }

#         chain = [getattr(settings, "OLLAMA_REWRITE_PRIMARY", "gemma3:4b")] + \
#             getattr(settings, "OLLAMA_REWRITE_FALLBACKS", [])

#         ok, text, meta = LLMRouter.generate_json(
#             prompt=prompt,
#             system_instruction=system_instruction,
#             temperature=0.1,
#             model_chain=chain,
#             json_schema=schema,
#         )

#         if not ok:
#             return {
#                 "rewritten_query": local,
#                 "notes": f"rewrite_fallback:{meta.get('error')}",
#                 "meta": meta,
#             }

#         try:
#             data = json.loads(text)
#             rewritten = (data.get("rewritten_query") or "").strip() or local

#             return {
#                 "rewritten_query": rewritten,
#                 "notes": data.get("notes", "ok"),
#                 "meta": meta,
#             }
#         except Exception:
#             return {
#                 "rewritten_query": local,
#                 "notes": "rewrite_json_parse_failed",
#                 "meta": meta,
#             }

#     @staticmethod
#     def _local_rewrite(q: str) -> str:
#         return (
#             q.replace(" now ", " know ")
#             .replace(" certficate", " certificate")
#             .replace(" certifcate", " certificate")
#             .strip()
#         )

# --------------------------------------------Gemini----------------------------------------------

# from __future__ import annotations

# import json
# from functools import lru_cache
# from typing import Dict, Any
# from django.conf import settings

# from core.services.llm.gemini_router import GeminiRouter

# from core.services.llm.router import LLMRouter


# class GeminiQueryRewriter:
#     @classmethod
#     @lru_cache(maxsize=512)
#     def rewrite_cached(cls, user_query: str) -> Dict[str, Any]:
#         q = (user_query or "").strip()
#         if not q:
#             return {"rewritten_query": "", "notes": "empty"}

#         # local fallback rewrite always available
#         local = cls._local_rewrite(q)

#         # if no key, return local
#         if not getattr(settings, "GEMINI_API_KEY", None):
#             return {"rewritten_query": local, "notes": "no_gemini_key"}

#         system_instruction = (
#             "You rewrite user queries to improve document retrieval.\n"
#             "Rules:\n"
#             "1) Keep the SAME meaning/intent.\n"
#             "2) Fix typos.\n"
#             "3) Add relevant keywords/synonyms.\n"
#             "4) Return JSON only.\n"
#         )

#         prompt = (
#             "Return JSON with keys:\n"
#             "- rewritten_query: string\n"
#             "- notes: short string\n\n"
#             f"User query: {q}\n"
#         )

#         chain = [settings.GEMINI_REWRITE_PRIMARY] + \
#             getattr(settings, "GEMINI_REWRITE_FALLBACKS", [])
#         ok, text, meta = GeminiRouter.generate_json(
#             prompt=prompt,
#             system_instruction=system_instruction,
#             temperature=0.1,
#             model_chain=chain,
#         )

#         if not ok:
#             return {"rewritten_query": local, "notes": f"rewrite_fallback:{meta.get('error')}", "meta": meta}

#         try:
#             data = json.loads(text)
#             rewritten = (data.get("rewritten_query") or "").strip() or local
#             return {"rewritten_query": rewritten, "notes": data.get("notes", "ok"), "meta": meta}
#         except Exception:
#             return {"rewritten_query": local, "notes": "rewrite_json_parse_failed", "meta": meta}

#     @staticmethod
#     def _local_rewrite(q: str) -> str:
#         return (
#             q.replace(" now ", " know ")
#             .replace(" certficate", " certificate")
#             .replace(" certifcate", " certificate")
#             .strip()
#         )
