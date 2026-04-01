from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Dict, Any

from django.conf import settings
from core.services.llm.gemini_client import GeminiClient


@dataclass(frozen=True)
class SmartIntentResult:
    handled: bool
    intent: str  # greeting | goodbye | thanks | help | other
    reply: str
    confidence: float
    source: str  # heuristic | gemini


class SmartChatIntentService:
    """
    Smarter intent detection:
    - heuristic scoring for common conversational intents
    - Gemini fallback only when heuristic is uncertain

    Important behavior:
    - Generic assistant-help messages should be handled here.
    - Real questions about Samah/profile/capabilities should NOT be trapped as help.
    """

    _greet_hint = re.compile(
        r"\b(hi|hello|hey|morning|evening|afternoon|sup)\b",
        re.I,
    )
    _bye_hint = re.compile(
        r"\b(bye|goodbye|later|take care|see you)\b",
        re.I,
    )
    _thanks_hint = re.compile(
        r"\b(thanks|thank you|thx|appreciate)\b",
        re.I,
    )

    # Narrowed generic help intent
    _help_hint = re.compile(
        r"\b("
        r"what can you do|"
        r"how do you work|"
        r"what do you know|"
        r"what can i ask|"
        r"show me examples|"
        r"how can you help me"
        r")\b",
        re.I,
    )

    # If the question is clearly about Samah/profile/capabilities,
    # do not trap it as generic assistant help.
    _profile_question_hint = re.compile(
        r"\b("
        r"samah|she|her|build|project|projects|experience|skills|"
        r"language|framework|salary|payment|contact|email|phone|"
        r"freelance|remote|django|fastapi|backend|frontend|"
        r"chatbot|ai|llm|availability|work style|"
        r"favorite|favourite|prefer|preferred|impact|career|timeline"
        r")\b",
        re.I,
    )

    _arabic_greet_hint = re.compile(r"\b(السلام عليكم|مرحبا|هلا)\b", re.I)
    _arabic_bye_hint = re.compile(r"\b(مع السلامه|سلام)\b", re.I)
    _arabic_thanks_hint = re.compile(r"\b(شكرا|مشكور)\b", re.I)

    @classmethod
    def detect(cls, message: str) -> SmartIntentResult:
        msg = (message or "").strip()

        if not msg:
            return SmartIntentResult(
                handled=True,
                intent="help",
                reply="Say hi 👋 or ask me anything about Samah’s experience, projects, and skills.",
                confidence=0.95,
                source="heuristic",
            )

        h = cls._heuristic_classify(msg)

        # If confident and not a normal content question, handle here
        if h["confidence"] >= 0.75 and h["intent"] != "other":
            return SmartIntentResult(
                handled=True,
                intent=h["intent"],
                reply=cls._reply_for_intent(h["intent"]),
                confidence=h["confidence"],
                source="heuristic",
            )

        # Gemini fallback disabled for now
        if cls._should_use_gemini(msg, h):
            g = cls._gemini_classify(msg)

            if g["intent"] != "other":
                return SmartIntentResult(
                    handled=True,
                    intent=g["intent"],
                    reply=cls._reply_for_intent(g["intent"]),
                    confidence=g["confidence"],
                    source="gemini",
                )

        return SmartIntentResult(
            handled=False,
            intent="other",
            reply="",
            confidence=max(h["confidence"], 0.2),
            source="heuristic",
        )

    @classmethod
    def _heuristic_classify(cls, msg: str) -> Dict[str, Any]:
        text = msg.strip()
        low = text.lower()

        tokens = re.findall(r"\b\w+\b", low)
        token_count = len(tokens)
        char_count = len(text)
        has_emoji = bool(re.search(r"[\U0001F300-\U0001FAFF]", text))

        scores = {
            "greeting": 0.0,
            "goodbye": 0.0,
            "thanks": 0.0,
            "help": 0.0,
            "other": 0.0,
        }

        # Short messages are often greetings/thanks/goodbye
        if token_count <= 3 or char_count <= 15:
            scores["greeting"] += 0.15
            scores["thanks"] += 0.10
            scores["goodbye"] += 0.10

        if has_emoji:
            scores["greeting"] += 0.10
            scores["thanks"] += 0.05

        if cls._greet_hint.search(text) or cls._arabic_greet_hint.search(text):
            scores["greeting"] += 0.45

        if cls._bye_hint.search(text) or cls._arabic_bye_hint.search(text):
            scores["goodbye"] += 0.50

        if cls._thanks_hint.search(text) or cls._arabic_thanks_hint.search(text):
            scores["thanks"] += 0.50

        if cls._help_hint.search(text):
            scores["help"] += 0.55

        # If this looks like a real question about Samah/profile/capabilities,
        # reduce generic help score so retrieval can handle it.
        if cls._profile_question_hint.search(text):
            scores["help"] -= 0.35
            scores["other"] += 0.20

        # Longer question-like text is more likely to be a real content query
        if "?" in text and token_count > 4:
            scores["other"] += 0.25

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        confidence = min(0.95, 0.4 + max(best_score, 0.0))

        # If score is too weak, let retrieval handle it
        if best_score < 0.35:
            best_intent = "other"
            confidence = 0.45

        return {
            "intent": best_intent,
            "confidence": confidence,
            "scores": scores,
        }

    @classmethod
    def _should_use_gemini(cls, msg: str, heuristic: Dict[str, Any]) -> bool:
        return False

        # Re-enable later if needed:
        # low = msg.lower().strip()
        # tokens = re.findall(r"\b\w+\b", low)
        # if len(tokens) > 10:
        #     return False
        # if heuristic["intent"] == "other" and heuristic["confidence"] >= 0.6:
        #     return False
        # return bool(getattr(settings, "GEMINI_API_KEY", None))

    @classmethod
    def _gemini_classify(cls, msg: str) -> Dict[str, Any]:
        system_instruction = (
            "Classify the user's message into exactly one intent.\n"
            "Intents: greeting, goodbye, thanks, help, other.\n"
            "Return JSON only."
        )

        prompt = (
            "Return JSON like:\n"
            "{ \"intent\": \"greeting|goodbye|thanks|help|other\", \"confidence\": 0.0-1.0 }\n\n"
            f"Message: {msg}"
        )

        client = GeminiClient.client()
        resp = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        )

        try:
            data = json.loads(resp.text)
            intent = data.get("intent", "other")
            conf = float(data.get("confidence", 0.6))
            if intent not in {"greeting", "goodbye", "thanks", "help", "other"}:
                intent = "other"
            conf = max(0.0, min(1.0, conf))
            return {"intent": intent, "confidence": conf}
        except Exception:
            return {"intent": "other", "confidence": 0.5}

    @staticmethod
    def _reply_for_intent(intent: str) -> str:
        if intent == "greeting":
            return "Hi! 👋 Ask me anything about Samah — skills, projects, experience, or how to contact her."
        if intent == "goodbye":
            return "Bye! 👋 If you need anything else about Samah’s profile, I’m here."
        if intent == "thanks":
            return "You’re welcome! 😊 Want to know anything else about Samah?"
        if intent == "help":
            return (
                "I can answer questions about Samah using her uploaded CV and documents. "
                "Try: “What are her strongest technical skills?”, “Which projects used Django?”, or “How can I contact her?”"
            )
        return ""
