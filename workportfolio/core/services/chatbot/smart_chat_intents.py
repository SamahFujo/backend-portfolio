from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from django.conf import settings
import random
from core.services.llm.router import LLMRouter
import random

@dataclass(frozen=True)
class QuestionRouteResult:
    route: str
    confidence: float
    source: str  # deepseek | fallback
    raw_label: str = ""


@dataclass(frozen=True)
class ProfileAnswerModeResult:
    mode: str  # fact | overview
    confidence: float
    source: str  # heuristic | llm | fallback
    raw_label: str = ""


@dataclass(frozen=True)
class SmartIntentResult:
    handled: bool
    intent: str
    reply: str
    confidence: float
    source: str  # heuristic | llm


@dataclass(frozen=True)
class ConversationalResponseResult:
    handled: bool
    category: str
    reply: str
    confidence: float
    source: str


@dataclass(frozen=True)
class UiActionIntentResult:
    intent: str  # contact_capture | send_history_email | none
    confidence: float
    source: str  # local_gate | local_confident | llm | local_fallback
    reason: str = ""
    email: Optional[str] = None


class SmartChatIntentService:
    """
    Smart chat intent + question route service.

    Responsibilities:
    1) Detect quick conversational intents such as greeting/thanks/help/goodbye.
    2) Classify real user questions into an answer route using LLM-first routing.

    Design choice:
    - Quick intents remain lightweight and mostly heuristic.
    - Real route classification is LLM-first (DeepSeek), with a small local fallback.
    """
    _greet_hint = re.compile(
        r"\b("
        # basic greetings
        r"hi|hello|hey|heyy|heyy|hiya|hii|helo|helloo|hullo|yo|sup|wassup|whats up|what's up|"
        r"good morning|good afternoon|good evening|morning|afternoon|evening|"
        r"nice to meet you|pleasure to meet you|"
        # conversational openings
        r"start|let's start|lets start|can we start|shall we start|"
        r"are you there|you there|anyone there|"
        r"how are you|how're you|how are u|how r you|"
        r"how is it going|how's it going|hows it going|"
        r"how have you been|"
        r"hope you are well|hope you're well|hope u are well|"
        r"hope you are doing well|hope you're doing well|"
        r"good day|greetings"
        r")\b",
        re.I,
    )

    _bye_hint = re.compile(
        r"\b("
        # direct goodbyes
        r"bye|goodbye|bye bye|byee|see you|see ya|cya|later|laters|"
        r"take care|takecare|"
        r"good night|gn|night|have a good night|"
        r"farewell|"
        # sign-off style
        r"talk later|talk to you later|speak soon|catch you later|catch up later|"
        r"have a nice day|have a good day|have a great day|"
        r"have a nice evening|have a good evening|"
        r"see you later|see you soon|until next time|"
        r"i'm done|im done|that is all|that's all|thats all|"
        r"done for now|we are done|we're done|"
        r"thanks bye|ok bye|okay bye|alright bye"
        r")\b",
        re.I,
    )

    _thanks_hint = re.compile(
        r"\b("
        # common thanks
        r"thanks|thank you|thankyou|thx|ty|tys?m|thank u|thanks a lot|many thanks|"
        r"much appreciated|appreciated|really appreciate|i appreciate it|"
        r"appreciate that|appreciate your help|"
        # stronger gratitude
        r"thank you so much|thanks so much|thank you very much|"
        r"big thanks|huge thanks|"
        r"thanks for your help|thank you for your help|"
        r"thanks for explaining|thanks for the help|"
        # positive acknowledgements
        r"perfect thanks|great thanks|awesome thanks|"
        r"got it thanks|understood thanks|"
        r"that helps|this helps|that was helpful|this was helpful|"
        r"super helpful|very helpful"
        r")\b",
        re.I,
    )

    _help_hint = re.compile(
        r"\b("
        # capability questions
        r"what can you do|what do you do|how do you work|how can you help me|"
        r"what do you know|what can i ask|what can i ask you|"
        r"what can you help with|what can you help me with|"
        r"how can i use you|how should i use you|"
        r"what are you able to do|what are your capabilities|"
        r"what can this chatbot do|what can the chatbot do|"
        # example requests
        r"show me examples|give me examples|example questions|sample questions|"
        r"what should i ask|what do people ask you|"
        r"give me ideas|give me prompts|prompt ideas|"
        # help requests
        r"help|i need help|can you help|could you help|please help|"
        r"assist me|i need assistance|support me|"
        r"guide me|can you guide me|walk me through|"
        # usage / explanation
        r"how does this work|how does it work|how are you working|"
        r"how do i ask you|how do i start|"
        r"what kind of questions can you answer|"
        r"what information do you have|what information do you know|"
        r")\b",
        re.I,
    )

    _profile_question_hint = re.compile(
        r"\b("
        # person reference
        r"samah|she|her|hers|you|your|yourself|candidate|profile|background|about her|about samah|"
        # work / career
        r"build|built|develop|developed|create|created|make|made|"
        r"project|projects|portfolio|case study|case studies|"
        r"experience|work experience|professional experience|employment|career|career path|timeline|journey|"
        r"role|roles|position|positions|responsibility|responsibilities|"
        r"team lead|leadership|management|managed|client|stakeholder|delivery|ownership|"
        # skills / tech
        r"skills|skillset|technical skills|tech stack|stack|tools|technology|technologies|"
        r"language|languages|programming|framework|frameworks|library|libraries|"
        r"python|javascript|typescript|react|nextjs|next\.js|django|drf|fastapi|flask|node|nodejs|"
        r"backend|frontend|full stack|fullstack|api|rest api|database|postgresql|mysql|mongodb|oracle|sql server|"
        r"docker|nginx|gunicorn|postman|azure|aws|"
        # ai / ml
        r"ai|ml|machine learning|deep learning|nlp|llm|llms|rag|transformers|"
        r"bert|roberta|t5|langchain|langfuse|ollama|gemini|openwebui|prompt engineering|"
        r"embedding|embeddings|vector|reranker|classification|ocr|tesseract|easyocr|opencv|"
        r"chatbot|agent|agents|workflow automation|"
        # project-fit style
        r"can she build|can samah build|can she develop|can samah develop|"
        r"can she do|can samah do|is she fit|is samah fit|"
        r"is she suitable|is samah suitable|"
        r"can she help|can samah help|"
        # personal preferences / working style
        r"availability|available|notice period|open to work|"
        r"freelance|remote|onsite|hybrid|relocation|work style|working style|"
        r"preferred|prefer|favorite|favourite|interest|interested|"
        # contact / hiring
        r"contact|email|mail|phone|mobile|linkedin|github|portfolio|website|"
        r"salary|expected salary|current salary|payment|rate|budget|hourly|monthly|"
        r"hire|hiring|recruit|recruiter|interview|cv|resume|"
        # achievements / proof
        r"certificate|certificates|certification|certifications|"
        r"award|awards|achievement|achievements|"
        r"impact|result|results|outcome|outcomes|success|"
        # domain knowledge
        r"dashboard|unspsc|procurement|property chatbot|electricity|payroll|social support"
        r")\b",
        re.I,
    )

    _ack_hint = re.compile(
        r"^\s*("
        r"ok|okay|okk|k|kk|"
        r"got it|got you|understood|noted|alright|all right|"
        r"sure|fine|cool|perfect|great|sounds good|makes sense|"
        r"تمام|اوكي|أوكي|ماشي|زين|حسنا|حسنًا"
        r")\s*[.!؟?]*\s*$",
        re.I,
    )

    _pause_hint = re.compile(
        r"^\s*("
        r"wait|wait wait|hold on|one sec|one second|give me a second|"
        r"give me a moment|moment|pause|stop|let me check|"
        r"انتظر|لحظة|ثواني|وقف|توقف"
        r")\s*[.!؟?]*\s*$",
        re.I,
    )

    _positive_reaction_hint = re.compile(
        r"^\s*("
        r"that'?s good|that is good|this is good|"
        r"that'?s great|that is great|this is great|"
        r"nice|very nice|good|great|cool|awesome|amazing|excellent|perfect|"
        r"i like it|love it|looks good|sounds good|"
        r"جميل|ممتاز|حلو|وايد زين|زين|تمام"
        r")\s*[.!؟?]*\s*$",
        re.I,
    )

    _arabic_greet_hint = re.compile(r"\b(السلام عليكم|مرحبا|هلا)\b", re.I)
    _arabic_bye_hint = re.compile(r"\b(مع السلامه|سلام)\b", re.I)
    _arabic_thanks_hint = re.compile(r"\b(شكرا|مشكور)\b", re.I)

    ALLOWED_INTENTS = {
        "greeting",
        "goodbye",
        "thanks",
        "help",
        "acknowledgement",
        "positive_reaction",
        "pause",
        "other",
    }

    ALLOWED_ROUTES = {
        "identity_question",
        "session_memory_question",
        "profile_docs_question",
        "capability_inference_question",
        "general_question",
    }

    ALLOWED_UI_ACTION_INTENTS = {
        "contact_capture",
        "send_history_email",
        "none",
    }

    ALLOWED_CONVERSATIONAL_CATEGORIES = {
        "none",
        "help",
        "bot_behavior_concern",
        "off_topic_request",
        "small_talk",
        "scope_question",
    }

    @staticmethod
    def _extract_email_from_text(message: str) -> Optional[str]:
        """
        Extract an email address if the user already typed it.
        Example:
        - "send this chat to me at test@example.com"
        """
        text = (message or "").strip()

        match = re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            text,
        )

        return match.group(0) if match else None

    @staticmethod
    def _may_need_ui_action_classifier(message: str) -> bool:
        """
        Cheap pre-check only.

        This does not make the final decision.
        It only decides whether the message is close enough to:
        - contact capture
        - send conversation history by email
        """
        q = (message or "").strip().lower()

        if not q:
            return False

        if SmartChatIntentService._looks_like_session_memory_question(q):
            if not any(term in q for term in ["send", "email", "mail", "forward", "export"]):
                return False

        # These are normal profile/contact-info questions.
        # They should continue to the normal QA/contact extractor.
        normal_contact_info_patterns = [
            "how can i contact samah",
            "how can i contact her",
            "how do i contact samah",
            "how do i contact her",
            "what is samah's email",
            "what is her email",
            "samah email",
            "samah's email",
            "her email",
            "her phone",
            "her linkedin",
            "her github",
            "contact details of samah",
            "samah contact details",
            "send me samah's email",
            "send samah email",
        ]

        if any(pattern in q for pattern in normal_contact_info_patterns):
            return False

        ui_trigger_terms = [
            # visitor wants Samah to contact them
            "contact me",
            "call me",
            "reach me",
            "get back to me",
            "follow up with me",
            "samah contact me",
            "samah can contact me",
            "samah to contact me",
            "samah call me",
            "samah to call me",
            "samah reach me",
            "leave my",
            "my details",
            "my contact",
            "my phone",
            "my mobile",
            "my number",
            "my email",
            "share my",
            "provide my",
            "submit my",
            "fill my",

            "send me the conversation",
            "send the conversation",
            "email the conversation",
            "email me the conversation",
            "send this conversation",

            # visitor wants chat history sent/exported
            "send this chat",
            "send me this chat",
            "send chat",
            "send the chat",
            "chat history",
            "conversation history",
            "this conversation",
            "our conversation",
            "chat transcript",
            "conversation transcript",
            "transcript",
            "email this chat",
            "email me this",
            "forward this",
            "export this",
        ]

        return any(term in q for term in ui_trigger_terms)

    @classmethod
    def _classify_ui_action_intent_local(cls, message: str) -> UiActionIntentResult:
        """
        Local fallback + confident local classifier.

        This is cheap and prevents unnecessary LLM calls.
        """
        q = (message or "").strip().lower()
        extracted_email = cls._extract_email_from_text(message)

        if not q:
            return UiActionIntentResult(
                intent="none",
                confidence=0.0,
                source="local_fallback",
                reason="Empty message.",
                email=extracted_email,
            )

        # Strong negative cases: user asks for Samah's contact information.
        normal_contact_questions = [
            "how can i contact samah",
            "how can i contact her",
            "how do i contact samah",
            "how do i contact her",
            "what is samah's email",
            "what is her email",
            "samah email",
            "samah's email",
            "her email",
            "her phone",
            "her linkedin",
            "her github",
            "contact details of samah",
            "samah contact details",
            "send me samah's email",
        ]

        if any(term in q for term in normal_contact_questions):
            return UiActionIntentResult(
                intent="none",
                confidence=0.95,
                source="local_confident",
                reason="User is asking for Samah's contact information, not asking to share their own details.",
                email=extracted_email,
            )

        history_terms = [
            "chat history",
            "conversation history",
            "this conversation",
            "our conversation",
            "current conversation",
            "the conversation",
            "conversation",
            "this chat",
            "the chat",
            "chat",
            "chat transcript",
            "conversation transcript",
            "transcript",
        ]

        send_terms = [
            "send",
            "email",
            "mail",
            "forward",
            "share",
            "export",
        ]

        contact_capture_terms = [
            "my details",
            "my contact",
            "my phone",
            "my mobile",
            "my number",
            "my email",
            "leave my details",
            "leave my contact",
            "share my details",
            "share my contact",
            "provide my details",
            "provide my contact",
            "submit my details",
            "fill my details",
            "samah contact me",
            "samah can contact me",
            "samah to contact me",
            "samah call me",
            "samah to call me",
            "call me",
            "contact me",
            "get back to me",
            "follow up with me",
        ]

        has_history = any(term in q for term in history_terms)
        has_send = any(term in q for term in send_terms)

        # Give send-history priority if transcript/history/chat is clearly mentioned.
        if has_history and has_send:
            return UiActionIntentResult(
                intent="send_history_email",
                confidence=0.88,
                source="local_confident",
                reason="User clearly wants the chat/conversation transcript sent.",
                email=extracted_email,
            )

        if any(term in q for term in contact_capture_terms):
            return UiActionIntentResult(
                intent="contact_capture",
                confidence=0.84,
                source="local_confident",
                reason="User wants to share their own details or wants Samah to contact them.",
                email=extracted_email,
            )

        return UiActionIntentResult(
            intent="none",
            confidence=0.45,
            source="local_fallback",
            reason="No clear UI action detected locally.",
            email=extracted_email,
        )

    @classmethod
    def classify_ui_action_intent(
        cls,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> UiActionIntentResult:
        """
        Cost-controlled UI-action classifier.

        It returns one of:
        - contact_capture
        - send_history_email
        - none

        LLM is only called for ambiguous UI-action-like messages.
        """
        msg = (message or "").strip()
        history = history or []

        if not msg:
            return UiActionIntentResult(
                intent="none",
                confidence=0.0,
                source="local_gate",
                reason="Empty message.",
                email=None,
            )

        local_result = cls._classify_ui_action_intent_local(msg)

        # If local result is already confident, do not call LLM.
        if local_result.intent in {"contact_capture", "send_history_email"} and local_result.confidence >= 0.82:
            return local_result

        # If this message does not look like a UI action, skip LLM completely.
        if not cls._may_need_ui_action_classifier(msg):
            return UiActionIntentResult(
                intent="none",
                confidence=0.99,
                source="local_gate",
                reason="Message does not look like a UI-action request.",
                email=local_result.email,
            )

        # Only ambiguous UI-action-like cases reach the LLM.
        llm_result = cls._llm_classify_ui_action_intent(
            msg=msg,
            history=history,
        )

        if llm_result:
            return llm_result

        # Last fallback.
        return UiActionIntentResult(
            intent=local_result.intent,
            confidence=local_result.confidence,
            source="local_fallback",
            reason=local_result.reason,
            email=local_result.email,
        )

    @classmethod
    def _llm_classify_ui_action_intent(
        cls,
        msg: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[UiActionIntentResult]:
        """
        Short LLM classifier for ambiguous UI-action requests only.
        """
        history = history or []

        compact_history = []
        for item in history[-4:]:
            role = (item.get("role") or "").strip()
            content = (item.get("content") or "").strip()
            if role and content:
                compact_history.append({
                    "role": role,
                    "content": content[:300],
                })

        system_instruction = (
            "Classify whether a portfolio chatbot should trigger a frontend UI action.\n"
            "Allowed intents: contact_capture, send_history_email, none.\n\n"
            "contact_capture: visitor wants to share their own details or wants Samah to contact/call/email them.\n"
            "send_history_email: visitor wants this chat/conversation/history/transcript emailed, sent, exported, or forwarded.\n"
            "none: normal questions, greetings, thanks, asking for Samah's contact info, or conversation-memory questions.\n\n"
            "Rules:\n"
            "- 'How can I contact Samah?' or 'What is Samah email?' = none.\n"
            "- 'Can Samah contact/call/email me?' = contact_capture.\n"
            "- 'Send/email/export this chat/conversation/transcript' = send_history_email.\n"
            "- 'What did I ask before?' = none.\n"
            "- Extract email only if the user typed one.\n"
            "- Return JSON only."
        )

        prompt = (
            "Return JSON exactly like this:\n"
            "{"
            "\"intent\":\"contact_capture|send_history_email|none\","
            "\"confidence\":0.0,"
            "\"reason\":\"short explanation\","
            "\"email\":null"
            "}\n\n"
            f"Recent history: {json.dumps(compact_history, ensure_ascii=False)}\n"
            f"User message: {msg}"
        )

        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "contact_capture",
                        "send_history_email",
                        "none",
                    ],
                },
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
                "email": {"type": ["string", "null"]},
            },
            "required": ["intent", "confidence", "reason", "email"],
            "additionalProperties": False,
        }

        chain = getattr(
            settings,
            "UI_ACTION_INTENT_MODEL_CHAIN",
            getattr(
                settings,
                "QUICK_INTENT_MODEL_CHAIN",
                [
                    getattr(settings, "INTENT_PRIMARY_MODEL", "deepseek-chat"),
                ],
            ),
        )

        try:
            ok, text, meta = LLMRouter.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.0,
                model_chain=chain,
                json_schema=schema,
                task=LLMRouter.TASK_INTENT,
            )
        except Exception:
            return None

        if not ok:
            return None

        try:
            data = json.loads(text)

            intent = data.get("intent", "none")
            confidence = float(data.get("confidence", 0.60))
            reason = data.get("reason", "")
            email = data.get("email")

            if intent not in cls.ALLOWED_UI_ACTION_INTENTS:
                intent = "none"

            confidence = max(0.0, min(1.0, confidence))

            # Safety threshold: do not trigger UI action if uncertain.
            if confidence < 0.72:
                return UiActionIntentResult(
                    intent="none",
                    confidence=confidence,
                    source="llm",
                    reason=reason or "LLM confidence below threshold.",
                    email=email,
                )

            return UiActionIntentResult(
                intent=intent,
                confidence=confidence,
                source="llm",
                reason=reason,
                email=email,
            )

        except Exception:
            return None
        
        
    @classmethod
    def detect_conversational_response(
        cls,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> ConversationalResponseResult:
        """
        Detect messages that should receive a conversational LLM answer
        instead of going to document retrieval.

        Examples:
        - are you dumping answers?
        - tell me a joke
        - why is your answer wrong?
        - what can you do?
        """
        msg = (message or "").strip()
        history = history or []

        if not msg:
            return ConversationalResponseResult(
                handled=True,
                category="help",
                reply="",
                confidence=0.95,
                source="local_classifier",
            )

        local_result = cls._detect_conversational_response_local(msg)

        if local_result.handled and local_result.confidence >= 0.82:
            return local_result

        llm_result = cls._llm_detect_conversational_response(
            message=msg,
            history=history,
        )

        if llm_result and llm_result.handled and llm_result.confidence >= 0.75:
            return llm_result

        return ConversationalResponseResult(
            handled=False,
            category="none",
            reply="",
            confidence=0.0,
            source="fallback",
        )
        
        
    @classmethod
    def _llm_detect_conversational_response(
        cls,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ConversationalResponseResult]:
        """
        LLM classifier for flexible conversational/meta/off-topic messages.
        This decides whether the message should skip retrieval.
        """
        history = history or []

        compact_history = []
        for item in history[-4:]:
            role = (item.get("role") or "").strip()
            content = (item.get("content") or "").strip()
            if role and content:
                compact_history.append({
                    "role": role,
                    "content": content[:300],
                })

        system_instruction = (
            "You are a classifier for Samah.ai's portfolio chatbot.\n"
            "Decide if the user's message should be answered conversationally instead of using document retrieval.\n\n"
            "Handle conversationally when the message is about:\n"
            "- what the chatbot can do\n"
            "- how the chatbot works\n"
            "- complaints or concerns about answer quality\n"
            "- whether answers are random, dumped, copied, or hallucinated\n"
            "- off-topic requests like jokes, games, poems, singing, or entertainment\n"
            "- small talk that is not asking about Samah's professional documents\n\n"
            "Do NOT handle conversationally if the user asks about Samah's experience, skills, projects, contact details, availability, salary, certifications, or suitability.\n"
            "Those must continue to retrieval.\n\n"
            "Return JSON only."
        )

        prompt = (
            "Return JSON exactly like this:\n"
            "{"
            "\"handled\":true,"
            "\"category\":\"help|bot_behavior_concern|off_topic_request|small_talk|scope_question|none\","
            "\"confidence\":0.0"
            "}\n\n"
            f"Recent history: {json.dumps(compact_history, ensure_ascii=False)}\n"
            f"User message: {message}"
        )

        schema = {
            "type": "object",
            "properties": {
                "handled": {"type": "boolean"},
                "category": {
                    "type": "string",
                    "enum": [
                        "help",
                        "bot_behavior_concern",
                        "off_topic_request",
                        "small_talk",
                        "scope_question",
                        "none",
                    ],
                },
                "confidence": {"type": "number"},
            },
            "required": ["handled", "category", "confidence"],
            "additionalProperties": False,
        }

        chain = getattr(
            settings,
            "CONVERSATIONAL_RESPONSE_MODEL_CHAIN",
            getattr(
                settings,
                "QUICK_INTENT_MODEL_CHAIN",
                [
                    getattr(settings, "INTENT_PRIMARY_MODEL", "deepseek-chat"),
                ],
            ),
        )

        try:
            ok, text, meta = LLMRouter.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.0,
                model_chain=chain,
                json_schema=schema,
                task=LLMRouter.TASK_INTENT,
            )

            if not ok:
                return None

            data = json.loads(text)

            handled = bool(data.get("handled", False))
            category = data.get("category", "none")
            confidence = float(data.get("confidence", 0.0))

            if category not in cls.ALLOWED_CONVERSATIONAL_CATEGORIES:
                category = "none"

            confidence = max(0.0, min(1.0, confidence))

            return ConversationalResponseResult(
                handled=handled and category != "none",
                category=category,
                reply="",
                confidence=confidence,
                source="llm_classifier",
            )

        except Exception:
            return None
        
    @classmethod
    def generate_conversational_reply(
        cls,
        message: str,
        category: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate the final natural answer using the LLM.
        Fallback is used only if the LLM/API fails.
        """
        history = history or []

        compact_history = []
        for item in history[-4:]:
            role = (item.get("role") or "").strip()
            content = (item.get("content") or "").strip()
            if role and content:
                compact_history.append({
                    "role": role,
                    "content": content[:300],
                })

        system_instruction = (
            "You are Samah.ai's professional portfolio chatbot.\n"
            "Generate a warm, intelligent, concise answer to the user.\n\n"
            "The chatbot is designed to answer questions about:\n"
            "- Samah's professional experience\n"
            "- AI/ML, LLM, RAG, Django, Next.js, React, backend/frontend projects\n"
            "- portfolio projects, certifications, availability, hiring, and contact questions\n\n"
            "Rules:\n"
            "- Do not sound like a hardcoded fallback.\n"
            "- Do not mention document retrieval.\n"
            "- Do not say 'based on the evidence' for chatbot behavior questions.\n"
            "- If the user is concerned about answer quality, reassure them politely.\n"
            "- If the user asks for unrelated content, politely redirect to Samah's professional profile.\n"
            "- If the user asks what the chatbot can do, explain naturally with examples.\n"
            "- Keep the answer 2 to 4 sentences.\n"
            "- End with one helpful suggestion related to Samah.\n"
            "- Return JSON only."
        )

        prompt = (
            "Generate the final user-facing chatbot reply.\n\n"
            "Return JSON exactly like this:\n"
            "{"
            "\"reply\":\"final answer\""
            "}\n\n"
            f"Category: {category}\n"
            f"Recent history: {json.dumps(compact_history, ensure_ascii=False)}\n"
            f"User message: {message}"
        )

        schema = {
            "type": "object",
            "properties": {
                "reply": {"type": "string"},
            },
            "required": ["reply"],
            "additionalProperties": False,
        }

        chain = getattr(
            settings,
            "CONVERSATIONAL_REPLY_MODEL_CHAIN",
            getattr(
                settings,
                "QUICK_INTENT_MODEL_CHAIN",
                [
                    getattr(settings, "INTENT_PRIMARY_MODEL", "deepseek-chat"),
                ],
            ),
        )

        try:
            ok, text, meta = LLMRouter.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.45,
                model_chain=chain,
                json_schema=schema,
                task=LLMRouter.TASK_INTENT,
            )

            if not ok:
                return cls._emergency_conversational_reply(category)

            data = json.loads(text)
            reply = (data.get("reply") or "").strip()

            if not reply:
                return cls._emergency_conversational_reply(category)

            return reply

        except Exception:
            return cls._emergency_conversational_reply(category)
        
        
    @staticmethod
    def _emergency_conversational_reply(category: str) -> str:
        """
        Used only when the LLM/API fails.
        """
        if category == "bot_behavior_concern":
            return (
                "I understand your concern. I’m designed to answer in a focused way about Samah’s professional profile, not to provide random responses. "
                "You can ask me about her projects, skills, experience, or contact details."
            )

        if category == "off_topic_request":
            return (
                "I’m mainly focused on Samah’s professional profile, so I may not be the best place for unrelated requests. "
                "You can ask me about her AI projects, Django/Next.js experience, or portfolio work."
            )

        return (
            "I can help you explore Samah’s professional background, projects, skills, certifications, availability, and contact details."
        )
        
    @classmethod
    def _detect_conversational_response_local(
        cls,
        message: str,
    ) -> ConversationalResponseResult:
        """
        Local classifier only. It should NOT generate the final user answer.
        Final answer will be generated by the LLM.
        """
        q = (message or "").strip().lower()

        bot_behavior_terms = [
            "dumping answers",
            "dump answers",
            "random answers",
            "are you random",
            "wrong answer",
            "bad answer",
            "why your answer",
            "why is your answer",
            "are you hallucinating",
            "do you hallucinate",
            "copy paste",
            "copying answers",
        ]

        off_topic_terms = [
            "tell me a joke",
            "joke",
            "make me laugh",
            "play a game",
            "sing",
            "write a poem",
        ]

        help_terms = [
            "what can you do",
            "how can you help",
            "how do you work",
            "what should i ask",
            "give me examples",
            "what can i ask",
        ]

        if any(term in q for term in bot_behavior_terms):
            return ConversationalResponseResult(
                handled=True,
                category="bot_behavior_concern",
                reply="",
                confidence=0.90,
                source="local_classifier",
            )

        if any(term in q for term in off_topic_terms):
            return ConversationalResponseResult(
                handled=True,
                category="off_topic_request",
                reply="",
                confidence=0.88,
                source="local_classifier",
            )

        if any(term in q for term in help_terms):
            return ConversationalResponseResult(
                handled=True,
                category="help",
                reply="",
                confidence=0.86,
                source="local_classifier",
            )

        return ConversationalResponseResult(
            handled=False,
            category="none",
            reply="",
            confidence=0.0,
            source="local_classifier",
        )

    @staticmethod
    def _looks_like_session_memory_question(msg: str) -> bool:
        """
        Detect questions about earlier messages in the current conversation/session.
        This is intentionally broader than the old fallback markers so phrases like
        'what was my second question' are caught before generic help/general routing.
        """
        low = (msg or "").strip().lower()

        direct_markers = [
            "first question", "second question", "third question",
            "last question", "previous question",
            "first answer", "second answer", "third answer",
            "last answer", "previous answer",
            "what did i ask", "what did you say",
            "show my last", "summarize what we discussed",
            "what did we discuss", "in this session", "in this chat",
        ]

        if any(marker in low for marker in direct_markers):
            return True

        topic_memory_patterns = [
            r"\bwhat was my question about\b",
            r"\bwhat did i ask about\b",
            r"\bwhat was i asking about\b",
            r"\bwhat did we discuss about\b",
            r"\bwhat did you say about\b",
            r"\bwhat was your answer about\b",
            r"\bremind me what i asked about\b",
            r"\bremind me what we discussed about\b",
            r"\bsummarize what we discussed about\b",
            r"\brecap what we discussed about\b",
        ]

        if any(re.search(pattern, low) for pattern in topic_memory_patterns):
            return True

        ordinal_words = [
            "first", "second", "third", "fourth", "fifth",
            "sixth", "seventh", "eighth", "ninth", "tenth",
        ]
        target_words = ["question", "questions",
                        "answer", "answers", "message", "messages"]

        if any(word in low for word in ordinal_words) and any(word in low for word in target_words):
            return True

        patterns = [
            r"\bwhat was my \d+(st|nd|rd|th)? question\b",
            r"\bwhat was your \d+(st|nd|rd|th)? answer\b",
            r"\bshow my last \d+ questions\b",
            r"\bshow your last \d+ answers\b",
            r"\bwhat did i ask before\b",
            r"\bwhat did you say before\b",
            r"\bwhat did i ask earlier\b",
            r"\bwhat did you answer earlier\b",
            r"\bwhat was the previous (question|answer|message)\b",
        ]
        return any(re.search(pattern, low) for pattern in patterns)

    @classmethod
    def detect(cls, message: str) -> SmartIntentResult:
        """
        Quick conversational detection only.

        This should remain lightweight so we avoid unnecessary LLM calls
        for obvious greetings/thanks/help/goodbye.
        """
        msg = (message or "").strip()

        if not msg:
            return SmartIntentResult(
                handled=True,
                intent="help",
                reply="Say hi 👋 or ask me anything about Samah’s experience, projects, and skills.",
                confidence=0.95,
                source="heuristic",
            )

        if cls._looks_like_cost_question(msg):
            return SmartIntentResult(
                handled=False,
                intent="other",
                reply="",
                confidence=0.85,
                source="heuristic",
            )

        h = cls._heuristic_classify(msg)

        if h["confidence"] >= 0.75 and h["intent"] != "other":
            return SmartIntentResult(
                handled=True,
                intent=h["intent"],
                reply=cls._reply_for_intent(h["intent"]),
                confidence=h["confidence"],
                source="heuristic",
            )

        if cls._should_use_llm_for_quick_intent(msg, h):
            g = cls._llm_classify_quick_intent(msg)

            if g["intent"] != "other":
                return SmartIntentResult(
                    handled=True,
                    intent=g["intent"],
                    reply=cls._reply_for_intent(g["intent"]),
                    confidence=g["confidence"],
                    source="llm",
                )

        return SmartIntentResult(
            handled=False,
            intent="other",
            reply="",
            confidence=max(h["confidence"], 0.2),
            source="heuristic",
        )

    @dataclass(frozen=True)
    class UiActionIntentResult:
        intent: str  # contact_capture | send_history_email | none
        confidence: float
        source: str  # local_gate | local_confident | llm | local_fallback
        reason: str = ""
        email: Optional[str] = None

    @classmethod
    def classify_question_route(
        cls,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> QuestionRouteResult:
        """
        LLM-first route classification for real questions.

        Routes:
        - identity_question
        - session_memory_question
        - profile_docs_question
        - capability_inference_question
        - general_question
        """
        msg = (message or "").strip()
        history = history or []

        if not msg:
            return QuestionRouteResult(
                route="general_question",
                confidence=0.50,
                source="fallback",
                raw_label="empty_message",
            )

        # Strong early routing for session-memory questions
        if cls._looks_like_session_memory_question(msg):
            return QuestionRouteResult(
                route="session_memory_question",
                confidence=0.95,
                source="fallback",
                raw_label="explicit_session_memory_marker",
            )

        if cls._looks_like_profile_overview_question(msg):
            return QuestionRouteResult(
                route="profile_docs_question",
                confidence=0.92,
                source="fallback",
                raw_label="explicit_profile_overview_marker",
            )

        llm_result = cls._llm_classify_route(msg, history=history)
        if llm_result:
            return llm_result

        return cls._fallback_route(msg)

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
            "acknowledgement": 0.0,
            "pause": 0.0,
            "positive_reaction": 0.0,
            "other": 0.0,

        }

        if token_count <= 3 or char_count <= 15:
            scores["greeting"] += 0.15
            scores["thanks"] += 0.10
            scores["goodbye"] += 0.10
            scores["acknowledgement"] += 0.10
            scores["positive_reaction"] += 0.10
            scores["pause"] += 0.10

        if has_emoji:
            scores["greeting"] += 0.10
            scores["thanks"] += 0.05

        if cls._greet_hint.search(text) or cls._arabic_greet_hint.search(text):
            scores["greeting"] += 0.45

        if cls._bye_hint.search(text) or cls._arabic_bye_hint.search(text):
            scores["goodbye"] += 0.50

        if cls._thanks_hint.search(text) or cls._arabic_thanks_hint.search(text):
            scores["thanks"] += 0.50

        if cls._positive_reaction_hint.search(text):
            scores["positive_reaction"] += 0.70

        if cls._help_hint.search(text):
            scores["help"] += 0.55

        if cls._ack_hint.search(text):
            scores["acknowledgement"] += 0.60

        if cls._pause_hint.search(text):
            scores["pause"] += 0.65

        if cls._profile_question_hint.search(text):
            scores["help"] -= 0.35
            scores["other"] += 0.20

        if "?" in text and token_count > 4:
            scores["other"] += 0.25

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        confidence = min(0.95, 0.4 + max(best_score, 0.0))

        if best_score < 0.35:
            best_intent = "other"
            confidence = 0.45

        return {
            "intent": best_intent,
            "confidence": confidence,
            "scores": scores,
        }

    @classmethod
    def _should_use_llm_for_quick_intent(cls, msg: str, heuristic: Dict[str, Any]) -> bool:
        tokens = re.findall(r"\b\w+\b", msg.lower().strip())
        token_count = len(tokens)

        # Do not waste LLM calls on long real-content questions.
        if token_count > 10:
            return False

        # If heuristic is already confident, no need for LLM.
        if heuristic["confidence"] >= 0.75 and heuristic["intent"] != "other":
            return False

        # Short ambiguous messages are exactly where LLM helps:
        # "ok", "got it", "wait", "sure", "continue", "what?", etc.
        if token_count <= 5:
            return True

        # Medium short messages can also be classified if heuristic is uncertain.
        if heuristic["confidence"] < 0.70:
            return True

        return False

    @classmethod
    def _llm_classify_quick_intent(cls, msg: str) -> Dict[str, Any]:
        """
        Optional LLM support for quick intent classification.
        This still uses your router, but does not assume a specific provider.
        """
        system_instruction = (
            "Classify the user's message into exactly one quick chat intent for a portfolio chatbot.\n"
            "Allowed intents: greeting, goodbye, thanks, help, acknowledgement, pause, other.\n\n"
            "Definitions:\n"
            "- greeting: the user is opening the conversation or saying hello. This includes multilingual greetings like 'ciao', 'chao', 'hola', 'bonjour', 'salam', and 'marhaba'.\n"
            "- goodbye: the user is ending the conversation.\n"
            "- thanks: the user is expressing gratitude.\n"
            "- help: the user asks what the chatbot can do, how it works, or asks for example questions.\n"
            "- acknowledgement: the user only confirms, accepts, or acknowledges, such as 'ok', 'got it', 'sounds good', 'understood'.\n"
            "- pause: the user asks to wait, pause, stop briefly, or give them a moment.\n"
            "- other: real questions, profile questions, technical questions, or anything that needs normal routing.\n\n"
            "Important rules:\n"
            "- Do not classify profile questions as help.\n"
            "- Do not classify acknowledgements like 'ok' or 'got it' as help.\n"
            "- Do not classify pause messages like 'wait wait' as help.\n"
            "- If the user sends only 'ciao', 'chao', 'hola', 'salam', or 'marhaba' at the start of a conversation, classify it as greeting.\n"
            "- Return JSON only."
        )

        prompt = (
            "Return JSON exactly like this:\n"
            "{"
            "\"intent\":\"greeting|goodbye|thanks|help|acknowledgement|pause|other\","
            "\"confidence\":0.0"
            "}\n\n"
            f"Message: {msg}"
        )

        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "greeting",
                        "goodbye",
                        "thanks",
                        "help",
                        "acknowledgement",
                        "pause",
                        "other",
                    ],
                },
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        }

        chain = getattr(
            settings,
            "QUICK_INTENT_MODEL_CHAIN",
            [
                getattr(settings, "INTENT_PRIMARY_MODEL", "deepseek-chat"),
            ],
        )

        ok, text, meta = LLMRouter.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.0,
            model_chain=chain,
            json_schema=schema,
            task=LLMRouter.TASK_INTENT,
        )

        if not ok:
            return {"intent": "other", "confidence": 0.5}

        try:
            data = json.loads(text)
            intent = data.get("intent", "other")
            conf = float(data.get("confidence", 0.6))

            if intent not in cls.ALLOWED_INTENTS:
                intent = "other"

            conf = max(0.0, min(1.0, conf))
            return {"intent": intent, "confidence": conf}
        except Exception:
            return {"intent": "other", "confidence": 0.5}

    @staticmethod
    def _looks_like_work_history_fact_question(msg: str) -> bool:
        low = (msg or "").strip().lower()
        markers = [
            "how long did she work",
            "how long has she worked",
            "what is her total experience",
            "worked at",
            "experience at",
            "work at",
            "employment at",
            "duration at",

        ]
        return any(marker in low for marker in markers)

    @staticmethod
    def _looks_like_contact_question(msg: str) -> bool:
        low = (msg or "").strip().lower()
        markers = [
            "how can i contact her",
            "how do i contact her",
            "contact her",
            "email her",
            "phone number",
            "contact details",
        ]
        return any(marker in low for marker in markers)

    @staticmethod
    def _looks_like_capability_help_question(msg: str) -> bool:
        low = (msg or "").strip().lower()
        markers = [
            "how can she help",
            "how can samah help",
            "how samah can help",
            "how she can help",
            "help me with",
            "help with",
            "assist with",
            "support me with",
        ]
        return any(marker in low for marker in markers)

    @staticmethod
    def _looks_like_availability_question(msg: str) -> bool:
        low = (msg or "").strip().lower()
        markers = [
            "freelance",
            "freelancer",
            "contract work",
            "contract-based",
            "consulting",
            "project-based",
            "open to work",
            "open for work",
            "open to opportunities",
            "job opportunities",
            "work opportunities",
            "available for",
            "availability",
            "open to freelance",
            "open to contract",
            "open to consulting",
        ]
        return any(marker in low for marker in markers)

    @staticmethod
    def _looks_like_profile_overview_question(msg: str) -> bool:
        low = (msg or "").strip().lower()

        markers = [
            "tell me more about samah",
            "tell me about samah",
            "who is samah",
            "introduce samah",
            "introduce her",
            "summarize samah",
            "summarise samah",
            "summary about samah",
            "overview about samah",
            "overview of samah",
            "what can you tell me about samah",
            "what can you tell me about her",
            "tell me more about her",
            "give me an overview about samah",
            "give me an overview about her",
            "describe samah",
            "describe her background",
            "what is her background",
        ]
        return any(marker in low for marker in markers)

    @staticmethod
    def _looks_like_cost_question(msg: str) -> bool:
        low = (msg or "").strip().lower()
        if SmartChatIntentService._looks_like_skill_rating_question(msg):
            return False
        markers = [
            "how much will it cost",
            "how much this will cost",
            "how much would it cost",
            "what is the cost",
            "what will be the cost",
            "how much does it cost",
            "hourly rate",
            "daily rate",
            "monthly rate",
            "pricing",
            "price",
            "budget",
            "quotation",
            "quote",
            "cost",
            "cost me",
            "compensation",
            "payment",
        ]
        return any(marker in low for marker in markers)
    
    
    @staticmethod
    def _looks_like_skill_rating_question(msg: str) -> bool:
        """
        Detect when the user is asking to rate/evaluate Samah's skill level,
        not asking about salary, hourly rate, or compensation.
        """
        low = (msg or "").strip().lower()

        rating_patterns = [
            r"\brate\s+(samah|her)\s+in\s+",
            r"\brate\s+(samah|her)\s+on\s+",
            r"\b(rate|score|evaluate)\s+(samah|her)\b.*\b(1\s*-\s*10|1\s+to\s+10|out of 10|/10)\b",
            r"\bhow good\s+(is\s+)?(samah|she)\b.*\bpython|django|react|next\.?js|javascript|typescript|ai|ml|llm\b",
            r"\bwhat level\s+(is\s+)?(samah|she)\b.*\bpython|django|react|next\.?js|javascript|typescript|ai|ml|llm\b",
        ]

        return any(re.search(pattern, low) for pattern in rating_patterns)

    @classmethod
    def _llm_classify_route(
        cls,
        msg: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[QuestionRouteResult]:
        """
        LLM-first route classifier.

        This determines HOW the question should be answered,
        not the final answer itself.
        """
        history = history or []

        compact_history = []
        for item in history[-4:]:
            role = (item.get("role") or "").strip()
            content = (item.get("content") or "").strip()
            if role and content:
                compact_history.append({
                    "role": role,
                    "content": content[:500],
                })

        system_instruction = (
            "You are a strict classifier for a portfolio chatbot.\n"
            "Your task is to classify the USER'S QUESTION into exactly one route.\n\n"
            "Choose exactly one from:\n"
            "- identity_question\n"
            "- session_memory_question\n"
            "- profile_docs_question\n"
            "- capability_inference_question\n"
            "- general_question\n\n"
            "Definitions:\n"
            "1) identity_question: asks explicitly about the chatbot/assistant itself, such as'who built you', 'what are you', 'what can you do', or 'how do you work'.\n"
            "2) session_memory_question: asks about earlier messages in the current conversation or session. "
            "This includes questions like 'what was my second question', 'what did I ask before', "
            "'what was my question about project management', 'what did we discuss about Django', "
            "'what did you say about cost', or 'summarize our discussion about freelance'.\n"
            "3) profile_docs_question: asks about Samah's profile, background, projects, skills, experience, certifications, contact details, availability, compensation, work style, strengths, impact, or any document-grounded professional information.\n"
            "This includes both broad requests like 'tell me more about samah' and specific requests like 'what is her email' or 'which projects used Django'.\n"
            "4) capability_inference_question: asks whether Samah, the chatbot, or a system/project can support, handle, build, extend, scale, fit, or help with something.\n"
            "5) general_question: anything else.\n\n"
            "Important rules:\n"
            "- Broad profile questions about Samah should be profile_docs_question, not general_question.\n"
            "- If the user asks what they previously asked, what you previously answered, or what was discussed about a topic in this chat, classify as session_memory_question.\n"
            "- Specific profile facts should also be profile_docs_question.\n"
            "- Questions asking if something CAN be done, supported, built, extended, or handled should usually be capability_inference_question.\n"
            "- If the user asks about contact details, experience, projects, availability, skills or background using words like 'you' or 'your', interpret that as Samah/profile_docs_question unless the user clearly means the chatbot itself.\n"
            "- Return JSON only."
        )

        prompt = (
            "Return JSON exactly like this:\n"
            "{"
            "\"route\":\"identity_question|session_memory_question|profile_docs_question|capability_inference_question|general_question\","
            "\"confidence\":0.0,"
            "\"reason\":\"short explanation\""
            "}\n\n"
            f"Recent history: {json.dumps(compact_history, ensure_ascii=False)}\n"
            f"User message: {msg}"
        )

        schema = {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "enum": [
                        "identity_question",
                        "session_memory_question",
                        "profile_docs_question",
                        "capability_inference_question",
                        "general_question",
                    ],
                },
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["route", "confidence", "reason"],
            "additionalProperties": False,
        }

        chain = getattr(
            settings,
            "QUESTION_ROUTE_MODEL_CHAIN",
            [
                "deepseek-chat",
            ],
        )

        try:
            ok, text, meta = LLMRouter.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.0,
                model_chain=chain,
                json_schema=schema,
                task=LLMRouter.TASK_INTENT,
            )
        except Exception:
            return None

        if not ok:
            return None

        try:
            data = json.loads(text)
            route = data.get("route", "general_question")
            confidence = float(data.get("confidence", 0.60))

            if route not in cls.ALLOWED_ROUTES:
                route = "general_question"

            confidence = max(0.0, min(1.0, confidence))

            return QuestionRouteResult(
                route=route,
                confidence=confidence,
                source="deepseek",
                raw_label=data.get("reason", ""),
            )
        except Exception:
            return None

    @classmethod
    def _fallback_route(cls, msg: str) -> QuestionRouteResult:
        """
        Tiny backup only if the LLM route classifier fails.
        This is intentionally small and not the main logic.
        """
        low = (msg or "").strip().lower()

        session_markers = [
            "first question",
            "second question",
            "third question",
            "fourth question",
            "fifth question",
            "last question",
            "previous question",
            "first answer",
            "second answer",
            "third answer",
            "last answer",
            "previous answer",
            "what did i ask",
            "what did you say",
            "what was my question",
            "what was your answer",
            "show my last",
            "summarize what we discussed",
            "summarise what we discussed",
            "what did we discuss",
            "earlier in this session",
            "earlier in this chat",
            "in this session",
        ]
        if any(marker in low for marker in session_markers) or cls._looks_like_session_memory_question(low):
            return QuestionRouteResult(
                route="session_memory_question",
                confidence=0.85,
                source="fallback",
                raw_label="session_marker",
            )

        identity_markers = [
            "who developed you",
            "who built you",
            "who made you",
            "what are you",
            "who are you",
            "what can you do",
            "what do you do",
        ]
        if any(marker in low for marker in identity_markers):
            return QuestionRouteResult(
                route="identity_question",
                confidence=0.80,
                source="fallback",
                raw_label="identity_marker",
            )

        capability_markers = [
            "can she",
            "can samah",
            "can this chatbot",
            "can the chatbot",
            "can this project",
            "can the project",
            "can this system",
            "can the system",
            "can it",
            "could it",
            "would it",
            "support",
            "handle",
            "build",
            "develop",
            "fit",
            "extend",
            "scale",
            "suitable",
            "help me with",
            "help with",
            "how can she help",
            "how can samah help",
            "how samah can help",
            "how she can help",
            "assist with",
        ]

        if any(marker in low for marker in capability_markers):
            return QuestionRouteResult(
                route="capability_inference_question",
                confidence=0.75,
                source="fallback",
                raw_label="capability_marker",
            )

        profile_doc_markers = [
            "experience",
            "years of experience",
            "total experience",
            "how long has she worked",
            "how long did she work",
            "worked at",
            "contact",
            "email",
            "phone",
            "linkedin",
            "availability",
            "salary",
            "expected salary",
            "hourly rate",
            "rate",
            "pricing",
            "price",
            "budget",
            "quotation",
            "quote",
            "cost",
            "cost me",
            "cv",
            "resume",
            "project",
            "projects",
            "skill",
            "skills",
            "certificate",
            "certification",
            "background",
            "overview",
            "summary",
            "about samah",
            "who is samah",
            "tell me more about samah",
        ]
        if any(marker in low for marker in profile_doc_markers):
            return QuestionRouteResult(
                route="profile_docs_question",
                confidence=0.75,
                source="fallback",
                raw_label="profile_docs_marker",
            )

        if cls._looks_like_work_history_fact_question(low):
            return QuestionRouteResult(
                route="profile_docs_question",
                confidence=0.78,
                source="fallback",
                raw_label="work_history_fact_marker",
            )

        if cls._looks_like_contact_question(low):
            return QuestionRouteResult(
                route="profile_docs_question",
                confidence=0.80,
                source="fallback",
                raw_label="contact_fact_marker",
            )

        if cls._looks_like_availability_question(low):
            return QuestionRouteResult(
                route="profile_docs_question",
                confidence=0.90,
                source="fallback",
                raw_label="availability_marker",
            )

        if cls._looks_like_capability_help_question(low):
            return QuestionRouteResult(
                route="capability_inference_question",
                confidence=0.78,
                source="fallback",
                raw_label="capability_help_marker",
            )

        return QuestionRouteResult(
            route="general_question",
            confidence=0.50,
            source="fallback",
            raw_label="default_general",
        )

    _INTENT_REPLIES: Dict[str, List[str]] = {
        "greeting": [
            "Hi! 👋 You can ask me about Samah’s experience, projects, technical skills, or contact details.",
            "Hello! I’m here to help you explore Samah’s background, projects, and skills.",
            "Welcome! Feel free to ask about Samah’s experience, technologies, achievements, or how to contact her.",
            "Hi there! Ask me anything about Samah’s professional profile — from projects to technical strengths.",
        ],
        "goodbye": [
            "Bye! 👋 If you need anything else about Samah’s profile, I’m here.",
            "Goodbye! Feel free to come back anytime if you’d like to know more about Samah.",
            "Take care! I’ll be here whenever you want to explore more about Samah’s experience or projects.",
            "See you soon! 👋 I’m always here if you have more questions about Samah.",
        ],
        "thanks": [
            "You’re welcome! 😊 Want to know anything else about Samah?",
            "Happy to help! Let me know if you’d like to explore more about Samah’s background or skills.",
            "My pleasure! Feel free to ask about Samah’s projects, experience, or technical expertise.",
            "Glad that helped! 😊 You can also ask about Samah’s work, achievements, or contact details.",
        ],
        "help": [
            "I can help you learn more about Samah’s experience, projects, technical skills, and professional background using her uploaded documents. You can ask things like: “What are her strongest technical skills?”, “Which projects used Django?”, or “How can I contact her?”",
            "You can ask me about Samah’s profile using her uploaded CV and documents. For example: “What AI technologies has she worked with?”, “What projects has she built?”, or “How can I reach her?”",
            "I’m here to answer questions about Samah’s background, experience, skills, and projects based on her uploaded documents. Try asking: “What is her experience with AI?”, “Which projects involved Django?”, or “What are her contact details?”",
            "Feel free to ask about Samah’s professional experience, technical stack, projects, or achievements. For example: “What are her key strengths?”, “What technologies does she use?”, or “How can I contact her?”",
        ],
        "acknowledgement": [
            "Got it 😊",
            "Sure 😊",
            "Understood.",
            "Sounds good.",
            "any other questions about Samah’s profile or experience I can help with?",
            "Let me know if you want to explore more about Samah’s projects, skills, or background.",
            "Feel free to ask more about Samah’s experience, technical skills, or contact details whenever you like.",
            "I’m here to help you learn more about Samah’s professional profile, so just ask if you have any questions!",
            "If you want to know more about Samah’s experience, projects, or skills, just let me know!",
            "Great 😊 Let me know if you’d like to explore more about Samah’s skills, projects, experience, or contact details.",
            "Perfect! You can ask me anything about Samah’s projects, technical skills, work experience, or availability.",
            "Sounds good 😊 I’m here if you’d like to know more about Samah’s background, projects, or how to contact her.",
            "Got it! Feel free to ask more about Samah’s experience, AI projects, skills, or professional profile.",
            "Okay 😊 What would you like to know more about Samah?",

        ],
        "pause": [
            "Sure, take your time.",
            "No worries, I’ll wait.",
            "Of course — take a moment.",
            "No problem, I’m here when you’re ready.",
        ],
        "positive_reaction": [
            "Glad you think so 😊",
            "Happy you liked it 😊",
            "That’s great to hear 😊",
        ],
    }

    _last_reply_by_intent: Dict[str, Optional[str]] = {}

    @classmethod
    def _reply_for_intent(cls, intent: str) -> str:
        replies = cls._INTENT_REPLIES.get(intent)
        if not replies:
            return ""

        if len(replies) == 1:
            chosen = replies[0]
        else:
            last_reply = cls._last_reply_by_intent.get(intent)
            candidates = [reply for reply in replies if reply != last_reply]
            chosen = random.SystemRandom().choice(candidates or replies)

        cls._last_reply_by_intent[intent] = chosen
        return chosen
