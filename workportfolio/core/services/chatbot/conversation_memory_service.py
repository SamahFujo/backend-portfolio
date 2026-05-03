from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from django.conf import settings

from core.models import ChatMessage
from core.services.llm.router import LLMRouter


@dataclass(frozen=True)
class MemoryPlan:
    """
    Represents what the user wants from the current conversation memory.

    Supported actions:
    - list_user_questions
    - list_bot_answers
    - summarize_topic
    - summarize_all
    - show_pairs
    - nth_user_question
    - nth_bot_answer
    - answer_to_specific_question
    - question_matching_topic
    - unknown
    """
    action: str
    topic: str = ""
    ordinal: Optional[int] = None
    quoted_text: str = ""
    confidence: float = 0.0
    reason: str = ""


class ConversationMemoryService:
    """
    Answers questions about the current chat/session history only.

    It supports:
    1. Full conversation summary
    2. Topic-based conversation summary
    3. Finding all previous user questions related to a topic
    4. Finding previous assistant answers related to a topic
    5. Pairing user questions with assistant responses
    6. Specific memory requests like:
       - what was my second question?
       - what was your first answer?
       - what did you answer to my Power BI question?
       - what was the bot response to "..."?

    It does not use profile documents, vector search, or RAG.
    """

    MAX_HISTORY_MESSAGES = 80
    MAX_MATCHED_MESSAGES = 12

    @classmethod
    def answer(
        cls,
        session_id,
        user_question: str,
        current_user_message_id=None,
    ) -> Dict[str, Any]:
        messages = cls._load_session_messages(
            session_id=session_id,
            exclude_message_id=current_user_message_id,
        )

        if not messages:
            return {
                "answer": "I do not have earlier messages in this conversation yet.",
                "confidence": 0.75,
                "source": "conversation_memory",
                "matched_messages": [],
            }

        # ✅ Smart memory layer first:
        # Handles paired Q/A, bot response requests, ordinal questions,
        # quoted-question matching, and topic-based memory.
        smart_result = cls._answer_with_memory_plan(
            messages=messages,
            user_question=user_question,
        )

        if smart_result:
            return smart_result

        # ✅ Existing fallback flow remains available.
        memory_intent = cls._detect_memory_intent(user_question)
        topic = cls._extract_topic(user_question)

        # Case 1: summarize the whole conversation
        if memory_intent == "summary" and not topic:
            answer = cls._summarize_messages(
                messages=messages,
                user_question=user_question,
                topic="",
            )
            return {
                "answer": answer,
                "confidence": 0.90,
                "source": "conversation_summary",
                "matched_messages": messages[-20:],
            }

        # Case 2: topic summary
        if memory_intent == "summary" and topic:
            matched = cls._find_relevant_messages(
                messages=messages,
                user_question=user_question,
                topic=topic,
            )

            if not matched:
                return {
                    "answer": f"I could not find earlier conversation messages clearly related to {topic}.",
                    "confidence": 0.65,
                    "source": "conversation_memory",
                    "matched_messages": [],
                }

            answer = cls._summarize_messages(
                messages=matched,
                user_question=user_question,
                topic=topic,
            )
            return {
                "answer": answer,
                "confidence": 0.88,
                "source": "conversation_topic_summary",
                "matched_messages": matched,
            }

        # Case 3: find previous questions/answers/messages about a topic
        matched = cls._find_relevant_messages(
            messages=messages,
            user_question=user_question,
            topic=topic,
        )

        if not matched:
            return {
                "answer": "I could not find previous messages in this conversation that clearly match that topic.",
                "confidence": 0.65,
                "source": "conversation_memory",
                "matched_messages": [],
            }

        answer = cls._build_lookup_answer(
            user_question=user_question,
            topic=topic,
            matched_messages=matched,
            memory_intent=memory_intent,
        )

        return {
            "answer": answer,
            "confidence": 0.90,
            "source": "conversation_memory",
            "matched_messages": matched,
        }

    @classmethod
    def _load_session_messages(
        cls,
        session_id,
        exclude_message_id=None,
    ) -> List[Dict[str, Any]]:
        qs = (
            ChatMessage.objects
            .filter(session_id=session_id)
            .order_by("created_at")
            .values("id", "role", "content", "created_at")
        )

        rows = list(qs)

        if exclude_message_id:
            rows = [
                row for row in rows
                if str(row.get("id")) != str(exclude_message_id)
            ]

        rows = rows[-cls.MAX_HISTORY_MESSAGES:]

        messages = []

        for index, row in enumerate(rows, start=1):
            content = (row.get("content") or "").strip()
            role = (row.get("role") or "").strip()

            if not content or role not in {"user", "assistant"}:
                continue

            messages.append({
                "id": str(row.get("id")),
                "index": index,
                "role": role,
                "content": content[:1500],
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
            })

        return messages

    # -------------------------------------------------------------------------
    # Smart paired-turn memory layer
    # -------------------------------------------------------------------------

    @classmethod
    def _answer_with_memory_plan(
        cls,
        messages: List[Dict[str, Any]],
        user_question: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Smart session-memory layer.

        This method improves questions like:
        - what was the bot response to these questions?
        - what did you answer about Power BI?
        - what was my second question?
        - what was your first answer?
        - what was the bot response to "i have a project need power bi skills"?
        - summarize only the Power BI part

        It does not change the chatbot workflow.
        It only runs after the route is already session_memory_question.
        """
        turns = cls._build_conversation_turns(messages)

        if not turns:
            return None

        plan = cls._detect_memory_plan(
            user_question=user_question,
            turns=turns,
        )

        if plan.confidence < 0.60 or plan.action == "unknown":
            return None

        selected_turns = cls._select_turns_from_plan(
            turns=turns,
            plan=plan,
        )

        if not selected_turns:
            return {
                "answer": "I could not find earlier conversation turns that clearly match that request.",
                "confidence": 0.65,
                "source": "conversation_memory_smart",
                "matched_messages": [],
            }

        answer = cls._generate_memory_plan_answer(
            user_question=user_question,
            plan=plan,
            selected_turns=selected_turns,
        )

        return {
            "answer": answer,
            "confidence": max(0.75, plan.confidence),
            "source": "conversation_memory_smart",
            "matched_messages": cls._flatten_turns_to_messages(selected_turns),
        }

    @staticmethod
    def _build_conversation_turns(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert flat messages into user-assistant pairs.

        Example:
        [
            {
                "turn": 1,
                "user_message": {...},
                "assistant_message": {...},
                "user": "question text",
                "assistant": "answer text",
            }
        ]

        This is the key improvement because memory questions often need:
        user question -> bot answer mapping.
        """
        turns = []
        pending_user = None

        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()

            if not content:
                continue

            if role == "user":
                pending_user = msg

            elif role == "assistant" and pending_user:
                turns.append({
                    "turn": len(turns) + 1,
                    "user_message": pending_user,
                    "assistant_message": msg,
                    "user": pending_user.get("content", ""),
                    "assistant": msg.get("content", ""),
                })
                pending_user = None

        return turns

    @classmethod
    def _detect_memory_plan(
        cls,
        user_question: str,
        turns: List[Dict[str, Any]],
    ) -> MemoryPlan:
        """
        Uses LLM to understand what memory operation the user wants.

        This is only called inside session memory flow, so the cost is limited.
        """
        compact_turns = [
            {
                "turn": turn["turn"],
                "user": turn["user"][:300],
                "assistant": turn["assistant"][:400],
            }
            for turn in turns[-10:]
        ]

        system_instruction = (
            "You classify what the user wants from the current chat history only.\n\n"
            "Allowed actions:\n"
            "- list_user_questions: user wants previous questions they asked.\n"
            "- list_bot_answers: user wants previous bot/assistant responses.\n"
            "- summarize_topic: user wants a summary about a specific topic from the chat.\n"
            "- summarize_all: user wants a summary of the full chat.\n"
            "- show_pairs: user wants both questions and bot answers.\n"
            "- nth_user_question: user asks for their first/second/third/etc question.\n"
            "- nth_bot_answer: user asks for the bot's first/second/third/etc answer.\n"
            "- answer_to_specific_question: user asks for the bot response to a specific previous question.\n"
            "- question_matching_topic: user asks for the specific previous question related to a topic.\n"
            "- unknown: unclear.\n\n"
            "Important rules:\n"
            "- If the user says 'bot response', 'your response', 'assistant answer', or 'what did you answer', choose list_bot_answers or answer_to_specific_question.\n"
            "- If the user says 'these questions', infer they refer to recently discussed/listed questions.\n"
            "- If the user asks for a summary about a topic, choose summarize_topic and extract the topic.\n"
            "- If the user asks what they asked, choose list_user_questions.\n"
            "- If the user asks for first/second/third question, choose nth_user_question and set ordinal.\n"
            "- If the user asks for first/second/third answer/response, choose nth_bot_answer and set ordinal.\n"
            "- If the user quotes a previous question and asks for the response, choose answer_to_specific_question and copy the quoted text.\n"
            "- Extract topic only when clearly mentioned, such as Power BI, Django, freelance, contact, salary, etc.\n"
            "- Return JSON only."
        )

        prompt = (
            "Return JSON exactly like this:\n"
            "{"
            "\"action\":\"list_user_questions|list_bot_answers|summarize_topic|summarize_all|show_pairs|nth_user_question|nth_bot_answer|answer_to_specific_question|question_matching_topic|unknown\","
            "\"topic\":\"\","
            "\"ordinal\":null,"
            "\"quoted_text\":\"\","
            "\"confidence\":0.0,"
            "\"reason\":\"short explanation\""
            "}\n\n"
            f"User memory request: {user_question}\n\n"
            f"Recent conversation turns: {json.dumps(compact_turns, ensure_ascii=False)}"
        )

        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_user_questions",
                        "list_bot_answers",
                        "summarize_topic",
                        "summarize_all",
                        "show_pairs",
                        "nth_user_question",
                        "nth_bot_answer",
                        "answer_to_specific_question",
                        "question_matching_topic",
                        "unknown",
                    ],
                },
                "topic": {"type": "string"},
                "ordinal": {"type": ["integer", "null"]},
                "quoted_text": {"type": "string"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": [
                "action",
                "topic",
                "ordinal",
                "quoted_text",
                "confidence",
                "reason",
            ],
            "additionalProperties": False,
        }

        chain = getattr(
            settings,
            "SESSION_MEMORY_MODEL_CHAIN",
            getattr(settings, "QUESTION_ROUTE_MODEL_CHAIN", ["deepseek-chat"]),
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
            return cls._local_memory_plan_fallback(user_question)

        if not ok:
            return cls._local_memory_plan_fallback(user_question)

        try:
            data = json.loads(text)

            action = data.get("action", "unknown")
            allowed_actions = {
                "list_user_questions",
                "list_bot_answers",
                "summarize_topic",
                "summarize_all",
                "show_pairs",
                "nth_user_question",
                "nth_bot_answer",
                "answer_to_specific_question",
                "question_matching_topic",
                "unknown",
            }

            if action not in allowed_actions:
                action = "unknown"

            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))

            ordinal = data.get("ordinal")
            if ordinal is not None:
                try:
                    ordinal = int(ordinal)
                except Exception:
                    ordinal = None

            return MemoryPlan(
                action=action,
                topic=(data.get("topic") or "").strip(),
                ordinal=ordinal,
                quoted_text=(data.get("quoted_text") or "").strip(),
                confidence=confidence,
                reason=data.get("reason", ""),
            )

        except Exception:
            return cls._local_memory_plan_fallback(user_question)

    @staticmethod
    def _local_memory_plan_fallback(user_question: str) -> MemoryPlan:
        """
        Cheap fallback if LLM classification fails.
        """
        text = user_question or ""
        low = text.strip().lower()

        # Quoted question fallback:
        # Example: what was the bot response to "i have a project need power bi skills"
        quote_match = re.search(r"[\"“'](.+?)[\"”']", text)

        if quote_match and any(term in low for term in ["answer", "response", "reply", "bot", "assistant"]):
            return MemoryPlan(
                action="answer_to_specific_question",
                quoted_text=quote_match.group(1).strip(),
                confidence=0.78,
                reason="Local fallback detected answer to quoted question.",
            )

        ordinal_map = {
            "first": 1,
            "1st": 1,
            "second": 2,
            "2nd": 2,
            "third": 3,
            "3rd": 3,
            "fourth": 4,
            "4th": 4,
            "fifth": 5,
            "5th": 5,
            "sixth": 6,
            "6th": 6,
            "seventh": 7,
            "7th": 7,
            "eighth": 8,
            "8th": 8,
            "ninth": 9,
            "9th": 9,
            "tenth": 10,
            "10th": 10,
        }

        # Ordinal fallback:
        # Example: what was my second question?
        for word, number in ordinal_map.items():
            if re.search(rf"\b{re.escape(word)}\b", low) and "question" in low:
                return MemoryPlan(
                    action="nth_user_question",
                    ordinal=number,
                    confidence=0.75,
                    reason="Local fallback detected nth user question.",
                )

            if re.search(rf"\b{re.escape(word)}\b", low) and any(term in low for term in ["answer", "response", "reply"]):
                return MemoryPlan(
                    action="nth_bot_answer",
                    ordinal=number,
                    confidence=0.75,
                    reason="Local fallback detected nth bot answer.",
                )

        if any(term in low for term in [
            "bot response",
            "bot responses",
            "your response",
            "your responses",
            "assistant response",
            "assistant answer",
            "your answer",
            "your answers",
            "what did you answer",
            "what was the response",
            "what were the responses",
        ]):
            return MemoryPlan(
                action="list_bot_answers",
                topic="",
                confidence=0.72,
                reason="Local fallback detected bot response request.",
            )

        if any(term in low for term in [
            "my question",
            "my questions",
            "what did i ask",
            "questions i asked",
        ]):
            return MemoryPlan(
                action="list_user_questions",
                topic="",
                confidence=0.70,
                reason="Local fallback detected user question request.",
            )

        if any(term in low for term in ["summarize", "summarise", "summary", "recap", "overview"]):
            extracted_topic = ""

            topic_patterns = [
                r"about\s+(.+)$",
                r"regarding\s+(.+)$",
                r"related to\s+(.+)$",
                r"in regard of\s+(.+)$",
                r"in regards to\s+(.+)$",
            ]

            for pattern in topic_patterns:
                match = re.search(pattern, text, flags=re.I)
                if match:
                    extracted_topic = match.group(1).strip(" ?.!")

                    extracted_topic = re.sub(
                        r"\b(my|our|the|previous|earlier|conversation|chat|question|questions|answer|answers|message|messages)\b",
                        "",
                        extracted_topic,
                        flags=re.I,
                    )

                    extracted_topic = re.sub(
                        r"\s+", " ", extracted_topic).strip()
                    break

            return MemoryPlan(
                action="summarize_topic" if extracted_topic else "summarize_all",
                topic=extracted_topic,
                confidence=0.68,
                reason="Local fallback detected summary request.",
            )

        return MemoryPlan(
            action="unknown",
            topic="",
            confidence=0.30,
            reason="No local memory pattern matched.",
        )

    @classmethod
    def _select_turns_from_plan(
        cls,
        turns: List[Dict[str, Any]],
        plan: MemoryPlan,
    ) -> List[Dict[str, Any]]:
        """
        Select relevant user-assistant turns based on:
        - ordinal number
        - quoted question text
        - topic
        - recent context
        """

        # Case 1: specific ordinal question/answer
        if plan.ordinal:
            index = plan.ordinal - 1

            if 0 <= index < len(turns):
                return [turns[index]]

            return []

        # Case 2: answer to a specific quoted question
        if plan.quoted_text:
            quoted_norm = cls._normalize(plan.quoted_text)

            best_turn = None
            best_score = 0

            quoted_words = [
                word for word in quoted_norm.split()
                if len(word) > 2
            ]

            for turn in turns:
                user_norm = cls._normalize(turn.get("user", ""))

                score = sum(1 for word in quoted_words if word in user_norm)

                if quoted_norm and quoted_norm in user_norm:
                    score += 10

                if score > best_score:
                    best_score = score
                    best_turn = turn

            if best_turn and best_score >= 2:
                return [best_turn]

            return []

        # Case 3: topic-based selection
        topic = (plan.topic or "").strip()

        if topic:
            topic_norm = cls._normalize(topic)

            stop_words = {
                "the", "and", "for", "with", "about",
                "her", "his", "she", "him", "you",
                "your", "our", "this", "that", "was",
                "were", "did", "does", "what", "which",
                "question", "questions", "answer", "answers",
                "message", "messages", "response", "responses",
                "bot", "assistant",
            }

            topic_words = [
                word for word in topic_norm.split()
                if len(word) > 2 and word not in stop_words
            ]

            matched_turns = []

            for turn in turns:
                haystack = cls._normalize(
                    f"{turn.get('user', '')} {turn.get('assistant', '')}"
                )

                if any(word in haystack for word in topic_words):
                    matched_turns.append(turn)

            return matched_turns[-cls.MAX_MATCHED_MESSAGES:]

        # Case 4: no topic.
        # For "these questions", "previous answers", "what was the bot response",
        # use recent paired turns.
        if plan.action in {
            "list_user_questions",
            "list_bot_answers",
            "show_pairs",
            "summarize_topic",
            "summarize_all",
        }:
            return turns[-5:]

        return turns[-5:]

    @classmethod
    def _generate_memory_plan_answer(
        cls,
        user_question: str,
        plan: MemoryPlan,
        selected_turns: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a natural final answer from selected turns only.
        """
        compact_turns = [
            {
                "turn": turn["turn"],
                "user": turn["user"][:700],
                "assistant": turn["assistant"][:900],
            }
            for turn in selected_turns
        ]

        system_instruction = (
            "You answer questions about the current chat history only.\n"
            "Use only the selected conversation turns.\n\n"
            "Rules:\n"
            "- If the user asks for bot responses, return the assistant responses, not only the user questions.\n"
            "- If the user asks for previous questions, return the user questions.\n"
            "- If the user asks for both, show each user question with its bot response.\n"
            "- If the user asks for a topic summary, summarize only the selected turns.\n"
            "- If action is nth_user_question, return only that user question.\n"
            "- If action is nth_bot_answer, return only that assistant answer.\n"
            "- If action is answer_to_specific_question, return the assistant response paired with that user question.\n"
            "- If action is question_matching_topic, return the matching user question only.\n"
            "- Do not invent anything outside the selected turns.\n"
            "- Keep the answer clear, friendly, and organized.\n"
            "- Use numbered bullets when there are multiple items.\n"
        )

        prompt = (
            f"User memory request: {user_question}\n"
            f"Detected action: {plan.action}\n"
            f"Detected topic: {plan.topic}\n"
            f"Detected ordinal: {plan.ordinal}\n"
            f"Detected quoted text: {plan.quoted_text}\n\n"
            f"Selected turns: {json.dumps(compact_turns, ensure_ascii=False)}\n\n"
            "Write the final answer for the user."
        )

        chain = getattr(
            settings,
            "SESSION_MEMORY_REPLY_MODEL_CHAIN",
            getattr(
                settings,
                "SESSION_MEMORY_MODEL_CHAIN",
                getattr(settings, "QUESTION_ROUTE_MODEL_CHAIN",
                        ["deepseek-chat"]),
            ),
        )

        try:
            ok, text, meta = LLMRouter.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.25,
                model_chain=chain,
                task=LLMRouter.TASK_INTENT,
            )
        except Exception:
            ok = False
            text = ""

        if ok and text:
            return text.strip()

        return cls._format_memory_plan_fallback(
            plan=plan,
            selected_turns=selected_turns,
        )

    @staticmethod
    def _format_memory_plan_fallback(
        plan: MemoryPlan,
        selected_turns: List[Dict[str, Any]],
    ) -> str:
        """
        Emergency formatter if LLM answer generation fails.
        """
        if not selected_turns:
            return "I could not find a matching earlier part of this conversation."

        if plan.action == "nth_user_question":
            turn = selected_turns[0]
            return (
                f"😊 Sure — your question number {plan.ordinal} was:\n\n"
                f"💬 “{turn.get('user', '').strip()}”"
            )

        if plan.action == "nth_bot_answer":
            turn = selected_turns[0]
            return (
                f"😊 Sure — my answer number {plan.ordinal} was:\n\n"
                f"🤖 “{turn.get('assistant', '').strip()}”"
            )

        if plan.action == "answer_to_specific_question":
            turn = selected_turns[0]
            return (
                "😊 Sure — this was the bot response to that question:\n\n"
                f"💬 You asked: “{turn.get('user', '').strip()}”\n\n"
                f"🤖 Bot answered: “{turn.get('assistant', '').strip()}”"
            )

        if plan.action == "question_matching_topic":
            turn = selected_turns[0]
            return (
                "😊 Sure — the matching question was:\n\n"
                f"💬 “{turn.get('user', '').strip()}”"
            )

        if plan.action == "list_bot_answers":
            lines = ["😊 Sure — here are the related bot responses:", ""]

            for i, turn in enumerate(selected_turns, start=1):
                user_text = turn.get("user", "").strip()
                assistant_text = turn.get("assistant", "").strip()

                if len(user_text) > 250:
                    user_text = user_text[:250].rstrip() + "..."

                if len(assistant_text) > 700:
                    assistant_text = assistant_text[:700].rstrip() + "..."

                lines.append(
                    f"{i}. 💬 You asked: “{user_text}”\n"
                    f"   🤖 Bot answered: “{assistant_text}”"
                )

            return "\n\n".join(lines)

        if plan.action == "list_user_questions":
            lines = ["😊 Sure — here are the related questions you asked:", ""]

            for i, turn in enumerate(selected_turns, start=1):
                user_text = turn.get("user", "").strip()

                if len(user_text) > 500:
                    user_text = user_text[:500].rstrip() + "..."

                lines.append(f"{i}. 💬 “{user_text}”")

            return "\n".join(lines)

        if plan.action == "show_pairs":
            lines = [
                "😊 Sure — here are the related questions and bot responses:", ""]

            for i, turn in enumerate(selected_turns, start=1):
                user_text = turn.get("user", "").strip()
                assistant_text = turn.get("assistant", "").strip()

                if len(user_text) > 250:
                    user_text = user_text[:250].rstrip() + "..."

                if len(assistant_text) > 700:
                    assistant_text = assistant_text[:700].rstrip() + "..."

                lines.append(
                    f"{i}. 💬 You asked: “{user_text}”\n"
                    f"   🤖 Bot answered: “{assistant_text}”"
                )

            return "\n\n".join(lines)

        lines = ["Here is the related part of the conversation:", ""]

        for i, turn in enumerate(selected_turns, start=1):
            user_text = turn.get("user", "").strip()
            assistant_text = turn.get("assistant", "").strip()

            if len(user_text) > 250:
                user_text = user_text[:250].rstrip() + "..."

            if len(assistant_text) > 700:
                assistant_text = assistant_text[:700].rstrip() + "..."

            lines.append(
                f"{i}. You asked: “{user_text}”\n"
                f"   Bot answered: “{assistant_text}”"
            )

        return "\n\n".join(lines)

    @staticmethod
    def _flatten_turns_to_messages(
        turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert selected turns back to message list for debug/matched_messages.
        """
        messages = []

        for turn in turns:
            user_message = turn.get("user_message")
            assistant_message = turn.get("assistant_message")

            if user_message:
                messages.append(user_message)

            if assistant_message:
                messages.append(assistant_message)

        return messages

    # -------------------------------------------------------------------------
    # Existing fallback logic, improved slightly
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_memory_intent(user_question: str) -> str:
        low = (user_question or "").strip().lower()

        if re.search(r"\b(summarize|summarise|summary|recap|overview)\b", low):
            return "summary"

        # ✅ Important:
        # Check answers/responses before questions.
        # Example: "what was the bot response to these questions"
        # should be answers, not questions.
        if re.search(
            r"\b(answer|answers|reply|replies|replied|said|say|response|responses)\b",
            low,
        ):
            return "answers"

        if re.search(r"\b(question|questions|ask|asked)\b", low):
            return "questions"

        return "messages"

    @staticmethod
    def _extract_topic(user_question: str) -> str:
        text = (user_question or "").strip()

        patterns = [
            r"about\s+(.+)$",
            r"regarding\s+(.+)$",
            r"related to\s+(.+)$",
            r"in regard of\s+(.+)$",
            r"in regards to\s+(.+)$",
            r"for\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                topic = match.group(1).strip(" ?.!")

                # Clean memory wording from topic
                topic = re.sub(
                    r"\b(my|our|the|previous|earlier|conversation|chat|question|questions|answer|answers|response|responses|message|messages|bot|assistant)\b",
                    "",
                    topic,
                    flags=re.I,
                )

                topic = re.sub(r"\s+", " ", topic).strip()
                return topic

        return ""

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\u0600-\u06FF\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _find_relevant_messages(
        cls,
        messages: List[Dict[str, Any]],
        user_question: str,
        topic: str,
    ) -> List[Dict[str, Any]]:
        """
        Finds messages related to the requested topic.

        Important:
        - If the user asks about a specific topic, do not return unrelated history.
        - Local exact/keyword matching is trusted first.
        - LLM semantic matching is allowed only as fallback.
        - LLM results are post-filtered so it cannot return the whole conversation.
        """
        topic = (topic or "").strip()

        # If the user asked about a topic, require topic relevance.
        if topic:
            local_matches = cls._local_match(
                messages=messages,
                topic=topic,
                user_question=user_question,
            )

            if local_matches:
                return local_matches[:cls.MAX_MATCHED_MESSAGES]

            llm_matches = cls._llm_select_relevant_messages(
                messages=messages,
                user_question=user_question,
                topic=topic,
            )

            filtered = cls._filter_topic_relevant_messages(
                messages=llm_matches,
                topic=topic,
            )

            return filtered[:cls.MAX_MATCHED_MESSAGES]

        # No topic means general memory request.
        llm_matches = cls._llm_select_relevant_messages(
            messages=messages,
            user_question=user_question,
            topic=topic,
        )

        return llm_matches[:cls.MAX_MATCHED_MESSAGES]

    @classmethod
    def _local_match(
        cls,
        messages: List[Dict[str, Any]],
        topic: str,
        user_question: str,
    ) -> List[Dict[str, Any]]:
        if not topic:
            return []

        topic_norm = cls._normalize(topic)

        stop_words = {
            "the", "and", "for", "with", "about",
            "her", "his", "she", "him", "you",
            "your", "our", "this", "that", "was",
            "were", "did", "does", "what", "which",
            "question", "questions", "answer", "answers",
            "response", "responses", "message", "messages",
            "bot", "assistant",
        }

        topic_words = [
            word for word in topic_norm.split()
            if len(word) > 2 and word not in stop_words
        ]

        if not topic_words:
            return []

        matched = []

        for msg in messages:
            content_norm = cls._normalize(msg["content"])

            matched_words = [
                word for word in topic_words
                if word in content_norm
            ]

            if not matched_words:
                continue

            score = len(matched_words)

            if msg["role"] == "user":
                score += 0.25

            matched.append({
                **msg,
                "match_score": score,
                "topic_matched_words": matched_words,
            })

        matched = sorted(
            matched,
            key=lambda x: (-float(x.get("match_score", 0)), x.get("index", 0)),
        )

        return matched[:cls.MAX_MATCHED_MESSAGES]

    @classmethod
    def _filter_topic_relevant_messages(
        cls,
        messages: List[Dict[str, Any]],
        topic: str,
    ) -> List[Dict[str, Any]]:
        """
        Final protection layer.

        Even if the LLM selects unrelated messages, this keeps only messages
        that have a clear relationship with the requested topic.
        """
        topic_norm = cls._normalize(topic)

        topic_words = [
            word for word in topic_norm.split()
            if len(word) > 2 and word not in {
                "the", "and", "for", "with", "about",
                "her", "his", "she", "him", "you",
                "your", "our", "this", "that",
                "question", "questions", "answer", "answers",
                "response", "responses", "message", "messages",
                "bot", "assistant",
            }
        ]

        if not topic_words:
            return []

        filtered = []

        for msg in messages:
            content_norm = cls._normalize(msg.get("content", ""))

            matched_words = [
                word for word in topic_words
                if word in content_norm
            ]

            if matched_words:
                filtered.append({
                    **msg,
                    "topic_matched_words": matched_words,
                })

        return filtered

    @classmethod
    def _llm_select_relevant_messages(
        cls,
        messages: List[Dict[str, Any]],
        user_question: str,
        topic: str,
    ) -> List[Dict[str, Any]]:
        compact_messages = [
            {
                "index": msg["index"],
                "role": msg["role"],
                "content": msg["content"][:700],
            }
            for msg in messages
        ]

        system_instruction = (
            "You select relevant messages from a conversation history.\n"
            "The user is asking about the current conversation only.\n"
            "Return only messages that are clearly related to the user's requested topic.\n"
            "Do not return general messages just because they are part of the conversation.\n"
            "Do not return greetings, thanks, acknowledgements, or unrelated questions.\n"
            "If no message is clearly related to the topic, return an empty matched_indexes list.\n"
            "If the user asks for their previous questions, prefer user messages only.\n"
            "If the user asks what the assistant answered, prefer assistant messages only.\n"
            "Do not include the current user question if it appears in history.\n"
            "Do not invent messages.\n"
            "Return JSON only."
        )

        prompt = (
            "Return JSON exactly like this:\n"
            "{"
            "\"matched_indexes\":[1,2],"
            "\"confidence\":0.0"
            "}\n\n"
            f"User question: {user_question}\n"
            f"Requested topic: {topic}\n\n"
            "Selection rule:\n"
            "- Select a message only if it is clearly about the requested topic.\n"
            "- If the requested topic is not found in the conversation, return matched_indexes as [].\n"
            "- Do not select all messages.\n\n"
            f"Conversation messages: {json.dumps(compact_messages, ensure_ascii=False)}"
        )

        schema = {
            "type": "object",
            "properties": {
                "matched_indexes": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "confidence": {"type": "number"},
            },
            "required": ["matched_indexes", "confidence"],
            "additionalProperties": False,
        }

        chain = getattr(
            settings,
            "SESSION_MEMORY_MODEL_CHAIN",
            getattr(settings, "QUESTION_ROUTE_MODEL_CHAIN", ["deepseek-chat"]),
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
            return []

        if not ok:
            return []

        try:
            data = json.loads(text)
            indexes = set(data.get("matched_indexes", []))
        except Exception:
            return []

        selected = [msg for msg in messages if msg["index"] in indexes]

        return selected[:cls.MAX_MATCHED_MESSAGES]

    @classmethod
    def _summarize_messages(
        cls,
        messages: List[Dict[str, Any]],
        user_question: str,
        topic: str = "",
    ) -> str:
        compact_messages = [
            {
                "role": msg["role"],
                "content": msg["content"][:1000],
            }
            for msg in messages
        ]

        system_instruction = (
            "You summarize the current chat conversation.\n"
            "Use only the provided conversation messages.\n"
            "Do not mention documents unless they were discussed in the messages.\n"
            "Do not invent facts.\n"
            "Keep the summary clear, useful, and concise.\n"
        )

        if topic:
            prompt = (
                f"Summarize the parts of the conversation related to: {topic}\n\n"
                f"User request: {user_question}\n"
                f"Conversation messages: {json.dumps(compact_messages, ensure_ascii=False)}\n\n"
                "Write the answer in this style:\n"
                "Here is a summary of what we discussed about [topic]:\n"
                "- ...\n"
                "- ...\n"
                "- ...\n"
            )
        else:
            prompt = (
                f"User request: {user_question}\n"
                f"Conversation messages: {json.dumps(compact_messages, ensure_ascii=False)}\n\n"
                "Write a real summary of the conversation so far.\n"
                "Group related points together.\n"
                "Mention the main questions the user asked and the main answers given.\n"
            )

        chain = getattr(
            settings,
            "SESSION_MEMORY_MODEL_CHAIN",
            getattr(settings, "QUESTION_ROUTE_MODEL_CHAIN", ["deepseek-chat"]),
        )

        try:
            ok, text, meta = LLMRouter.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.2,
                model_chain=chain,
                task=LLMRouter.TASK_INTENT,
            )
        except Exception:
            ok = False
            text = ""

        if ok and text:
            return text.strip()

        recent_user_questions = [
            msg["content"] for msg in messages
            if msg["role"] == "user"
        ][-8:]

        if not recent_user_questions:
            return "I found earlier conversation messages, but I could not generate a detailed summary right now."

        bullets = "\n".join([f"- {q}" for q in recent_user_questions])

        return (
            "Here is a quick summary based on your recent questions:\n\n"
            f"{bullets}"
        )

    @staticmethod
    def _build_lookup_answer(
        user_question: str,
        topic: str,
        matched_messages: List[Dict[str, Any]],
        memory_intent: str,
    ) -> str:
        if memory_intent == "questions":
            selected = [m for m in matched_messages if m["role"] == "user"]
            title = f"😊 Sure — I found these questions related to {topic}:" if topic else "😊 Sure — I found these related questions:"

        elif memory_intent == "answers":
            selected = [
                m for m in matched_messages if m["role"] == "assistant"]
            title = f"😊 Sure — I found these answers related to {topic}:" if topic else "😊 Sure — I found these related answers:"

        else:
            selected = matched_messages
            title = f"😊 Sure — I found these related messages about {topic}:" if topic else "😊 Sure — I found these related messages:"

        if not selected:
            if memory_intent == "questions":
                return f"I found related conversation messages about {topic}, but I could not find a previous user question specifically about it."

            if memory_intent == "answers":
                return f"I found related conversation messages about {topic}, but I could not find a previous assistant answer specifically about it."

            return f"I could not find previous messages clearly related to {topic}."

        lines = [title, ""]

        for i, msg in enumerate(selected, start=1):
            icon = "💬" if msg["role"] == "user" else "🤖"
            role_label = "You asked" if msg["role"] == "user" else "I answered"
            content = msg["content"].strip()

            if len(content) > 500:
                content = content[:500].rstrip() + "..."

            lines.append(f"{i}. {icon} {role_label}: “{content}”")

        lines.append("")

        return "\n".join(lines)
