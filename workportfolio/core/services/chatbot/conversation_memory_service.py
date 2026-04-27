from __future__ import annotations

import re
import json
from typing import List, Dict, Any, Optional
from django.conf import settings

from core.models import ChatMessage
from core.services.llm.router import LLMRouter


class ConversationMemoryService:
    """
    Answers questions about the current chat/session history only.

    It supports:
    1. Full conversation summary
    2. Topic-based conversation summary
    3. Finding all previous user questions related to a topic
    4. Finding previous assistant answers related to a topic

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

    @staticmethod
    def _detect_memory_intent(user_question: str) -> str:
        low = (user_question or "").strip().lower()

        if re.search(r"\b(summarize|summarise|summary|recap|overview)\b", low):
            return "summary"

        if re.search(r"\b(question|questions|ask|asked)\b", low):
            return "questions"

        if re.search(r"\b(answer|answers|reply|replied|said|say)\b", low):
            return "answers"

        return "messages"

    @staticmethod
    def _extract_topic(user_question: str) -> str:
        text = (user_question or "").strip()

        patterns = [
            r"about\s+(.+)$",
            r"regarding\s+(.+)$",
            r"related to\s+(.+)$",
            r"for\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                topic = match.group(1).strip(" ?.!")

                # Clean memory wording from topic
                topic = re.sub(
                    r"\b(my|our|the|previous|earlier|conversation|chat|question|questions|answer|answers|message|messages)\b",
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

            # If local match found anything, trust it.
            if local_matches:
                return local_matches[:cls.MAX_MATCHED_MESSAGES]

            # Use LLM only as semantic fallback.
            llm_matches = cls._llm_select_relevant_messages(
                messages=messages,
                user_question=user_question,
                topic=topic,
            )

            # Strict safety filter: do not allow the LLM to return everything.
            filtered = cls._filter_topic_relevant_messages(
                messages=llm_matches,
                topic=topic,
            )

            return filtered[:cls.MAX_MATCHED_MESSAGES]

        # No topic means general memory request.
        # Example: "what did I ask before?"
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
            "message", "messages",
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

            # ✅ Important:
            # If the message does not contain/relate to the topic,
            # do NOT add it.
            if not matched_words:
                continue

            score = len(matched_words)

            # Prefer user messages only AFTER topic match is confirmed.
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

        # Safe fallback if LLM summary fails
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
