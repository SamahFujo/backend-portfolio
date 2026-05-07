"""
Public chatbot API views.
"""

import re
import logging

from django.conf import settings
from django.core.validators import validate_email
from django.core.exceptions import ValidationError

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from core.models import ChatSession, ChatMessage
from core.serializers import AskQuestionSerializer
from core.throttles import ChatRateThrottle, ContactRateThrottle
from core.services.resend_email import send_chat_history_email
from core.services.chatbot.hybrid_query_rewriter import GeminiQueryRewriter
from core.services.chatbot.smart_chat_intents import SmartChatIntentService
from core.services.chatbot.profile_qa_service import ProfileQAService
from core.services.chatbot.conversation_memory_service import ConversationMemoryService

from .contact_views import CsrfExemptSessionAuthentication


logger = logging.getLogger(__name__)





"""API views for the portfolio chatbot backend."""


class AskAboutMeAPIView(APIView):

    throttle_classes = [ChatRateThrottle]

    @staticmethod
    def _get_client_ip(request):
        """
        Extract the real client IP address.

        If the app is behind NGINX or a proxy, HTTP_X_FORWARDED_FOR may contain
        the original client IP.
        """
        forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")

        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        return request.META.get("REMOTE_ADDR")

    @staticmethod
    def _get_request_metadata(request):
        """
        Collect lightweight request metadata for analytics and security auditing.
        """
        return {
            "ip_address": AskAboutMeAPIView._get_client_ip(request),
            "user_agent": request.META.get("HTTP_USER_AGENT", ""),
            "referrer": request.META.get("HTTP_REFERER", ""),
        }

    @staticmethod
    def _clean_answer_text(answer: str) -> str:
        answer = (answer or "").strip()
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer

    @staticmethod
    def _with_optional_debug(payload, **debug_fields):
        if settings.DEBUG:
            payload.update(debug_fields)
        return payload

    @staticmethod
    def _build_contact_capture_payload(session, message: str, user_message_id=None):
        answer = (
            "Sure 😊 You can share your details using the contact form, "
            "and Samah will be able to follow up with you."
        )

        assistant_message = ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=answer,
            citations=[],
            confidence_score=0.95,
        )

        return {
            "session_id": str(session.id),
            "message_id": str(assistant_message.id),
            "answer": answer,
            "citations": [],
            "confidence": 0.95,
            "mode": "contact_capture",
            "intent_source": "contact_capture_rule",
            "ui_action": {
                "type": "open_contact_modal",
                "label": "Share my details",
                "initial_subject": "Contact request from portfolio chatbot",
                "initial_message": (
                    "Hi Samah,\n\n"
                    "I would like to share my details so you can contact me.\n\n"
                    "Reason for contacting:"
                ),
            },
        }

    @staticmethod
    def _build_send_history_prompt_payload(session, message: str, initial_email: str | None = None):
        answer = (
            "Sure 😊 Please enter your email address below, "
            "and I’ll send you this conversation history."
        )

        assistant_message = ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=answer,
            citations=[],
            confidence_score=0.95,
        )

        return {
            "session_id": str(session.id),
            "message_id": str(assistant_message.id),
            "answer": answer,
            "citations": [],
            "confidence": 0.95,
            "mode": "send_history_email",
            "intent_source": "ui_action_classifier",
            "ui_action": {
                "type": "collect_email_for_history",
                "label": "Send",
                "initial_email": initial_email or "",
            },
        }

    def post(self, request, *args, **kwargs):
        serializer = AskQuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data.get("session_id")
        visitor_id = serializer.validated_data.get("visitor_id")
        visitor_email = serializer.validated_data.get("visitor_email")
        message = serializer.validated_data["message"].strip()

        request_meta = self._get_request_metadata(request)

        if session_id:
            session = ChatSession.objects.filter(
                id=session_id,
                is_active=True
            ).first()

            if session is None:
                return Response(
                    {"detail": "Session not found."},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Update missing tracking fields only if they were not stored before.
            update_fields = []

            if visitor_email and not session.visitor_email:
                session.visitor_email = visitor_email
                update_fields.append("visitor_email")

            if request_meta["ip_address"] and not session.ip_address:
                session.ip_address = request_meta["ip_address"]
                update_fields.append("ip_address")

            if request_meta["user_agent"] and not session.user_agent:
                session.user_agent = request_meta["user_agent"]
                update_fields.append("user_agent")

            if request_meta["referrer"] and not session.referrer:
                session.referrer = request_meta["referrer"]
                update_fields.append("referrer")

            if update_fields:
                session.save(update_fields=update_fields)

        else:
            session = ChatSession.objects.create(
                visitor_id=visitor_id,
                visitor_email=visitor_email,
                ip_address=request_meta["ip_address"],
                user_agent=request_meta["user_agent"],
                referrer=request_meta["referrer"],
            )

        user_message = ChatMessage.objects.create(
            session=session,
            role="user",
            content=message,
            metadata={
                "visitor_id": visitor_id,
                "visitor_email": visitor_email,
                "ip_address": request_meta["ip_address"],
                "user_agent": request_meta["user_agent"],
                "referrer": request_meta["referrer"],
            }
        )

        full_history = list(
            ChatMessage.objects
            .filter(session=session)
            .order_by("created_at")
            .values("role", "content")
        )
        recent_history = full_history[-6:]

        # 0) Frontend UI actions
        # This is handled by SmartChatIntentService to avoid duplicate intent logic in the view.
        ui_intent = SmartChatIntentService.classify_ui_action_intent(
            message=message,
            history=recent_history,
        )

        if ui_intent.intent == "contact_capture":
            payload = self._build_contact_capture_payload(
                session=session,
                message=message,
                user_message_id=user_message.id,
            )

            payload["intent_source"] = ui_intent.source
            payload["ui_intent_reason"] = ui_intent.reason
            payload["confidence"] = ui_intent.confidence

            return Response(
                self._with_optional_debug(
                    payload,
                    ui_intent={
                        "intent": ui_intent.intent,
                        "confidence": ui_intent.confidence,
                        "source": ui_intent.source,
                        "reason": ui_intent.reason,
                        "email": ui_intent.email,
                    },
                    retrieval_debug=[],
                    used_sources=[],
                    debug_history_count=len(full_history),
                    debug_recent_history=full_history[-4:],
                    debug_raw_message=message,
                ),
                status=status.HTTP_200_OK,
            )

        if ui_intent.intent == "send_history_email":
            payload = self._build_send_history_prompt_payload(
                session=session,
                message=message,
                initial_email=ui_intent.email,
            )

            payload["intent_source"] = ui_intent.source
            payload["ui_intent_reason"] = ui_intent.reason
            payload["confidence"] = ui_intent.confidence

            return Response(
                self._with_optional_debug(
                    payload,
                    ui_intent={
                        "intent": ui_intent.intent,
                        "confidence": ui_intent.confidence,
                        "source": ui_intent.source,
                        "reason": ui_intent.reason,
                        "email": ui_intent.email,
                    },
                    retrieval_debug=[],
                    used_sources=[],
                    debug_history_count=len(full_history),
                    debug_recent_history=full_history[-4:],
                    debug_raw_message=message,
                ),
                status=status.HTTP_200_OK,
            )

        # 1) Block Arabic-only queries
        if GeminiQueryRewriter.is_fully_arabic_query(message):
            blocked_reply = GeminiQueryRewriter.ENGLISH_ONLY_MESSAGE
            assistant_message = ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=blocked_reply,
                citations=[],
                confidence_score=1.0,
            )

            return Response(
                self._with_optional_debug({
                    "session_id": str(session.id),
                    "message_id": str(assistant_message.id),
                    "retrieval_query": None,
                    "rewrite_notes": "arabic_only_blocked",
                    "verdict": "unsupported_language",
                    "answer": blocked_reply,
                    "citations": [],
                    "applied_filters": None,
                    "answer_source": "language_guard",
                    "extractor_used": None,
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": "local_guard",
                    "fallback_used": False,
                    "generation_ok": True,
                    "safe_fallback": False,
                    "primary_meta": None,
                    "secondary_meta": None,
                },
                    retrieval_debug=[],
                    used_sources=[],
                    debug_history_count=len(full_history),
                    debug_recent_history=full_history[-4:],
                    debug_chunks_before_llm=[],
                    debug_prompt_chunks=[],
                    prompt_mode=None,
                    chunk_budget=None,
                ),
                status=status.HTTP_200_OK
            )

        # 2) Cheap local cleanup for routing only.
        # This does NOT call the LLM.
        # It fixes small typos/formatting so route classification has a cleaner message.
        route_message = GeminiQueryRewriter._local_rewrite(message)

        # 3) Route first before expensive rewrite/query planning.
        route_result = SmartChatIntentService.classify_question_route(
            message=route_message,
            history=recent_history,
        )

        # 4) Route using rewritten query, not raw message
        # route_result = SmartChatIntentService.classify_question_route(
        #     message=rewritten_query,
        #     history=recent_history,
        # )

        # Conversation memory route has the highest priority after blocking and rewriting, because if the user is asking a question that
        # can be answered from recent conversation, we want to answer that first before trying more complex retrieval and reasoning.
        # This also allows the chatbot to have more dynamic and context-aware conversations, as it can refer back to what was recently discussed.
        if route_result.route == "session_memory_question":
            memory_result = ConversationMemoryService.answer(
                session_id=session.id,
                user_question=message,
                current_user_message_id=user_message.id,
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=memory_result["answer"],
                citations=[],
                confidence_score=memory_result["confidence"],
            )

            payload = {
                "session_id": str(session.id),
                "message_id": str(assistant_message.id),
                "answer": memory_result["answer"],
                "citations": [],
                "confidence": memory_result["confidence"],
                "mode": "conversation_memory",
                "intent_source": route_result.source,
            }

            return Response(
                self._with_optional_debug(
                    payload,
                    route=route_result.route,
                    memory_source=memory_result["source"],
                    matched_messages=memory_result["matched_messages"],
                    debug_route_message=route_message,
                ),
                status=status.HTTP_200_OK,
            )

        # 4) Identity questions do not need rewrite/query planning or retrieval.
        if route_result.route == "identity_question":
            qa_result = ProfileQAService.answer_question(
                question=route_message,
                retrieval_query=route_message,
                question_route=route_result.route,
                history=full_history,
                query_plan={},
            )

            answer_text = qa_result.get(
                "answer") or "I’m Samah.ai’s portfolio assistant."

            assistant_message = ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=answer_text,
                citations=[],
                confidence_score=0.9,
            )

            return Response(
                self._with_optional_debug({
                    "session_id": str(session.id),
                    "message_id": str(assistant_message.id),
                    "retrieval_query": None,
                    "rewrite_notes": "rewrite_skipped_identity_question",
                    "debug_raw_message": message,
                    "debug_route_message": route_message,
                    "verdict": qa_result.get("verdict"),
                    "answer": answer_text,
                    "question_route": route_result.route,
                    "question_route_confidence": route_result.confidence,
                    "question_route_reason": route_result.raw_label,
                    "citations": [],
                    "applied_filters": None,
                    "answer_source": qa_result.get("meta", {}).get("answer_source"),
                    "extractor_used": None,
                    "model_used": qa_result.get("meta", {}).get("model_used"),
                    "tried_models": qa_result.get("meta", {}).get("tried_models", []),
                    "provider_used": qa_result.get("meta", {}).get("provider_used"),
                    "fallback_used": qa_result.get("meta", {}).get("fallback_used"),
                    "generation_ok": qa_result.get("meta", {}).get("generation_ok"),
                    "safe_fallback": qa_result.get("meta", {}).get("safe_fallback"),
                    "primary_meta": None,
                    "secondary_meta": None,
                },
                    retrieval_debug=[],
                    used_sources=[],
                    debug_history_count=len(full_history),
                    debug_recent_history=full_history[-4:],
                    debug_chunks_before_llm=[],
                    debug_prompt_chunks=[],
                    prompt_mode=None,
                    chunk_budget=None,
                ),
                status=status.HTTP_200_OK,
            )

        # 5) Conversational/meta/off-topic handling before expensive rewrite/query planning.
        # This prevents chatbot-behavior or off-topic questions from going into RAG.
        # Examples:
        # - "are you dumping the answers?"
        # - "tell me a joke"
        # - "why is your answer wrong?"
        conversational_result = SmartChatIntentService.detect_conversational_response(
            message=route_message,
            history=recent_history,
        )

        if conversational_result.handled:
            answer = SmartChatIntentService.generate_conversational_reply(
                message=route_message,
                category=conversational_result.category,
                history=recent_history,
            )

            assistant_message = ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=answer,
                citations=[],
                confidence_score=conversational_result.confidence,
            )

            return Response(
                self._with_optional_debug({
                    "session_id": str(session.id),
                    "message_id": str(assistant_message.id),
                    "retrieval_query": None,
                    "rewrite_notes": "rewrite_skipped_conversational_response",
                    "verdict": "conversational_response",
                    "answer": answer,
                    "question_route": route_result.route,
                    "question_route_confidence": route_result.confidence,
                    "question_route_reason": route_result.raw_label,
                    "citations": [],
                    "applied_filters": None,
                    "answer_source": "conversational_llm",
                    "extractor_used": None,
                    "model_used": None,
                    "tried_models": [],
                    "provider_used": None,
                    "fallback_used": False,
                    "generation_ok": True,
                    "safe_fallback": False,
                    "primary_meta": None,
                    "secondary_meta": None,
                    "confidence": conversational_result.confidence,
                    "mode": conversational_result.category,
                    "intent_source": conversational_result.source,
                },
                    retrieval_debug=[],
                    used_sources=[],
                    debug_history_count=len(full_history),
                    debug_recent_history=full_history[-4:],
                    debug_raw_message=message,
                    debug_route_message=route_message,
                    debug_rewritten_query=None,
                    debug_rewrite_meta=None,
                    debug_rewrite_debug=None,
                    debug_chunks_before_llm=[],
                    debug_prompt_chunks=[],
                    prompt_mode=None,
                    chunk_budget=None,
                ),
                status=status.HTTP_200_OK,
            )

        # 6) Quick social intents can skip rewrite/query planning.
        # Examples: hi, thanks, ok, wait, goodbye.
        if route_result.route == "general_question":
            quick = SmartChatIntentService.detect(route_message)

            if quick.handled:
                assistant_message = ChatMessage.objects.create(
                    session=session,
                    role="assistant",
                    content=quick.reply,
                    citations=[],
                    confidence_score=quick.confidence,
                )

                return Response(
                    self._with_optional_debug({
                        "session_id": str(session.id),
                        "message_id": str(assistant_message.id),
                        "answer": quick.reply,
                        "citations": [],
                        "confidence": quick.confidence,
                        "mode": quick.intent,
                        "intent_source": quick.source,
                        "retrieval_query": None,
                        "rewrite_notes": "rewrite_skipped_quick_intent",
                    },
                        retrieval_debug=[],
                        debug_history_count=len(full_history),
                        debug_recent_history=full_history[-4:],
                        debug_raw_message=message,
                        debug_route_message=route_message,
                    ),
                    status=status.HTTP_200_OK,
                )

        # 7) Expensive rewrite/query planning only for routes that need QA/retrieval.
        REWRITE_REQUIRED_ROUTES = {
            "profile_docs_question",
            "capability_inference_question",
            "general_question",
        }

        if route_result.route in REWRITE_REQUIRED_ROUTES:
            rewrite = GeminiQueryRewriter.rewrite_cached(
                user_query=message,
                history=recent_history,
            )

            rewritten_query = (rewrite.get("rewritten_query")
                               or route_message).strip() or route_message

            retrieval_query = (rewrite.get("retrieval_query")
                               or rewritten_query).strip() or rewritten_query

            query_plan = {
                "answer_type": rewrite.get("answer_type"),
                "preferred_document_types": rewrite.get("preferred_document_types") or [],
                "avoid_document_types": rewrite.get("avoid_document_types") or [],
                "needs_document_retrieval": rewrite.get("needs_document_retrieval", True),
                "source": rewrite.get("meta", {}).get("provider", "rewrite"),
                "retrieval_query": retrieval_query,
            }

            rewrite_notes = rewrite.get("notes")
        else:
            rewritten_query = route_message
            retrieval_query = route_message
            query_plan = {}
            rewrite_notes = "rewrite_skipped_for_route"

        # 8) Main QA orchestration
        qa_result = ProfileQAService.answer_question(
            question=rewritten_query,
            retrieval_query=retrieval_query,
            question_route=route_result.route,
            history=full_history,
            query_plan=query_plan,
        )

        verdict = qa_result.get("verdict", "not_enough_evidence")
        answer_text = qa_result.get(
            "answer") or "I don’t have enough evidence to answer that."
        bullets = qa_result.get("bullets") or []
        meta = qa_result.get("meta") or {}
        used_sources = qa_result.get("used_sources") or []
        retrieval_debug = qa_result.get("retrieval_debug") or []

        if bullets and not meta.get("safe_fallback"):
            answer_text += "\n\nKey points:\n- " + "\n- ".join(bullets)

        citations = []
        for src in used_sources:
            citations.append({
                "document_id": src.get("document_id"),
                "document_title": src.get("doc_title"),
                "chunk_id": src.get("chunk_id"),
                "chunk_index": src.get("chunk_index"),
                "section_title": None,
                "page_number": None,
                "distance": None,
            })

        confidence = 0.9 if verdict in {"yes", "no", "supported"} else 0.6

        if meta.get("safe_fallback"):
            confidence = 0.2
        elif meta.get("fallback_used"):
            confidence = min(confidence, 0.5)

        if meta.get("answer_source") == "deterministic_extractor":
            confidence = min(0.95, confidence +
                             float(meta.get("confidence_boost", 0.0)))

        if meta.get("answer_source") == "chatbot_profile":
            confidence = 0.85
        elif meta.get("answer_source") == "assisted_inference_llm":
            confidence = 0.70
        elif meta.get("answer_source") == "session_memory":
            confidence = max(confidence, route_result.confidence)

        assistant_message = ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=answer_text,
            citations=citations,
            confidence_score=confidence,
            metadata={
                "mode": meta.get("prompt_mode"),
                "question_route": route_result.route,
                "answer_source": meta.get("answer_source"),
                "model_used": meta.get("model_used"),
                "provider_used": meta.get("provider_used"),
                "fallback_used": meta.get("fallback_used"),
                "safe_fallback": meta.get("safe_fallback"),
                "retrieval_query": retrieval_query,
                "rewrite_notes": rewrite_notes,
            }
        )

        print("REWRITE DEBUG:", {
            "raw_message": message,
            "rewritten_query": rewritten_query,
            "rewrite_notes": rewrite_notes,
            "rewrite_meta": rewrite.get("meta") if "rewrite" in locals() else None,
            "answer": answer_text,
        })

        return Response(
            self._with_optional_debug({
                "visitor_email": session.visitor_email,
                "visitor_id": session.visitor_id,
                "session_id": str(session.id),
                "message_id": str(assistant_message.id),
                "retrieval_query": retrieval_query,
                "rewrite_notes": rewrite_notes,
                "debug_raw_message": message,
                "debug_rewritten_query": rewritten_query,
                "debug_rewrite_meta": rewrite.get("meta") if "rewrite" in locals() else None,
                "debug_rewrite_debug": rewrite.get("debug") if "rewrite" in locals() else None,
                "verdict": verdict,
                "answer": answer_text,
                "question_route": route_result.route,
                "question_route_confidence": route_result.confidence,
                "question_route_reason": route_result.raw_label,
                "citations": citations,
                "applied_filters": qa_result.get("applied_filters"),
                "answer_source": meta.get("answer_source"),
                "extractor_used": meta.get("extractor_used"),
                "model_used": meta.get("model_used"),
                "tried_models": meta.get("tried_models"),
                "provider_used": meta.get("provider_used"),
                "fallback_used": meta.get("fallback_used"),
                "generation_ok": meta.get("generation_ok"),
                "safe_fallback": meta.get("safe_fallback"),
                "primary_meta": meta.get("primary_meta"),
                "secondary_meta": meta.get("secondary_meta"),
            },
                retrieval_debug=retrieval_debug,
                used_sources=used_sources,
                debug_history_count=len(full_history),
                debug_recent_history=full_history[-4:],
                debug_chunks_before_llm=qa_result.get(
                    "debug_chunks_before_llm"),
                debug_prompt_chunks=meta.get("debug_prompt_chunks"),
                prompt_mode=meta.get("prompt_mode"),
                chunk_budget=meta.get("chunk_budget"),
            ),
            status=status.HTTP_200_OK
        )


class SendChatHistoryEmailAPIView(APIView):
    """
    Sends the current chatbot session history to the visitor's email using Resend.
    """

    permission_classes = [AllowAny]
    authentication_classes = [CsrfExemptSessionAuthentication]
    throttle_classes = [ContactRateThrottle]

    @staticmethod
    def _format_chat_history(session):
        """
        Convert all messages in the chat session into a readable email transcript.
        """

        messages = (
            ChatMessage.objects
            .filter(session=session)
            .order_by("created_at")
        )

        lines = [
            "Samah.ai Chat Conversation History",
            "=" * 40,
            "",
        ]

        for msg in messages:
            role = "Visitor" if msg.role == "user" else "Samah.ai Assistant"

            created_at = ""
            if getattr(msg, "created_at", None):
                created_at = msg.created_at.strftime("%Y-%m-%d %H:%M")

            content = (msg.content or "").strip()
            if not content:
                continue

            if created_at:
                lines.append(f"[{created_at}] {role}:")
            else:
                lines.append(f"{role}:")

            lines.append(content)
            lines.append("")

        lines.append("-" * 40)
        lines.append(
            "This conversation was sent from the Samah.ai portfolio chatbot.")

        return "\n".join(lines).strip()

    def post(self, request, *args, **kwargs):
        session_id = request.data.get("session_id")
        email = (request.data.get("email") or "").strip()

        if not session_id:
            return Response(
                {
                    "success": False,
                    "message": "Session ID is required.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not email:
            return Response(
                {
                    "success": False,
                    "message": "Email is required.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            validate_email(email)
        except ValidationError:
            return Response(
                {
                    "success": False,
                    "message": "Please enter a valid email address.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        session = ChatSession.objects.filter(
            id=session_id,
            is_active=True,
        ).first()

        if session is None:
            return Response(
                {
                    "success": False,
                    "message": "Chat session not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        history_text = self._format_chat_history(session)

        if not history_text:
            return Response(
                {
                    "success": False,
                    "message": "No conversation history found for this session.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            resend_result = send_chat_history_email(
                recipient_email=email,
                history_text=history_text,
            )

            return Response(
                {
                    "success": True,
                    "message": "Conversation history sent successfully.",
                    "provider": "resend",
                    "email_id": resend_result.get("id"),
                },
                status=status.HTTP_200_OK,
            )

        except Exception:
            logger.exception(
                "Failed to send chatbot conversation history email")

            return Response(
                {
                    "success": False,
                    "message": "Could not send the conversation history email right now. Please try again later.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def options(self, request, *args, **kwargs):
        return Response(status=status.HTTP_200_OK)

