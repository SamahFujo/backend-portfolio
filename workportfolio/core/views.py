import re

from django.conf import settings
from .services.resend_contact_email import send_get_in_touch_email
from .serializers import GetInTouchSerializer
from .serializers import AskQuestionSerializer
from .models import ChatSession, ChatMessage, ProfileDocument, DocumentChunk
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.throttling import AnonRateThrottle
from rest_framework.authentication import SessionAuthentication
from .serializers import StartProjectRequestSerializer, ProfileDocumentUploadSerializer, ProfileDocumentSerializer
from .services.resend_email import send_start_project_email
from .services.documents.ingestion_service import IngestionService
from .services.chatbot.hybrid_query_rewriter import GeminiQueryRewriter
from .services.chatbot.smart_chat_intents import SmartChatIntentService
from .services.chatbot.profile_qa_service import ProfileQAService
from .permissions import HasInternalAPIKey
from .throttles import ChatRateThrottle, ContactRateThrottle, UploadRateThrottle
from .services.chatbot.conversation_memory_service import ConversationMemoryService
import logging
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
logger = logging.getLogger(__name__)
from .services.resend_email import send_start_project_email, send_chat_history_email

class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Disable CSRF for this API endpoint (public portfolio form). this is safe because we are not using session authentication for any sensitive operations, and we have other protections in place (throttling, CORS, etc). It allows the public form to submit without needing a CSRF token.
    """

    def enforce_csrf(self, request):
        # To disable CSRF checks for this view, we override this method to do nothing.
        return


"""API views for the public "Start Project" form."""


class StartProjectRequestView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [CsrfExemptSessionAuthentication]
    throttle_classes = [ContactRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = StartProjectRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            resend_result = send_start_project_email(serializer.validated_data)
            return Response(
                {
                    "success": True,
                    "message": "Project request sent successfully.",
                    "provider": "resend",
                    "email_id": resend_result.get("id"),
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            logger.exception("Failed to send project request email")
            return Response(
                {
                    "success": False,
                    "message": "Could not send project request email right now. Please try again later.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def options(self, request, *args, **kwargs):
        # Usually not needed explicitly, but safe if debugging preflight behavior
        return Response(status=status.HTTP_200_OK)
    
    



class GetInTouchView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [CsrfExemptSessionAuthentication]
    throttle_classes = [ContactRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = GetInTouchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            result = send_get_in_touch_email(serializer.validated_data)
            return Response(
                {
                    "success": True,
                    "message": "Message sent successfully.",
                    "provider": "resend",
                    "email_id": result.get("id"),
                },
                status=status.HTTP_200_OK,
            )
        except Exception:
            logger.exception("Failed to send get-in-touch email")
            return Response(
                {
                    "success": False,
                    "message": "Could not send your message right now. Please try again later.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


""" API views for the portfolio chatbot backend."""


class ProfileDocumentStatsAPIView(APIView):
    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    def get(self, request, doc_id, *args, **kwargs):
        doc = ProfileDocument.objects.filter(id=doc_id).first()
        if not doc:
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        chunks_qs = DocumentChunk.objects.filter(document=doc)
        chunks_count = chunks_qs.count()
        embedded_count = chunks_qs.exclude(embedding__isnull=True).count()
        raw_len = len(doc.raw_text or "")

        return Response({
            "document_id": str(doc.id),
            "title": doc.title,
            "document_type": doc.document_type,
            "status": doc.status,
            "is_active": getattr(doc, "is_active", True),
            "raw_text_length": raw_len,
            "chunks_count": chunks_count,
            "embedded_chunks_count": embedded_count,
        }, status=status.HTTP_200_OK)


"""API views for the portfolio chatbot backend."""
class AskAboutMeAPIView(APIView):

    throttle_classes = [ChatRateThrottle]

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
    def _is_contact_capture_request(message: str) -> bool:
        """
        Detect when the visitor wants to share their own details
        so Samah can contact them.

        This is different from:
        - "How can I contact Samah?"
        - "What is Samah's email?"
        Those are profile/contact-information questions.

        This handles:
        - "I want to provide my details so Samah can contact me"
        - "Can I leave my phone number?"
        - "I want Samah to call me"
        - "How can I send my contact details?"
        """
        q = (message or "").strip().lower()

        if not q:
            return False

        visitor_detail_markers = [
            "my details",
            "my contact",
            "my information",
            "my info",
            "my phone",
            "my mobile",
            "my number",
            "my email",
            "leave my details",
            "leave my contact",
            "send my details",
            "send my contact",
            "share my details",
            "share my contact",
            "provide my details",
            "provide my contact",
            "fill my details",
            "submit my details",
        ]

        samah_followup_markers = [
            "contact me",
            "call me",
            "email me",
            "reach me",
            "get back to me",
            "follow up with me",
            "samah can contact me",
            "samah to contact me",
            "samah call me",
            "samah to call me",
            "samah reach me",
        ]

        form_markers = [
            "contact form",
            "open form",
            "fill form",
            "send message",
            "submit message",
        ]

        # Strong intent: user explicitly wants to provide their own details.
        if any(marker in q for marker in visitor_detail_markers):
            return True

        # Strong intent: user asks Samah to contact them.
        if any(marker in q for marker in samah_followup_markers):
            return True

        # Form-based intent.
        if any(marker in q for marker in form_markers) and any(
            word in q for word in ["contact", "details", "message", "samah"]
        ):
            return True

        return False

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
    def _extract_email_from_text(message: str) -> str | None:
        """
        Extract an email address if the user already typed it in the chat.
        Example:
        - "send the chat to me at test@example.com"
        """
        text = (message or "").strip()

        match = re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            text,
        )

        return match.group(0) if match else None

    @staticmethod
    def _is_send_history_request(message: str) -> bool:
        """
        Detect when the visitor wants the chatbot conversation history
        sent to their email.

        Examples:
        - send this conversation to my email
        - email me this chat
        - send chat history
        - can you send me our conversation
        - send transcript to me
        """
        q = (message or "").strip().lower()

        if not q:
            return False

        history_markers = [
            "conversation history",
            "chat history",
            "this conversation",
            "our conversation",
            "this chat",
            "chat transcript",
            "conversation transcript",
            "transcript",
            "history",
        ]

        send_markers = [
            "send",
            "email",
            "mail",
            "forward",
            "share",
        ]

        user_markers = [
            "to me",
            "my email",
            "me through email",
            "through email",
            "by email",
            "via email",
            "to my mail",
        ]

        has_history = any(marker in q for marker in history_markers)
        has_send = any(marker in q for marker in send_markers)
        has_user_target = any(marker in q for marker in user_markers)

        # Strong direct phrases
        direct_phrases = [
            "email me this chat",
            "email me the chat",
            "send me this chat",
            "send me the chat",
            "send me the conversation",
            "send conversation to my email",
            "send the conversation to my email",
            "send chat history",
            "send the chat history",
        ]

        if any(phrase in q for phrase in direct_phrases):
            return True

        return has_history and has_send and has_user_target

    @staticmethod
    def _build_send_history_prompt_payload(session, message: str):
        initial_email = AskAboutMeAPIView._extract_email_from_text(message)

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
            "intent_source": "history_email_rule",
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
        message = serializer.validated_data["message"].strip()

        if session_id:
            session = ChatSession.objects.filter(
                id=session_id, is_active=True).first()
            if session is None:
                return Response(
                    {"detail": "Session not found."},
                    status=status.HTTP_404_NOT_FOUND
                )
        else:
            session = ChatSession.objects.create()

        user_message = ChatMessage.objects.create(
            session=session,
            role="user",
            content=message
        )

        full_history = list(
            ChatMessage.objects
            .filter(session=session)
            .order_by("created_at")
            .values("role", "content")
        )
        recent_history = full_history[-6:]

        # 0) Contact capture UI action
        # This does not replace your profile QA workflow.
        # It only catches cases where the visitor wants to share their own details.
        if self._is_contact_capture_request(message):
            payload = self._build_contact_capture_payload(
                session=session,
                message=message,
                user_message_id=user_message.id,
            )

            return Response(
                self._with_optional_debug(
                    payload,
                    retrieval_debug=[],
                    used_sources=[],
                    debug_history_count=len(full_history),
                    debug_recent_history=full_history[-4:],
                    debug_raw_message=message,
                ),
                status=status.HTTP_200_OK,
            )

        # 0.5) Send conversation history by email UI action
        if self._is_send_history_request(message):
            payload = self._build_send_history_prompt_payload(
                session=session,
                message=message,
            )

            return Response(
                self._with_optional_debug(
                    payload,
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

        # 2) Rewrite first
        rewrite = GeminiQueryRewriter.rewrite_cached(
            user_query=message,
            history=recent_history,
        )
        rewritten_query = (rewrite.get("rewritten_query")
                           or message).strip() or message
        retrieval_query = rewritten_query
        rewrite_notes = rewrite.get("notes")

        # 3) Route using rewritten query, not raw message
        route_result = SmartChatIntentService.classify_question_route(
            message=rewritten_query,
            history=recent_history,
        )

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
                ),
                status=status.HTTP_200_OK,
            )
        # 4) Quick social intents only if this is not a stronger conversational route
        if route_result.route in {"general_question", "identity_question"}:
            quick = SmartChatIntentService.detect(rewritten_query)
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
                        "retrieval_query": retrieval_query,
                        "rewrite_notes": rewrite_notes,
                    },
                        retrieval_debug=[],
                        debug_history_count=len(full_history),
                        debug_recent_history=full_history[-4:],
                        debug_raw_message=message,
                        debug_rewritten_query=rewritten_query,
                        debug_rewrite_meta=rewrite.get("meta"),
                        debug_rewrite_debug=rewrite.get("debug"),
                    ),
                    status=status.HTTP_200_OK
                )

        # 7) Main QA orchestration
        qa_result = ProfileQAService.answer_question(
            question=rewritten_query,
            retrieval_query=retrieval_query,
            question_route=route_result.route,
            history=full_history,
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
        )

        print("REWRITE DEBUG:", {
            "raw_message": message,
            "rewritten_query": rewritten_query,
            "rewrite_notes": rewrite_notes,
            "rewrite_meta": rewrite.get("meta"),
            "answer": answer_text,
        })

        return Response(
            self._with_optional_debug({
                "session_id": str(session.id),
                "message_id": str(assistant_message.id),
                "retrieval_query": retrieval_query,
                "rewrite_notes": rewrite_notes,
                "debug_raw_message": message,
                "debug_rewritten_query": rewritten_query,
                "debug_rewrite_meta": rewrite.get("meta"),
                "debug_rewrite_debug": rewrite.get("debug"),
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
    Sends the current chatbot session history to the visitor's email.

    Expected payload:
    {
        "session_id": "...",
        "email": "visitor@example.com"
    }
    """

    @staticmethod
    def _format_chat_history(session: ChatSession) -> str:
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
                {"detail": "Session ID is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not email:
            return Response(
                {"detail": "Email is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            validate_email(email)
        except ValidationError:
            return Response(
                {"detail": "Please enter a valid email address."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        session = ChatSession.objects.filter(
            id=session_id,
            is_active=True,
        ).first()

        if session is None:
            return Response(
                {"detail": "Chat session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        history_text = self._format_chat_history(session)

        if not history_text:
            return Response(
                {"detail": "No conversation history found for this session."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        subject = "Your Samah.ai Chat Conversation History"

        try:
            send_mail(
                subject=subject,
                message=history_text,
                from_email=getattr(settings, "DEFAULT_FROM_EMAIL", None),
                recipient_list=[email],
                fail_silently=False,
            )
        except Exception as exc:
            if settings.DEBUG:
                return Response(
                    {
                        "detail": "Could not send the conversation history email.",
                        "error": str(exc),
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            return Response(
                {"detail": "Could not send the conversation history email."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "detail": "Conversation history sent successfully.",
                "email": email,
            },
            status=status.HTTP_200_OK,
        )




class ProfileDocumentUploadAPIView(APIView):
    """
    Upload a profile-related document and process it immediately.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [UploadRateThrottle]
    admin_api_key = settings.ADMIN_API_KEY

    def post(self, request, *args, **kwargs):
        serializer = ProfileDocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        document = serializer.save(status="uploaded")
        IngestionService.process_document(document)

        return Response(
            ProfileDocumentSerializer(document).data,
            status=status.HTTP_201_CREATED
        )


class ProfileDocumentListAPIView(APIView):
    """
    List all uploaded profile documents.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    def get(self, request, *args, **kwargs):
        documents = ProfileDocument.objects.all().order_by("-created_at")
        serializer = ProfileDocumentSerializer(documents, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
