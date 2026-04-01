from .services.resend_contact_email import send_get_in_touch_email
from .serializers import GetInTouchSerializer
from rest_framework.authentication import BasicAuthentication, SessionAuthentication
from django.conf import settings
from .serializers import AskQuestionSerializer
from .models import ChatSession, ChatMessage, ProfileDocument, DocumentChunk
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.throttling import AnonRateThrottle
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from .serializers import StartProjectRequestSerializer, ProfileDocumentUploadSerializer, ProfileDocumentSerializer
from .services.resend_email import send_start_project_email
from .services.documents.ingestion_service import IngestionService
from .services.chatbot.hybrid_query_rewriter import GeminiQueryRewriter
from .services.chatbot.smart_chat_intents import SmartChatIntentService
from .services.chatbot.profile_qa_service import ProfileQAService


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
    authentication_classes = [
        CsrfExemptSessionAuthentication, BasicAuthentication]
    # optional, but nice if DRF throttling is configured
    throttle_classes = [AnonRateThrottle]

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
            # Keep response user-friendly, log detailed error server-side in production
            return Response(
                {
                    "success": False,
                    "message": "Could not send project request email right now. Please try again later.",
                    "error": str(e),  # You can remove this in production
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def options(self, request, *args, **kwargs):
        # Usually not needed explicitly, but safe if debugging preflight behavior
        return Response(status=status.HTTP_200_OK)


class GetInTouchView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [
        CsrfExemptSessionAuthentication, BasicAuthentication]
    throttle_classes = [AnonRateThrottle]

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
        except Exception as e:
            return Response(
                {
                    "success": False,
                    "message": "Could not send your message right now. Please try again later.",
                    "error": str(e),  # remove in production
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


""" API views for the portfolio chatbot backend."""


class ProfileDocumentStatsAPIView(APIView):
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
    """
    Enterprise RAG endpoint:
    - Create / reuse session
    - Save user message
    - Quick replies for greetings/thanks/help
    - Rewrite query for retrieval
    - Profile QA orchestration:
        * scope filters
        * retrieval
        * deterministic extractors
        * Gemini grounded fallback
    - Save assistant message
    """

    def post(self, request, *args, **kwargs):
        # 1) Validate input
        serializer = AskQuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data.get("session_id")
        message = serializer.validated_data["message"].strip()

        # 2) Get or create session
        if session_id:
            session = ChatSession.objects.filter(
                id=session_id, is_active=True).first()
            if session is None:
                return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        else:
            session = ChatSession.objects.create()

        # 3) Save user message
        ChatMessage.objects.create(
            session=session, role="user", content=message)

        # 4) Quick intents
        quick = SmartChatIntentService.detect(message)
        if quick.handled:
            assistant_message = ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=quick.reply,
                citations=[],
                confidence_score=quick.confidence,
            )
            return Response(
                {
                    "session_id": str(session.id),
                    "message_id": str(assistant_message.id),
                    "answer": quick.reply,
                    "citations": [],
                    "confidence": quick.confidence,
                    "mode": quick.intent,
                    "intent_source": quick.source,
                    "retrieval_debug": [],
                },
                status=status.HTTP_200_OK
            )

        # 5) Rewrite query (kept for debug/traceability)
        rewrite = GeminiQueryRewriter.rewrite_cached(message)
        retrieval_query = rewrite.get("rewritten_query") or message

        # 6) Main QA orchestration
        qa_result = ProfileQAService.answer_question(message)

        verdict = qa_result.get("verdict", "not_enough_evidence")
        answer_text = qa_result.get(
            "answer") or "I don’t have enough evidence to answer that."
        bullets = qa_result.get("bullets") or []

        if bullets:
            answer_text += "\n\nKey points:\n- " + "\n- ".join(bullets)

        used_sources = qa_result.get("used_sources") or []
        retrieval_debug = qa_result.get("retrieval_debug") or []
        meta = qa_result.get("meta") or {}

        # 7) Convert used_sources into response citations
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

        # 8) Save assistant message
        confidence = 0.9 if verdict in {"yes", "no"} else 0.6
        if meta.get("answer_source") == "deterministic_extractor":
            confidence = min(0.95, confidence +
            float(meta.get("confidence_boost", 0.0)))

        assistant_message = ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=answer_text,
            citations=citations,
            confidence_score=confidence,
        )

        # 9) Return response
        return Response(
            {
                "session_id": str(session.id),
                "message_id": str(assistant_message.id),
                "retrieval_query": retrieval_query,
                "rewrite_notes": rewrite.get("notes"),
                "verdict": verdict,
                "answer": answer_text,
                "citations": citations,
                "retrieval_debug": retrieval_debug,
                "used_sources": used_sources,
                "applied_filters": qa_result.get("applied_filters"),
                "answer_source": meta.get("answer_source"),
                "extractor_used": meta.get("extractor_used"),
                "gemini_model_used": meta.get("model_used"),
                "gemini_tried_models": meta.get("tried_models"),
            },
            status=status.HTTP_200_OK
        )


class ProfileDocumentUploadAPIView(APIView):
    """
    Upload a profile-related document and process it immediately.
    """

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

    def get(self, request, *args, **kwargs):
        documents = ProfileDocument.objects.all().order_by("-created_at")
        serializer = ProfileDocumentSerializer(documents, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
