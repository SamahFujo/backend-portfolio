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

from .services.retrieval.reranked_vector_retrieval import RerankedVectorRetrievalService



from .services.documents.ingestion_service import IngestionService
import re
from difflib import SequenceMatcher

from .services.chatbot.gemini_query_rewriter import GeminiQueryRewriter
from .services.retrieval.reranked_vector_retrieval import RerankedVectorRetrievalService
from .services.chatbot.gemini_grounded_answerer import GeminiGroundedAnswerer
from .services.chatbot.smart_chat_intents import SmartChatIntentService


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Disable CSRF for this API endpoint (public portfolio form).
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
    - Quick replies for greetings/thanks/help (NO Gemini)
    - Rewrite query for retrieval (Gemini optional + cached + safe fallback)
    - Retrieve evidence using vector search + reranking
    - Gemini grounded answer (safe fallback if quota exceeded)
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
            session = ChatSession.objects.filter(id=session_id, is_active=True).first()
            if session is None:
                return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        else:
            session = ChatSession.objects.create()

        # 3) Save user message
        ChatMessage.objects.create(session=session, role="user", content=message)

        # 4) Quick intents (greeting, goodbye, thanks, help) -> NEVER call Gemini here
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

        # 5) Rewrite query for retrieval (SAFE + CACHED + fallback to local rewrite)
        # IMPORTANT: use rewrite_cached, not rewrite
        rewrite = GeminiQueryRewriter.rewrite_cached(message)
        retrieval_query = rewrite.get("rewritten_query") or message
        

        # # Remove near-duplicate chunks to reduce repetition in output.
        # def normalize_text(t: str) -> str:
        #     t = (t or "").lower()
        #     t = re.sub(r"\s+", " ", t).strip()
        #     return t

        # def dedupe_chunks(chunks, similarity_threshold: float = 0.92):
        #     """
        #     Remove near-duplicate chunks to reduce repetition in output.
        #     """
        #     kept = []
        #     kept_norm = []
        #     for c in chunks:
        #         norm = normalize_text(c.content)
        #         is_dup = False
        #         for kn in kept_norm:
        #             if SequenceMatcher(None, norm, kn).ratio() >= similarity_threshold:
        #                 is_dup = True
        #                 break
        #         if not is_dup:
        #             kept.append(c)
        #             kept_norm.append(norm)
        #     return kept

        # 6) Retrieve evidence using rewritten query
        chunks, retrieval_debug = RerankedVectorRetrievalService.retrieve_relevant_chunks(
            query=retrieval_query,
            candidate_k=settings.RERANK_CANDIDATE_K,
            top_n=settings.RERANK_TOP_N,
            min_rerank_score=settings.RERANK_MIN_SCORE,
        )

        # 7) Gemini final grounded answer (SAFE fallback on quota exhaustion)
        gemini_result = GeminiGroundedAnswerer.answer(question=message, evidence_chunks=chunks)

        # 8) Citations ONLY for chunks Gemini used (safe index handling)
        citations = []
        used_indices = gemini_result.get("used_chunk_indices") or []
        for i in used_indices:
            if not isinstance(i, int) or i < 0 or i >= len(chunks):
                continue
            c = chunks[i]
            citations.append({
                "document_id": str(c.document.id),
                "document_title": c.document.title,
                "chunk_id": str(c.id),
                "chunk_index": c.chunk_index,
                "section_title": c.section_title,
                "page_number": c.page_number,
                "distance": float(getattr(c, "distance", 0.0)) if getattr(c, "distance", None) is not None else None,
            })

        # 9) Build answer text with bullets
        answer_text = gemini_result.get("answer") or "I don’t have enough evidence to answer that."
        bullets = gemini_result.get("bullets") or []
        if bullets:
            answer_text += "\n\nKey points:\n- " + "\n- ".join(bullets)

        # 10) Save assistant message
        verdict = gemini_result.get("verdict", "not_enough_evidence")
        assistant_message = ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=answer_text,
            citations=citations,
            confidence_score=0.9 if verdict in {"yes", "no"} else 0.6,
        )

        # 11) Return response
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
