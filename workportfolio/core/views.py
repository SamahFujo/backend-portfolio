from django.conf import settings
from .services.chatbot.answer_service import AnswerService
from .serializers import AskQuestionSerializer
from .models import ChatSession, ChatMessage, ProfileDocument
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.throttling import AnonRateThrottle
from rest_framework.authentication import SessionAuthentication, BasicAuthentication

from .serializers import StartProjectRequestSerializer, ProfileDocumentUploadSerializer, ProfileDocumentSerializer
from .services.resend_email import send_start_project_email

from .services.retrieval.reranked_vector_retrieval import RerankedVectorRetrievalService
from .services.chatbot.answer_service import AnswerService


from .services.documents.ingestion_service import IngestionService
from .services.chatbot.answer_composer import AnswerComposer

from .services.chatbot.gemini_query_rewriter import GeminiQueryRewriter
from .services.retrieval.reranked_vector_retrieval import RerankedVectorRetrievalService
from .services.chatbot.gemini_grounded_answerer import GeminiGroundedAnswerer


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Disable CSRF for this API endpoint (public portfolio form).
    """

    def enforce_csrf(self, request):
        # To disable CSRF checks for this view, we override this method to do nothing.
        return


"""
API views for the public "Start Project" form.
"""


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


"""
API views for the portfolio chatbot backend.
"""


class AskAboutMeAPIView(APIView):
    """
    Enterprise RAG endpoint:
    - Create / reuse session
    - Save user message
    - Retrieve evidence using vector search + reranking
    - Compose answer (generic RAG OR optional extractor formatting)
    - Save assistant message with citations and confidence
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
                return Response(
                    {"detail": "Session not found."},
                    status=status.HTTP_404_NOT_FOUND
                )
        else:
            session = ChatSession.objects.create()

        # 3) Save user message
        ChatMessage.objects.create(
            session=session,
            role="user",
            content=message,
        )

        # 1) Rewrite query for retrieval (typos, synonyms, expansion)
        rewrite = GeminiQueryRewriter.rewrite(message)
        retrieval_query = rewrite["rewritten_query"] or message

        # 2) Retrieve evidence using rewritten query
        chunks, retrieval_debug = RerankedVectorRetrievalService.retrieve_relevant_chunks(
            query=retrieval_query,
            candidate_k=settings.RERANK_CANDIDATE_K,
            top_n=settings.RERANK_TOP_N,
            min_rerank_score=settings.RERANK_MIN_SCORE,
        )

        # 3) Use Gemini to produce final grounded answer
        gemini_result = GeminiGroundedAnswerer.answer(question=message, evidence_chunks=chunks)

        # 4) Citations ONLY for the chunks Gemini says it used
        citations = []
        for i in gemini_result["used_chunk_indices"]:
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

        # 5) Build answer text with bullets
        answer_text = gemini_result["answer"]
        if gemini_result["bullets"]:
            answer_text += "\n\nKey points:\n- " + "\n- ".join(gemini_result["bullets"])

        # 6) Save assistant message
        assistant_message = ChatMessage.objects.create(
            session=session,
            role="assistant",
            content=answer_text,
            citations=citations,
            confidence_score=0.9 if gemini_result["verdict"] in {"yes", "no"} else 0.6,
        )

        # 7) Return response
        return Response(
            {
                "session_id": str(session.id),
                "message_id": str(assistant_message.id),
                "retrieval_query": retrieval_query,
                "rewrite_notes": rewrite.get("notes"),
                "verdict": gemini_result["verdict"],
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
