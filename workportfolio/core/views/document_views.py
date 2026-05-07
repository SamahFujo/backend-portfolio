"""
Profile document management API views.
"""

from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from core.models import ProfileDocument, DocumentChunk
from core.permissions import HasInternalAPIKey
from core.throttles import UploadRateThrottle
from core.serializers import (
    ProfileDocumentUploadSerializer,
    ProfileDocumentSerializer,
)
from core.services.documents.ingestion_service import IngestionService


class ProfileDocumentStatsAPIView(APIView):
    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    def get(self, request, doc_id, *args, **kwargs):
        doc = ProfileDocument.objects.filter(id=doc_id).first()

        if not doc:
            return Response(
                {"detail": "Document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        chunks_qs = DocumentChunk.objects.filter(document=doc)
        chunks_count = chunks_qs.count()
        embedded_count = chunks_qs.exclude(embedding__isnull=True).count()
        raw_len = len(doc.raw_text or "")

        return Response(
            {
                "document_id": str(doc.id),
                "title": doc.title,
                "document_type": doc.document_type,
                "status": doc.status,
                "is_active": getattr(doc, "is_active", True),
                "raw_text_length": raw_len,
                "chunks_count": chunks_count,
                "embedded_chunks_count": embedded_count,
            },
            status=status.HTTP_200_OK,
        )


class ProfileDocumentUploadAPIView(APIView):
    """
    Upload a profile-related document and process it immediately.
    """

    permission_classes = [AllowAny]
    throttle_classes = [UploadRateThrottle]
    admin_api_key = settings.ADMIN_API_KEY

    def post(self, request, *args, **kwargs):
        serializer = ProfileDocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        document = serializer.save(status="uploaded")
        IngestionService.process_document(document)

        return Response(
            ProfileDocumentSerializer(document).data,
            status=status.HTTP_201_CREATED,
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
