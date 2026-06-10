"""
Admin document views for chatbot knowledge management.

These APIs allow the admin panel to:
- upload documents
- process/reprocess documents
- inspect extracted text
- review chunks
- check quality logs
- approve or reject documents before chatbot activation
"""

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from core.models import (
    ProfileDocument,
    DocumentChunk,
)

from core.permissions import HasInternalAPIKey

from core.serializers import (
    ProfileDocumentUploadSerializer,
    ProfileDocumentListSerializer,
    ProfileDocumentDetailSerializer,
    ProfileDocumentUpdateSerializer,
    ProfileDocumentReplaceSerializer,
    ProfileDocumentInspectionSerializer,
    ProfileDocumentApproveSerializer,
    ProfileDocumentRejectSerializer,
    ProfileDocumentReprocessSerializer,
    ProfileDocumentMarkReviewedSerializer,
    DocumentChunkReviewSerializer,
    DocumentQualityCheckSerializer,
    DocumentProcessingLogSerializer,
)

from core.services.documents.ingestion_service import IngestionService
from core.services.knowledge_quality import DocumentApprovalService
from core.services.knowledge_quality.processing_log_service import log_info
from core.throttles import AdminAPIRateThrottle


class AdminDocumentListCreateAPIView(APIView):
    """
    GET:
        List chatbot knowledge documents.

    POST:
        Upload a new document.
        The document will be saved as uploaded, but not active.
    """

    permission_classes = [HasInternalAPIKey]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    throttle_classes = [AdminAPIRateThrottle]

    def get(self, request, *args, **kwargs):
        documents = ProfileDocument.objects.all().order_by(
            "priority",
            "-updated_at",
        )

        status_filter = request.query_params.get("status")
        document_type = request.query_params.get("document_type")
        search = request.query_params.get("search")

        if status_filter:
            documents = documents.filter(status=status_filter)

        if document_type:
            documents = documents.filter(document_type=document_type)

        if search:
            documents = documents.filter(title__icontains=search)

        serializer = ProfileDocumentListSerializer(
            documents,
            many=True,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        serializer = ProfileDocumentUploadSerializer(
            data=request.data,
            context={"request": request},
        )

        serializer.is_valid(raise_exception=True)
        document = serializer.save()

        log_info(
            document=document,
            step="document_uploaded",
            message="Document uploaded successfully and waiting for processing.",
            metadata={
                "filename": document.original_filename,
                "file_size": document.file_size,
            },
        )

        response_serializer = ProfileDocumentDetailSerializer(
            document,
            context={"request": request},
        )

        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


class AdminDocumentDetailAPIView(APIView):
    """
    GET:
        Retrieve one document details.

    PATCH:
        Update document metadata only.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def get_object(self, document_id):
        return get_object_or_404(ProfileDocument, id=document_id)

    def get(self, request, document_id, *args, **kwargs):
        document = self.get_object(document_id)

        serializer = ProfileDocumentDetailSerializer(
            document,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request, document_id, *args, **kwargs):
        document = self.get_object(document_id)

        serializer = ProfileDocumentUpdateSerializer(
            document,
            data=request.data,
            partial=True,
            context={"request": request},
        )

        serializer.is_valid(raise_exception=True)
        document = serializer.save()

        response_serializer = ProfileDocumentDetailSerializer(
            document,
            context={"request": request},
        )

        return Response(response_serializer.data, status=status.HTTP_200_OK)


class AdminDocumentProcessAPIView(APIView):
    """
    Process a newly uploaded document.

    This runs:
    - extraction
    - classification
    - chunking
    - embedding
    - quality checks
    - marks document ready_for_review
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def post(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        if document.status == "approved":
            return Response(
                {
                    "detail": "Approved documents cannot be processed again directly. Use replace or archive first."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        processed_document = IngestionService.process_document(document)

        serializer = ProfileDocumentDetailSerializer(
            processed_document,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class AdminDocumentReprocessAPIView(APIView):
    """
    Reprocess an existing non-approved document.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def post(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        serializer = ProfileDocumentReprocessSerializer(
            data=request.data,
            context={"document": document},
        )
        serializer.is_valid(raise_exception=True)

        processed_document = IngestionService.process_document(document)

        response_serializer = ProfileDocumentDetailSerializer(
            processed_document,
            context={"request": request},
        )

        return Response(response_serializer.data, status=status.HTTP_200_OK)


class AdminDocumentInspectionAPIView(APIView):
    """
    Return deep inspection data for extracted document text.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def get(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        serializer = ProfileDocumentInspectionSerializer(
            document,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class AdminDocumentChunksAPIView(APIView):
    """
    List chunks for a document.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def get(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        chunks = document.chunks.all().order_by("chunk_index")

        quality_status = request.query_params.get("quality_status")
        has_embedding = request.query_params.get("has_embedding")

        if quality_status:
            chunks = chunks.filter(quality_status=quality_status)

        if has_embedding in ["true", "false"]:
            chunks = chunks.filter(has_embedding=has_embedding == "true")

        serializer = DocumentChunkReviewSerializer(
            chunks,
            many=True,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class AdminDocumentChunkDetailAPIView(APIView):
    """
    PATCH:
        Edit one generated chunk manually.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def patch(self, request, chunk_id, *args, **kwargs):
        chunk = get_object_or_404(DocumentChunk, id=chunk_id)

        if chunk.document.status == "approved":
            return Response(
                {
                    "detail": "Chunks belonging to approved documents cannot be edited directly."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = DocumentChunkReviewSerializer(
            chunk,
            data=request.data,
            partial=True,
            context={"request": request},
        )

        serializer.is_valid(raise_exception=True)
        chunk = serializer.save()

        return Response(serializer.data, status=status.HTTP_200_OK)


class AdminDocumentQualityChecksAPIView(APIView):
    """
    List quality checks for one document.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def get(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        checks = document.quality_checks.all().order_by("-created_at")

        serializer = DocumentQualityCheckSerializer(
            checks,
            many=True,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class AdminDocumentLogsAPIView(APIView):
    """
    List processing logs for one document.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def get(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        logs = document.processing_logs.all().order_by("-created_at")

        serializer = DocumentProcessingLogSerializer(
            logs,
            many=True,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class AdminDocumentMarkReviewedAPIView(APIView):
    """
    Mark a document as manually reviewed by an admin.

    This confirms the admin inspected:
    - extracted text
    - chunks
    - embeddings
    - quality checks
    - metadata
    before approval.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def post(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        serializer = ProfileDocumentMarkReviewedSerializer(
            data=request.data,
            context={"document": document},
        )
        serializer.is_valid(raise_exception=True)

        document.is_reviewed = True
        document.reviewed_at = timezone.now()
        document.review_notes = serializer.validated_data.get(
            "review_notes", "")
        document.save(
            update_fields=[
                "is_reviewed",
                "reviewed_at",
                "review_notes",
                "updated_at",
            ]
        )

        log_info(
            document=document,
            step="document_marked_reviewed",
            message="Document was manually reviewed by admin.",
            metadata={
                "review_notes": document.review_notes,
            },
        )

        response_serializer = ProfileDocumentDetailSerializer(
            document,
            context={"request": request},
        )

        return Response(response_serializer.data, status=status.HTTP_200_OK)


class AdminDocumentApproveAPIView(APIView):
    """
    Approve document and activate it for chatbot retrieval.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def post(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        serializer = ProfileDocumentApproveSerializer(
            data=request.data,
            context={"document": document},
        )
        serializer.is_valid(raise_exception=True)

        service = DocumentApprovalService()

        approved_document = service.approve(
            document=document,
            admin_notes=serializer.validated_data.get("admin_notes", ""),
            force_approve=serializer.validated_data.get(
                "force_approve", False),
        )

        response_serializer = ProfileDocumentDetailSerializer(
            approved_document,
            context={"request": request},
        )

        return Response(response_serializer.data, status=status.HTTP_200_OK)


class AdminDocumentRejectAPIView(APIView):
    """
    Reject document and keep it inactive.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def post(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        serializer = ProfileDocumentRejectSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        service = DocumentApprovalService()

        rejected_document = service.reject(
            document=document,
            reason=serializer.validated_data["reason"],
        )

        response_serializer = ProfileDocumentDetailSerializer(
            rejected_document,
            context={"request": request},
        )

        return Response(response_serializer.data, status=status.HTTP_200_OK)


class AdminDocumentArchiveAPIView(APIView):
    """
    Archive document and deactivate all chunks.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def post(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        service = DocumentApprovalService()
        archived_document = service.archive(document)

        serializer = ProfileDocumentDetailSerializer(
            archived_document,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class AdminDocumentRestoreAPIView(APIView):
    """
    Restore an archived document.

    The document returns to ready_for_review and must be reviewed/approved again.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def post(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        service = DocumentApprovalService()
        restored_document = service.restore(document)

        serializer = ProfileDocumentDetailSerializer(
            restored_document,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)



class AdminDocumentPermanentDeleteAPIView(APIView):
    """
    Permanently delete a document.

    This action is irreversible.

    Allowed only for:
    - archived documents
    - rejected documents
    - failed documents
    - uploaded documents
    - legacy processed documents that are not active/approved/chatbot-enabled

    Approved active documents must be archived first.
    """

    permission_classes = [HasInternalAPIKey]
    throttle_classes = [AdminAPIRateThrottle]

    def delete(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        confirm_text = str(request.data.get("confirm_text", "")).strip()

        if confirm_text != "DELETE":
            return Response(
                {
                    "detail": "Permanent delete requires confirm_text='DELETE'."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        service = DocumentApprovalService()
        result = service.delete_permanently(document)

        return Response(result, status=status.HTTP_200_OK)

class AdminDocumentReplaceAPIView(APIView):
    """
    Replace document file.

    For now, this replaces the same document file and resets its status.
    Later we can upgrade this to create a separate version record.
    """

    permission_classes = [HasInternalAPIKey]
    parser_classes = [MultiPartParser, FormParser]
    throttle_classes = [AdminAPIRateThrottle]

    def post(self, request, document_id, *args, **kwargs):
        document = get_object_or_404(ProfileDocument, id=document_id)

        serializer = ProfileDocumentReplaceSerializer(
            data=request.data,
            context={"document": document},
        )
        serializer.is_valid(raise_exception=True)

        uploaded_file = serializer.validated_data["file"]

        document.file = uploaded_file
        document.original_filename = getattr(uploaded_file, "name", "")
        document.file_size = getattr(uploaded_file, "size", 0)
        document.mime_type = getattr(uploaded_file, "content_type", "") or ""
        document.file_hash = serializer.context.get("file_hash", "")

        document.raw_text = ""
        document.extracted_text_preview = ""
        document.status = "uploaded"
        document.quality_status = "pending"
        document.is_active = False
        document.is_approved = False
        document.is_available_for_chatbot = False
        document.is_reviewed = False
        document.reviewed_at = None
        document.review_notes = ""
        document.extraction_score = 0
        document.chunk_quality_score = 0
        document.embedding_quality_score = 0
        document.overall_quality_score = 0
        document.validation_summary = {}
        document.processing_metadata = {}

        replacement_note = serializer.validated_data.get(
            "replacement_note", "")

        if replacement_note:
            document.admin_notes = f"{document.admin_notes}\n\nReplacement note: {replacement_note}".strip(
            )

        document.save()

        document.chunks.all().delete()
        document.quality_checks.all().delete()

        log_info(
            document=document,
            step="document_replaced",
            message="Document file was replaced and reset for processing.",
            metadata={
                "replacement_note": replacement_note,
                "filename": document.original_filename,
            },
        )

        response_serializer = ProfileDocumentDetailSerializer(
            document,
            context={"request": request},
        )

        return Response(response_serializer.data, status=status.HTTP_200_OK)
