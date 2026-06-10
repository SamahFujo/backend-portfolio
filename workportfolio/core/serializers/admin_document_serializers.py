import hashlib
import json
import re
from pathlib import Path

from django.conf import settings
from rest_framework import serializers

from ..models import (
    ProfileDocument,
    DocumentChunk,
    DocumentQualityCheck,
    DocumentProcessingLog,
    ProfileDocumentVersion,
)


## this is an old serialaizers need to be removed in the future when we are 
# sure that they are not used anywhere anymore.
#--------------------------------------------------------------------------------
class ProfileDocumentSerializer(serializers.ModelSerializer):
    """
    Backward-compatible serializer for existing document views.
    """

    class Meta:
        model = ProfileDocument
        fields = "__all__"


class DocumentChunkSerializer(serializers.ModelSerializer):
    """
    Backward-compatible serializer for existing chunk views.
    """

    class Meta:
        model = DocumentChunk
        fields = "__all__"




class DocumentFileValidationMixin:
    """
    Shared validation logic for uploaded chatbot knowledge documents.
    """

#--------------------------------------------------------------------------------




class DocumentFileValidationMixin:
    """
    Shared validation logic for uploaded chatbot knowledge documents.
    """

    DEFAULT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
    DEFAULT_ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
        # Some browsers/dev tools send this for files.
        "application/octet-stream",
    }

    DANGEROUS_EXTENSIONS = {
        ".exe",
        ".bat",
        ".cmd",
        ".sh",
        ".js",
        ".html",
        ".htm",
        ".php",
        ".zip",
        ".rar",
        ".7z",
    }

    def validate_document_file(self, file):
        if not file:
            raise serializers.ValidationError("Please upload a document file.")

        original_name = getattr(file, "name", "") or ""
        extension = Path(original_name).suffix.lower()

        allowed_extensions = getattr(
            settings,
            "ALLOWED_DOCUMENT_EXTENSIONS",
            self.DEFAULT_ALLOWED_EXTENSIONS,
        )

        max_size = getattr(
            settings,
            "MAX_DOCUMENT_UPLOAD_SIZE",
            10 * 1024 * 1024,
        )

        if not original_name.strip():
            raise serializers.ValidationError(
                "Uploaded file must have a valid filename.")

        if extension in self.DANGEROUS_EXTENSIONS:
            raise serializers.ValidationError(
                "This file type is not allowed for security reasons."
            )

        if extension not in allowed_extensions:
            allowed = ", ".join(sorted(allowed_extensions))
            raise serializers.ValidationError(
                f"Unsupported file type. Allowed types are: {allowed}."
            )

        if file.size <= 0:
            raise serializers.ValidationError("Uploaded file is empty.")

        if file.size > max_size:
            max_size_mb = max_size // (1024 * 1024)
            raise serializers.ValidationError(
                f"File is too large. Maximum allowed size is {max_size_mb} MB."
            )

        content_type = getattr(file, "content_type", "") or ""

        allowed_content_types = getattr(
            settings,
            "ALLOWED_DOCUMENT_CONTENT_TYPES",
            self.DEFAULT_ALLOWED_CONTENT_TYPES,
        )

        if content_type and content_type not in allowed_content_types:
            raise serializers.ValidationError(
                "Invalid file content type. Please upload a valid PDF, DOCX, TXT, or Markdown file."
            )

        return file

    def calculate_file_hash(self, file):
        """
        Calculates SHA256 hash for duplicate detection.
        Resets file pointer after reading.
        """

        sha256_hash = hashlib.sha256()

        current_position = file.tell() if hasattr(file, "tell") else 0

        for chunk in file.chunks():
            sha256_hash.update(chunk)

        if hasattr(file, "seek"):
            file.seek(current_position)

        return sha256_hash.hexdigest()


class ProfileDocumentUploadSerializer(
    DocumentFileValidationMixin,
    serializers.ModelSerializer,
):
    """
    Admin serializer for uploading a new chatbot knowledge document.

    Validation covered:
    - Required title
    - Valid document type
    - Safe file extension
    - Safe MIME type
    - File size limit
    - Duplicate file detection using SHA256 hash
    - Tags validation
    - Safe default status values
    """

    class Meta:
        model = ProfileDocument
        fields = [
            "id",
            "title",
            "file",
            "document_type",
            "source_label",
            "priority",
            "tags",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
        ]

        extra_kwargs = {
            "title": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Document title is required.",
                    "blank": "Document title is required.",
                },
            },
            "file": {
                "required": True,
                "allow_null": False,
                "error_messages": {
                    "required": "Please upload a document file.",
                },
            },
            "document_type": {
                "required": False,
                "allow_blank": True,
                "allow_null": True,
            },
            "source_label": {
                "required": False,
                "allow_blank": True,
                "allow_null": True,
            },
            "tags": {
                "required": False,
                "allow_null": True,
            },
        }

    def validate_title(self, value):
        value = value.strip()

        if len(value) < 3:
            raise serializers.ValidationError(
                "Document title is too short. Please write at least 3 characters."
            )

        if len(value) > 255:
            raise serializers.ValidationError(
                "Document title is too long. Please keep it under 255 characters."
            )

        return value

    def validate_document_type(self, value):
        value = (value or "").strip()

        if not value:
            return None

        allowed_types = {
            "cv",
            "certificate",
            "project",
            "experience",
            "profile",
            "recommendation",
            "research",
            "other",
        }

        normalized_value = value.lower()

        if normalized_value not in allowed_types:
            raise serializers.ValidationError(
                "Invalid document type. Choose one of: CV, certificate, project, experience, profile, recommendation, research, or other."
            )

        return normalized_value

    def validate_source_label(self, value):
        value = (value or "").strip()

        if value and len(value) > 255:
            raise serializers.ValidationError(
                "Source label is too long. Please keep it under 255 characters."
            )

        return value or None

    def validate_priority(self, value):
        if value is None:
            return 5

        if value < 1 or value > 10:
            raise serializers.ValidationError(
                "Priority must be between 1 and 10. Use 1 for highest priority."
            )

        return value

    def validate_tags(self, value):
        """
        Supports:
        - JSON list from application/json
        - JSON string from multipart/form-data
        - Null / empty value
        """

        if value in [None, "", []]:
            return []

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise serializers.ValidationError(
                    "Tags must be a valid list. Example: ['cv', 'django', 'ai']."
                )

        if not isinstance(value, list):
            raise serializers.ValidationError(
                "Tags must be a list of text values.")

        cleaned_tags = []
        seen = set()

        for tag in value:
            if not isinstance(tag, str):
                raise serializers.ValidationError(
                    "Each tag must be written as text.")

            clean_tag = tag.strip().lower()

            if not clean_tag:
                continue

            if len(clean_tag) > 40:
                raise serializers.ValidationError(
                    "Each tag must be under 40 characters."
                )

            if not re.match(r"^[a-z0-9\s\-_/#]+$", clean_tag):
                raise serializers.ValidationError(
                    f"Invalid tag '{tag}'. Tags can only contain letters, numbers, spaces, hyphens, underscores, slash, or #."
                )

            if clean_tag not in seen:
                seen.add(clean_tag)
                cleaned_tags.append(clean_tag)

        if len(cleaned_tags) > 20:
            raise serializers.ValidationError(
                "Too many tags. Please keep tags under 20 items."
            )

        return cleaned_tags

    def validate_file(self, value):
        value = self.validate_document_file(value)

        file_hash = self.calculate_file_hash(value)

        existing_document = ProfileDocument.objects.filter(
            file_hash=file_hash
        ).exclude(
            status__in=["rejected", "archived"]
        ).first()

        if existing_document:
            raise serializers.ValidationError(
                f"This file already exists as '{existing_document.title}'. Please replace the existing document instead of uploading a duplicate."
            )

        self.context["file_hash"] = file_hash

        return value

    def create(self, validated_data):
        file = validated_data.get("file")

        validated_data["original_filename"] = getattr(file, "name", "")
        validated_data["file_size"] = getattr(file, "size", 0)
        validated_data["mime_type"] = getattr(file, "content_type", "") or ""
        validated_data["file_hash"] = self.context.get("file_hash", "")

        validated_data["status"] = "uploaded"
        validated_data["quality_status"] = "pending"
        validated_data["is_active"] = False
        validated_data["is_approved"] = False
        validated_data["is_available_for_chatbot"] = False

        return super().create(validated_data)


class ProfileDocumentListSerializer(serializers.ModelSerializer):
    """
    Lightweight admin serializer for listing chatbot knowledge documents.
    """

    chunks_count = serializers.SerializerMethodField()
    embedded_chunks_count = serializers.SerializerMethodField()
    active_chunks_count = serializers.SerializerMethodField()
    failed_chunks_count = serializers.SerializerMethodField()
    missing_embeddings_count = serializers.SerializerMethodField()
    quality_issues_count = serializers.SerializerMethodField()

    class Meta:
        model = ProfileDocument
        fields = [
            "id",
            "title",
            "document_type",
            "source_label",
            "status",
            "quality_status",
            "is_active",
            "is_approved",
            "is_available_for_chatbot",
            "priority",
            "tags",
            "original_filename",
            "file_size",
            "mime_type",
            "extraction_score",
            "chunk_quality_score",
            "embedding_quality_score",
            "overall_quality_score",
            "chunks_count",
            "embedded_chunks_count",
            "active_chunks_count",
            "failed_chunks_count",
            "missing_embeddings_count",
            "quality_issues_count",
            "created_at",
            "updated_at",
            "processed_at",
            "approved_at",
            "rejected_at",
        ]

    def get_chunks_count(self, obj):
        return obj.chunks.count()

    def get_embedded_chunks_count(self, obj):
        return obj.chunks.filter(has_embedding=True).count()

    def get_active_chunks_count(self, obj):
        return obj.chunks.filter(is_active=True).count()

    def get_failed_chunks_count(self, obj):
        return obj.chunks.filter(quality_status="failed").count()

    def get_missing_embeddings_count(self, obj):
        return obj.chunks.filter(has_embedding=False).count()

    def get_quality_issues_count(self, obj):
        return obj.quality_checks.exclude(check_status="passed").count()


class ProfileDocumentDetailSerializer(serializers.ModelSerializer):
    """
    Detailed admin serializer for one chatbot knowledge document.
    """

    file_url = serializers.SerializerMethodField()
    chunks_count = serializers.SerializerMethodField()
    embedded_chunks_count = serializers.SerializerMethodField()
    missing_embeddings_count = serializers.SerializerMethodField()
    critical_issues_count = serializers.SerializerMethodField()
    warnings_count = serializers.SerializerMethodField()

    class Meta:
        model = ProfileDocument
        fields = [
            "id",
            "title",
            "file",
            "file_url",
            "document_type",
            "source_label",
            "raw_text",
            "extracted_text_preview",
            "status",
            "quality_status",
            "is_active",
            "is_approved",
            "is_available_for_chatbot",
            "priority",
            "tags",
            "original_filename",
            "file_size",
            "file_hash",
            "mime_type",
            "extraction_score",
            "chunk_quality_score",
            "embedding_quality_score",
            "overall_quality_score",
            "validation_summary",
            "processing_metadata",
            "admin_notes",
            "rejection_reason",
            "chunks_count",
            "embedded_chunks_count",
            "missing_embeddings_count",
            "critical_issues_count",
            "warnings_count",
            "created_at",
            "updated_at",
            "processed_at",
            "approved_at",
            "rejected_at",
        ]

        read_only_fields = [
            "id",
            "file",
            "file_url",
            "raw_text",
            "extracted_text_preview",
            "status",
            "quality_status",
            "is_active",
            "is_approved",
            "is_available_for_chatbot",
            "original_filename",
            "file_size",
            "file_hash",
            "mime_type",
            "extraction_score",
            "chunk_quality_score",
            "embedding_quality_score",
            "overall_quality_score",
            "validation_summary",
            "processing_metadata",
            "rejection_reason",
            "created_at",
            "updated_at",
            "processed_at",
            "approved_at",
            "rejected_at",
        ]

    def get_file_url(self, obj):
        request = self.context.get("request")

        if obj.file:
            url = obj.file.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_chunks_count(self, obj):
        return obj.chunks.count()

    def get_embedded_chunks_count(self, obj):
        return obj.chunks.filter(has_embedding=True).count()

    def get_missing_embeddings_count(self, obj):
        return obj.chunks.filter(has_embedding=False).count()

    def get_critical_issues_count(self, obj):
        return obj.quality_checks.filter(severity="critical").exclude(
            check_status="passed"
        ).count()

    def get_warnings_count(self, obj):
        return obj.quality_checks.filter(severity="warning").exclude(
            check_status="passed"
        ).count()

    def validate_admin_notes(self, value):
        value = (value or "").strip()

        if len(value) > 3000:
            raise serializers.ValidationError(
                "Admin notes are too long. Please keep them under 3000 characters."
            )

        return value


class ProfileDocumentUpdateSerializer(serializers.ModelSerializer):
    """
    Admin serializer for updating document metadata only.

    This should not be used to replace the file.
    Use ProfileDocumentReplaceSerializer for file replacement.
    """

    class Meta:
        model = ProfileDocument
        fields = [
            "title",
            "document_type",
            "source_label",
            "priority",
            "tags",
            "admin_notes",
        ]

    def validate_title(self, value):
        value = value.strip()

        if len(value) < 3:
            raise serializers.ValidationError("Document title is too short.")

        if len(value) > 255:
            raise serializers.ValidationError(
                "Document title is too long. Please keep it under 255 characters."
            )

        return value

    def validate_priority(self, value):
        if value is None:
            return 5

        if value < 1 or value > 10:
            raise serializers.ValidationError(
                "Priority must be between 1 and 10."
            )

        return value

    def validate_tags(self, value):
        return ProfileDocumentUploadSerializer().validate_tags(value)

    def validate_admin_notes(self, value):
        value = (value or "").strip()

        if len(value) > 3000:
            raise serializers.ValidationError(
                "Admin notes are too long. Please keep them under 3000 characters."
            )

        return value


class ProfileDocumentReplaceSerializer(
    DocumentFileValidationMixin,
    serializers.Serializer,
):
    """
    Serializer for replacing an existing document file.

    The old document should remain inactive or archived depending on your workflow.
    The new file should be reprocessed and reviewed before activation.
    """

    file = serializers.FileField(required=True)

    replacement_note = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=1000,
    )

    def validate_file(self, value):
        value = self.validate_document_file(value)

        file_hash = self.calculate_file_hash(value)

        existing_document = ProfileDocument.objects.filter(
            file_hash=file_hash
        ).exclude(
            pk=getattr(self.context.get("document"), "pk", None)
        ).exclude(
            status__in=["rejected", "archived"]
        ).first()

        if existing_document:
            raise serializers.ValidationError(
                f"This replacement file already exists as '{existing_document.title}'."
            )

        self.context["file_hash"] = file_hash

        return value

    def validate_replacement_note(self, value):
        value = (value or "").strip()

        if value and len(value) < 5:
            raise serializers.ValidationError(
                "Replacement note is too short. Please explain the reason briefly."
            )

        return value


class DocumentChunkReviewSerializer(serializers.ModelSerializer):
    """
    Admin serializer for reviewing and optionally editing generated chunks.
    """

    document_title = serializers.CharField(
        source="document.title", read_only=True)
    has_quality_issues = serializers.SerializerMethodField()
    content_preview = serializers.SerializerMethodField()

    class Meta:
        model = DocumentChunk
        fields = [
            "id",
            "document",
            "document_title",
            "chunk_index",
            "content",
            "content_preview",
            "section_title",
            "page_number",
            "token_count",
            "character_count",
            "embedding_model",
            "embedding_dimension",
            "has_embedding",
            "quality_status",
            "quality_score",
            "quality_issues",
            "has_quality_issues",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "document",
            "document_title",
            "chunk_index",
            "embedding_model",
            "embedding_dimension",
            "has_embedding",
            "quality_score",
            "quality_issues",
            "created_at",
            "updated_at",
        ]

    def get_has_quality_issues(self, obj):
        return bool(obj.quality_issues)

    def get_content_preview(self, obj):
        content = obj.content or ""
        return content[:300] + "..." if len(content) > 300 else content

    def validate_content(self, value):
        value = value.strip()

        if len(value) < 50:
            raise serializers.ValidationError(
                "Chunk content is too short. It may not provide enough context for chatbot retrieval."
            )

        if len(value) > 5000:
            raise serializers.ValidationError(
                "Chunk content is too long. Please keep it under 5000 characters."
            )

        words = re.findall(r"\b\w+\b", value)

        if len(words) < 10:
            raise serializers.ValidationError(
                "Chunk has too few meaningful words."
            )

        return value

    def validate_section_title(self, value):
        value = (value or "").strip()

        if len(value) > 255:
            raise serializers.ValidationError(
                "Section title is too long. Please keep it under 255 characters."
            )

        return value or None

    def validate_page_number(self, value):
        if value is None:
            return value

        if value < 1:
            raise serializers.ValidationError(
                "Page number must be greater than or equal to 1."
            )

        if value > 10000:
            raise serializers.ValidationError(
                "Page number is too large."
            )

        return value


class DocumentQualityCheckSerializer(serializers.ModelSerializer):
    """
    Serializer for displaying document quality checks in the admin panel.
    """

    class Meta:
        model = DocumentQualityCheck
        fields = [
            "id",
            "document",
            "check_name",
            "check_status",
            "severity",
            "message",
            "details",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "document",
            "check_name",
            "check_status",
            "severity",
            "message",
            "details",
            "created_at",
            "updated_at",
        ]


class DocumentProcessingLogSerializer(serializers.ModelSerializer):
    """
    Serializer for the document processing timeline.
    """

    class Meta:
        model = DocumentProcessingLog
        fields = [
            "id",
            "document",
            "step",
            "level",
            "message",
            "metadata",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "document",
            "step",
            "level",
            "message",
            "metadata",
            "created_at",
            "updated_at",
        ]

class ProfileDocumentMarkReviewedSerializer(serializers.Serializer):
    """
    Serializer for marking a document as reviewed by an admin.
    """

    review_notes = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=3000,
    )

    def validate(self, attrs):
        document = self.context.get("document")

        if not document:
            raise serializers.ValidationError("Document context is required.")

        if document.status not in ["ready_for_review", "approved"]:
            raise serializers.ValidationError(
                "Only ready-for-review or approved documents can be marked as reviewed."
            )

        if not document.raw_text or not document.raw_text.strip():
            raise serializers.ValidationError(
                "Cannot mark this document as reviewed because no extracted text exists."
            )

        if not document.chunks.exists():
            raise serializers.ValidationError(
                "Cannot mark this document as reviewed because no chunks were generated."
            )

        missing_embeddings_count = document.chunks.filter(
            has_embedding=False
        ).count()

        if missing_embeddings_count > 0:
            raise serializers.ValidationError(
                f"Cannot mark this document as reviewed because {missing_embeddings_count} chunks are missing embeddings."
            )

        attrs["review_notes"] = (attrs.get("review_notes") or "").strip()

        return attrs
    
class ProfileDocumentApproveSerializer(serializers.Serializer):
    """
    Serializer for approving a document before chatbot activation.
    """

    admin_notes = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=3000,
    )

    force_approve = serializers.BooleanField(
        required=False,
        default=False,
        help_text="Allows approval with warnings, but never with critical failed checks.",
    )

    def validate(self, attrs):
        document = self.context.get("document")

        if not document:
            raise serializers.ValidationError("Document context is required.")

        if document.status not in ["ready_for_review", "embedded", "validation_warning"]:
            raise serializers.ValidationError(
                "Only documents that are ready for review can be approved."
            )
            
        if not document.is_reviewed:
            raise serializers.ValidationError(
                "This document must be reviewed by an admin before approval."
            )

        if not document.raw_text or not document.raw_text.strip():
            raise serializers.ValidationError(
                "Cannot approve this document because no extracted text exists."
            )

        chunks = document.chunks.all()

        if not chunks.exists():
            raise serializers.ValidationError(
                "Cannot approve this document because no chunks were generated."
            )

        missing_embeddings_count = chunks.filter(has_embedding=False).count()

        if missing_embeddings_count > 0:
            raise serializers.ValidationError(
                f"Cannot approve this document because {missing_embeddings_count} chunks are missing embeddings."
            )

        critical_issues = document.quality_checks.filter(
            severity="critical"
        ).exclude(
            check_status="passed"
        )

        if critical_issues.exists():
            raise serializers.ValidationError(
                "Cannot approve this document because it has failed critical quality checks."
            )

        failed_chunks = chunks.filter(quality_status="failed")

        if failed_chunks.exists():
            raise serializers.ValidationError(
                f"Cannot approve this document because {failed_chunks.count()} chunks failed quality checks."
            )

        force_approve = attrs.get("force_approve", False)

        if document.overall_quality_score < 75 and not force_approve:
            raise serializers.ValidationError(
                "Document quality score is below 75. Use force approval only if you reviewed it manually and warnings are acceptable."
            )

        return attrs


class ProfileDocumentRejectSerializer(serializers.Serializer):
    """
    Serializer for rejecting a document.
    """

    reason = serializers.CharField(
        required=True,
        allow_blank=False,
        min_length=5,
        max_length=2000,
        error_messages={
            "required": "Rejection reason is required.",
            "blank": "Rejection reason is required.",
            "min_length": "Please write a clear rejection reason.",
        },
    )

    def validate_reason(self, value):
        value = value.strip()

        if len(value) < 5:
            raise serializers.ValidationError(
                "Please write a clear rejection reason."
            )

        return value


class ProfileDocumentReprocessSerializer(serializers.Serializer):
    """
    Serializer for reprocessing an existing document.
    """

    clear_existing_chunks = serializers.BooleanField(
        required=False,
        default=True,
    )

    run_extraction = serializers.BooleanField(
        required=False,
        default=True,
    )

    run_validation = serializers.BooleanField(
        required=False,
        default=True,
    )

    run_chunking = serializers.BooleanField(
        required=False,
        default=True,
    )

    run_embedding = serializers.BooleanField(
        required=False,
        default=True,
    )

    reason = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=1000,
    )

    def validate(self, attrs):
        document = self.context.get("document")

        if not document:
            raise serializers.ValidationError("Document context is required.")

        if document.status == "approved":
            raise serializers.ValidationError(
                "Approved documents should not be reprocessed directly. Replace the document or create a new version instead."
            )

        selected_steps = [
            attrs.get("run_extraction", True),
            attrs.get("run_validation", True),
            attrs.get("run_chunking", True),
            attrs.get("run_embedding", True),
        ]

        if not any(selected_steps):
            raise serializers.ValidationError(
                "Please select at least one processing step."
            )

        reason = (attrs.get("reason") or "").strip()

        if reason and len(reason) < 5:
            raise serializers.ValidationError(
                {"reason": ["Reason is too short."]}
            )

        attrs["reason"] = reason

        return attrs


class ProfileDocumentInspectionSerializer(serializers.ModelSerializer):
    """
    Serializer for deep document inspection.
    Used by the admin panel inspection page.
    """

    text_statistics = serializers.SerializerMethodField()
    chunk_statistics = serializers.SerializerMethodField()
    embedding_statistics = serializers.SerializerMethodField()
    quality_summary = serializers.SerializerMethodField()

    class Meta:
        model = ProfileDocument
        fields = [
            "id",
            "title",
            "document_type",
            "source_label",
            "status",
            "quality_status",
            "raw_text",
            "extracted_text_preview",
            "text_statistics",
            "chunk_statistics",
            "embedding_statistics",
            "quality_summary",
            "validation_summary",
            "processing_metadata",
            "overall_quality_score",
            "created_at",
            "updated_at",
        ]

    def get_text_statistics(self, obj):
        text = obj.raw_text or ""
        words = re.findall(r"\b\w+\b", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        symbols = re.findall(r"[^a-zA-Z0-9\s.,;:!?()\-_/]", text)
        symbol_ratio = len(symbols) / max(len(text), 1)

        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "symbol_count": len(symbols),
            "symbol_ratio": round(symbol_ratio, 4),
            "has_text": bool(text.strip()),
        }

    def get_chunk_statistics(self, obj):
        chunks = obj.chunks.all()
        total = chunks.count()

        return {
            "total_chunks": total,
            "active_chunks": chunks.filter(is_active=True).count(),
            "passed_chunks": chunks.filter(quality_status="passed").count(),
            "warning_chunks": chunks.filter(quality_status="warning").count(),
            "failed_chunks": chunks.filter(quality_status="failed").count(),
        }

    def get_embedding_statistics(self, obj):
        chunks = obj.chunks.all()
        total = chunks.count()
        embedded = chunks.filter(has_embedding=True).count()

        return {
            "total_chunks": total,
            "embedded_chunks": embedded,
            "missing_embeddings": total - embedded,
            "embedding_completion_rate": round((embedded / total) * 100, 2) if total else 0,
        }

    def get_quality_summary(self, obj):
        checks = obj.quality_checks.all()

        return {
            "total_checks": checks.count(),
            "passed": checks.filter(check_status="passed").count(),
            "warnings": checks.filter(check_status="warning").count(),
            "failed": checks.filter(check_status="failed").count(),
            "critical": checks.filter(severity="critical").exclude(check_status="passed").count(),
        }


class ProfileDocumentVersionSerializer(serializers.ModelSerializer):
    """
    Serializer for document version history.
    """

    class Meta:
        model = ProfileDocumentVersion
        fields = [
            "id",
            "document",
            "version_number",
            "file",
            "raw_text_snapshot",
            "status_snapshot",
            "quality_score_snapshot",
            "notes",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "document",
            "version_number",
            "file",
            "raw_text_snapshot",
            "status_snapshot",
            "quality_score_snapshot",
            "created_at",
            "updated_at",
        ]
