"""
Document approval service.

This service controls whether a document can become active in the chatbot
knowledge base.
"""

from django.db import transaction
from django.utils import timezone
from rest_framework.exceptions import ValidationError

from core.models import ProfileDocument
from core.services.knowledge_quality.processing_log_service import (
    log_info,
    log_warning,
)



class DocumentApprovalService:
    """
    Handles approval, rejection, and archive actions for chatbot documents.
    """

    MIN_APPROVAL_SCORE = 75

    def approve(
        self,
        document: ProfileDocument,
        admin_notes: str = "",
        force_approve: bool = False,
    ) -> ProfileDocument:
        """
        Approve a document and activate its chunks.

        A document can only be approved if:
        - It has extracted text
        - It has chunks
        - All chunks have embeddings
        - No critical quality check failed
        - No chunk has failed quality status
        - Overall score is acceptable unless force_approve=True
        """

        self.validate_can_approve(
            document=document,
            force_approve=force_approve,
        )

        with transaction.atomic():
            document.status = "approved"
            document.quality_status = (
                "warning" if force_approve and document.overall_quality_score < self.MIN_APPROVAL_SCORE else "passed"
            )
            document.is_active = True
            document.is_approved = True
            document.is_available_for_chatbot = True
            document.approved_at = timezone.now()
            document.rejected_at = None
            document.rejection_reason = ""

            if admin_notes:
                document.admin_notes = admin_notes.strip()

            document.save(
                update_fields=[
                    "status",
                    "quality_status",
                    "is_active",
                    "is_approved",
                    "is_available_for_chatbot",
                    "approved_at",
                    "rejected_at",
                    "rejection_reason",
                    "admin_notes",
                    "updated_at",
                ]
            )

            document.chunks.filter(
                has_embedding=True,
                quality_status__in=["passed", "warning"],
            ).update(is_active=True)

            document.chunks.exclude(
                has_embedding=True,
                quality_status__in=["passed", "warning"],
            ).update(is_active=False)

            log_info(
                document=document,
                step="document_approved",
                message="Document was approved and activated for chatbot retrieval.",
                metadata={
                    "force_approve": force_approve,
                    "overall_quality_score": document.overall_quality_score,
                },
            )

        return document

    def reject(
        self,
        document: ProfileDocument,
        reason: str,
    ) -> ProfileDocument:
        """
        Reject a document and deactivate its chunks.
        """

        reason = (reason or "").strip()

        if len(reason) < 5:
            raise ValidationError("Rejection reason is required.")

        with transaction.atomic():
            document.status = "rejected"
            document.quality_status = "failed"
            document.is_active = False
            document.is_approved = False
            document.is_available_for_chatbot = False
            document.rejected_at = timezone.now()
            document.rejection_reason = reason
            document.save(
                update_fields=[
                    "status",
                    "quality_status",
                    "is_active",
                    "is_approved",
                    "is_available_for_chatbot",
                    "rejected_at",
                    "rejection_reason",
                    "updated_at",
                ]
            )

            document.chunks.update(is_active=False)

            log_warning(
                document=document,
                step="document_rejected",
                message="Document was rejected and removed from chatbot retrieval.",
                metadata={"reason": reason},
            )

        return document

    def archive(self, document: ProfileDocument) -> ProfileDocument:
        """
        Archive a document and deactivate its chunks.
        """

        with transaction.atomic():
            document.status = "archived"
            document.is_active = False
            document.is_approved = False
            document.is_available_for_chatbot = False
            document.save(
                update_fields=[
                    "status",
                    "is_active",
                    "is_approved",
                    "is_available_for_chatbot",
                    "updated_at",
                ]
            )

            document.chunks.update(is_active=False)

            log_info(
                document=document,
                step="document_archived",
                message="Document was archived and deactivated.",
            )

        return document

    def validate_can_approve(
        self,
        document: ProfileDocument,
        force_approve: bool = False,
    ) -> None:
        """
        Validate whether a document can be approved.
        """

        if document.status not in [
            "ready_for_review",
            "embedded",
            "validation_warning",
        ]:
            raise ValidationError(
                "Only documents that are ready for review can be approved."
            )

        if not document.raw_text or not document.raw_text.strip():
            raise ValidationError(
                "Cannot approve this document because no extracted text exists."
            )

        chunks = document.chunks.all()

        if not chunks.exists():
            raise ValidationError(
                "Cannot approve this document because no chunks were generated."
            )

        missing_embeddings_count = chunks.filter(has_embedding=False).count()

        if missing_embeddings_count > 0:
            raise ValidationError(
                f"Cannot approve this document because {missing_embeddings_count} chunks are missing embeddings."
            )

        critical_issues = document.quality_checks.filter(
            severity="critical"
        ).exclude(
            check_status="passed"
        )

        if critical_issues.exists():
            raise ValidationError(
                "Cannot approve this document because it has failed critical quality checks."
            )

        failed_chunks_count = chunks.filter(quality_status="failed").count()

        if failed_chunks_count > 0:
            raise ValidationError(
                f"Cannot approve this document because {failed_chunks_count} chunks failed quality checks."
            )

        if document.overall_quality_score < self.MIN_APPROVAL_SCORE and not force_approve:
            raise ValidationError(
                f"Document quality score is below {self.MIN_APPROVAL_SCORE}. Use force approval only after manual review."
            )


def calculate_overall_quality_score(document: ProfileDocument) -> float:
    """
    Calculate weighted overall quality score.

    Extraction = 40%
    Chunk quality = 30%
    Embedding quality = 30%
    """

    extraction_score = document.extraction_score or 0
    chunk_score = document.chunk_quality_score or 0
    embedding_score = document.embedding_quality_score or 0

    overall_score = (
        extraction_score * 0.4
        + chunk_score * 0.3
        + embedding_score * 0.3
    )

    document.overall_quality_score = round(overall_score, 2)
    document.save(update_fields=["overall_quality_score", "updated_at"])

    return document.overall_quality_score