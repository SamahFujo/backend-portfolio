"""
Knowledge quality service package.

This package contains reusable services for validating, inspecting,
approving, rejecting, and logging chatbot knowledge documents.

How it works:
Upload document
↓
Extract text
↓
Classify document type
↓
Chunk document
↓
Generate embeddings
↓
Validate document quality
↓
Validate chunk quality
↓
Validate embedding quality
↓
Calculate overall score
↓
Ready for admin review
↓
Admin approves or rejects
↓
Only approved documents become available to the chatbot

"""

from .processing_log_service import (
    create_processing_log,
    log_info,
    log_warning,
    log_error,
    log_critical,
)

from .document_quality_service import DocumentQualityService
from .chunk_quality_service import ChunkQualityService
from .embedding_quality_service import EmbeddingQualityService
from .approval_service import DocumentApprovalService

__all__ = [
    "create_processing_log",
    "log_info",
    "log_warning",
    "log_error",
    "log_critical",
    "DocumentQualityService",
    "ChunkQualityService",
    "EmbeddingQualityService",
    "DocumentApprovalService",
]
