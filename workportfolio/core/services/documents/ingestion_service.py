"""
Document ingestion service.

Pipeline:
- parse file
- store raw text
- classify document type
- create chunks using document-type-aware chunking
- generate embeddings
- save chunks
- run quality checks
- mark document ready for admin review
"""

import re

from django.db import transaction

from core.models import ProfileDocument, DocumentChunk
from core.services.documents.parser_service import ParserService
from core.services.documents.chunk_service import ChunkService
from core.services.documents.embedding_service import EmbeddingService
from core.services.documents.doc_type_classifier import DocumentTypeClassifier

from core.services.knowledge_quality import (
    DocumentQualityService,
    ChunkQualityService,
    EmbeddingQualityService,
)

from core.services.knowledge_quality.approval_service import (
    calculate_overall_quality_score,
)

from core.services.knowledge_quality.processing_log_service import (
    log_info,
    log_error,
)


class IngestionService:
    """
    Handles full document ingestion lifecycle.
    """

    @staticmethod
    def process_document(document: ProfileDocument) -> ProfileDocument:
        """
        Parse, classify, chunk, embed, validate, and prepare document for admin review.
        """

        try:
            log_info(
                document=document,
                step="ingestion_started",
                message="Document ingestion started.",
            )

            document.status = "extracting"
            document.save(update_fields=["status", "updated_at"])

            raw_text = ParserService.extract_text(document.file.path)
            raw_text = (raw_text or "").strip()

            if not raw_text:
                document.raw_text = ""
                document.status = "extraction_failed"
                document.quality_status = "failed"
                document.save(
                    update_fields=[
                        "raw_text",
                        "status",
                        "quality_status",
                        "updated_at",
                    ]
                )

                log_error(
                    document=document,
                    step="extraction_failed",
                    message="No text was extracted from the document.",
                )

                return document

            document.status = "extracted"
            document.raw_text = raw_text
            document.extracted_text_preview = raw_text[:1000]
            document.save(
                update_fields=[
                    "status",
                    "raw_text",
                    "extracted_text_preview",
                    "updated_at",
                ]
            )

            log_info(
                document=document,
                step="extraction_completed",
                message="Document text extracted successfully.",
                metadata={
                    "raw_text_length": len(raw_text),
                },
            )

            result = DocumentTypeClassifier.classify(
                title=document.title,
                raw_text=raw_text,
            )

            document.document_type = result.doc_type
            document.tags = result.tags
            document.processing_metadata = {
                **(document.processing_metadata or {}),
                "doc_type_confidence": result.confidence,
                "doc_type_source": result.source,
            }
            document.save(
                update_fields=[
                    "document_type",
                    "tags",
                    "processing_metadata",
                    "updated_at",
                ]
            )

            log_info(
                document=document,
                step="document_type_classified",
                message="Document type classified successfully.",
                metadata={
                    "document_type": result.doc_type,
                    "confidence": result.confidence,
                    "source": result.source,
                    "tags": result.tags,
                },
            )

            document.status = "chunking"
            document.save(update_fields=["status", "updated_at"])

            chunks = ChunkService.chunk_document(
                raw_text=raw_text,
                document_type=result.doc_type,
                title=document.title,
            )

            if not chunks:
                document.status = "chunking_failed"
                document.quality_status = "failed"
                document.save(
                    update_fields=[
                        "status",
                        "quality_status",
                        "updated_at",
                    ]
                )

                log_error(
                    document=document,
                    step="chunking_failed",
                    message="No chunks were generated from the document.",
                )

                return document

            log_info(
                document=document,
                step="chunking_completed",
                message="Document chunks generated successfully.",
                metadata={
                    "chunks_count": len(chunks),
                },
            )

            document.status = "embedding"
            document.save(update_fields=["status", "updated_at"])

            embeddings = EmbeddingService.generate_embeddings(
                chunks,
                task="retrieval.passage",
            )

            if len(embeddings) != len(chunks):
                raise ValueError(
                    f"Embedding count mismatch. Expected {len(chunks)} embeddings, got {len(embeddings)}."
                )

            with transaction.atomic():
                document.raw_text = raw_text
                document.document_type = result.doc_type
                document.tags = result.tags
                document.status = "embedded"
                document.is_active = False
                document.is_approved = False
                document.is_available_for_chatbot = False

                document.save(
                    update_fields=[
                        "raw_text",
                        "document_type",
                        "tags",
                        "status",
                        "is_active",
                        "is_approved",
                        "is_available_for_chatbot",
                        "updated_at",
                    ]
                )

                document.chunks.all().delete()

                chunk_objects = []

                for index, chunk_text in enumerate(chunks):
                    embedding = embeddings[index]

                    chunk_objects.append(
                        DocumentChunk(
                            document=document,
                            chunk_index=index,
                            content=chunk_text,
                            embedding=embedding,
                            has_embedding=True,
                            embedding_model=EmbeddingService.MODEL_NAME,
                            embedding_dimension=len(embedding),
                            character_count=len(chunk_text or ""),
                            token_count=len(re.findall(
                                r"\b\w+\b", chunk_text or "")),
                            quality_status="pending",
                            quality_score=0,
                            quality_issues=[],
                            is_active=False,
                        )
                    )

                DocumentChunk.objects.bulk_create(chunk_objects)

            log_info(
                document=document,
                step="embedding_completed",
                message="Embeddings generated and chunks saved successfully.",
                metadata={
                    "chunks_count": len(chunks),
                    "embedding_model": EmbeddingService.MODEL_NAME,
                },
            )

            # Run quality checks after embedding to capture embedding-related issues in the logs 
            # and quality checks
            DocumentQualityService(document).run_all_checks()
            ChunkQualityService(document).run_all_checks()
            EmbeddingQualityService(document).run_all_checks()
            calculate_overall_quality_score(document)

            document.status = "ready_for_review"
            document.save(update_fields=["status", "updated_at"])

            log_info(
                document=document,
                step="ready_for_review",
                message="Document is ready for admin review.",
                metadata={
                    "overall_quality_score": document.overall_quality_score,
                },
            )

        except Exception as exc:
            document.status = "failed"
            document.quality_status = "failed"
            document.save(
                update_fields=[
                    "status",
                    "quality_status",
                    "updated_at",
                ]
            )

            log_error(
                document=document,
                step="ingestion_failed",
                message=str(exc),
                metadata={
                    "error_type": exc.__class__.__name__,
                },
            )

            raise

        return document
