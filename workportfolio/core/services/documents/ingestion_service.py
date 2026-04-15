"""
Document ingestion service.

Pipeline:
- parse file
- store raw text
- classify document type
- create chunks using document-type-aware chunking
- generate embeddings
- save chunks
"""

from django.db import transaction

from core.models import ProfileDocument, DocumentChunk
from .parser_service import ParserService
from .chunk_service import ChunkService
from .embedding_service import EmbeddingService
from .doc_type_classifier import DocumentTypeClassifier


class IngestionService:
    """
    Handles full document ingestion lifecycle.
    """

    @staticmethod
    def process_document(document: ProfileDocument) -> ProfileDocument:
        """
        Parse, chunk, embed, classify, and save a document.
        """
        try:
            raw_text = ParserService.extract_text(document.file.path)
            raw_text = (raw_text or "").strip()

            if not raw_text:
                document.raw_text = ""
                document.status = "failed"
                document.save(update_fields=["raw_text", "status", "updated_at"])
                return document

            result = DocumentTypeClassifier.classify(
                title=document.title,
                raw_text=raw_text,
            )

            chunks = ChunkService.chunk_document(
                raw_text=raw_text,
                document_type=result.doc_type,
                title=document.title,
            )

            if not chunks:
                document.raw_text = raw_text
                document.document_type = result.doc_type
                document.tags = result.tags
                document.status = "failed"

                update_fields = [
                    "raw_text",
                    "document_type",
                    "tags",
                    "status",
                    "updated_at",
                ]

                if hasattr(document, "doc_type_confidence"):
                    document.doc_type_confidence = result.confidence
                    update_fields.append("doc_type_confidence")

                if hasattr(document, "doc_type_source"):
                    document.doc_type_source = result.source
                    update_fields.append("doc_type_source")

                document.save(update_fields=update_fields)
                return document

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
                document.status = "processed"

                update_fields = [
                    "raw_text",
                    "document_type",
                    "tags",
                    "status",
                    "updated_at",
                ]

                if hasattr(document, "doc_type_confidence"):
                    document.doc_type_confidence = result.confidence
                    update_fields.append("doc_type_confidence")

                if hasattr(document, "doc_type_source"):
                    document.doc_type_source = result.source
                    update_fields.append("doc_type_source")

                if hasattr(document, "chunk_count"):
                    document.chunk_count = len(chunks)
                    update_fields.append("chunk_count")

                document.save(update_fields=update_fields)

                document.chunks.all().delete()

                chunk_objects = []
                for index, chunk_text in enumerate(chunks):
                    chunk_objects.append(
                        DocumentChunk(
                            document=document,
                            chunk_index=index,
                            content=chunk_text,
                            embedding=embeddings[index],
                        )
                    )

                DocumentChunk.objects.bulk_create(chunk_objects)

        except Exception:
            document.status = "failed"
            document.save(update_fields=["status", "updated_at"])
            raise

        return document