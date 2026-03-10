"""
Document ingestion service.

Pipeline:
- parse file
- store raw text
- create chunks
- generate embeddings
- save chunks
"""

from core.models import ProfileDocument, DocumentChunk
from .parser_service import ParserService
from .chunk_service import ChunkService
from .embedding_service import EmbeddingService


class IngestionService:
    """
    Handles full document ingestion lifecycle.
    """

    @staticmethod
    def process_document(document: ProfileDocument) -> ProfileDocument:
        """
        Parse, chunk, embed, and save a document.
        """
        try:
            raw_text = ParserService.extract_text(document.file.path)
            document.raw_text = raw_text
            document.status = "processed"
            document.save(update_fields=["raw_text", "status", "updated_at"])

            document.chunks.all().delete()

            chunks = ChunkService.chunk_text(raw_text)
            if not chunks:
                return document

            embeddings = EmbeddingService.generate_embeddings(chunks)

            chunk_objects = []
            for index, chunk_text in enumerate(chunks):
                embedding = embeddings[index] if index < len(embeddings) else None
                chunk_objects.append(
                    DocumentChunk(
                        document=document,
                        chunk_index=index,
                        content=chunk_text,
                        embedding=embedding,
                    )
                )

            DocumentChunk.objects.bulk_create(chunk_objects)

        except Exception:
            document.status = "failed"
            document.save(update_fields=["status", "updated_at"])
            raise

        return document