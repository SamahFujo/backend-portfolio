"""
Chunk quality service.

This service validates generated chunks before they are made available
for chatbot retrieval.
"""

import re
from difflib import SequenceMatcher
from typing import Any

from core.models import ProfileDocument, DocumentChunk, DocumentQualityCheck
from core.services.knowledge_quality.processing_log_service import (
    log_info,
    log_warning,
)


class ChunkQualityService:
    """
    Validates document chunks.

    Checks include:
    - Chunk is not empty
    - Chunk is not too short
    - Chunk is not too long
    - Chunk has enough meaningful words
    - Chunk is not only heading text
    - Duplicate/similar chunks
    """

    MIN_CHUNK_CHARS = 80
    MAX_CHUNK_CHARS = 3500
    MIN_WORDS = 15
    DUPLICATE_THRESHOLD = 0.92
    
    def get_chunk_limits(self) -> dict:
        """
        Return document-type-aware chunk quality limits.

        These limits align with your existing ChunkService strategies.
        """

        document_type = (self.document.document_type or "").strip().lower()

        defaults = {
            "min_chars": self.MIN_CHUNK_CHARS,
            "max_chars": self.MAX_CHUNK_CHARS,
            "min_words": self.MIN_WORDS,
        }

        limits_by_type = {
            "cv": {
                "min_chars": 80,
                "max_chars": 1800,
                "min_words": 12,
            },
            "projects": {
                "min_chars": 120,
                "max_chars": 1800,
                "min_words": 20,
            },
            "certificates": {
                "min_chars": 80,
                "max_chars": 1200,
                "min_words": 10,
            },
            "faq": {
                "min_chars": 60,
                "max_chars": 1200,
                "min_words": 8,
            },
            "recommendation": {
                "min_chars": 120,
                "max_chars": 1800,
                "min_words": 20,
            },
            "experience_letter": {
                "min_chars": 120,
                "max_chars": 1800,
                "min_words": 20,
            },
            "capabilities": {
                "min_chars": 100,
                "max_chars": 1600,
                "min_words": 15,
            },
            "security_deployment": {
                "min_chars": 100,
                "max_chars": 1600,
                "min_words": 15,
            },
        }

        return limits_by_type.get(document_type, defaults)

    def __init__(self, document: ProfileDocument):
        self.document = document

    def run_all_checks(self) -> dict[str, Any]:
        """
        Validate all chunks for a document and update document chunk quality score.
        """

        log_info(
            document=self.document,
            step="chunk_validation_started",
            message="Chunk quality validation started.",
        )

        chunks = list(self.document.chunks.all().order_by("chunk_index"))

        if not chunks:
            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="no_chunks_found",
                check_status="failed",
                severity="critical",
                message="No chunks were generated for this document.",
                details={},
            )

            self.document.chunk_quality_score = 0
            self.document.status = "chunking_failed"
            self.document.save(
                update_fields=["chunk_quality_score", "status", "updated_at"])

            log_warning(
                document=self.document,
                step="chunk_validation_failed",
                message="No chunks were found for validation.",
            )

            return {
                "score": 0,
                "passed": False,
                "total_chunks": 0,
                "issues": [
                    {
                        "code": "no_chunks_found",
                        "message": "No chunks were generated.",
                        "severity": "critical",
                    }
                ],
            }

        validated_chunks = []

        for chunk in chunks:
            validated_chunks.append(self.validate_single_chunk(chunk))

        duplicate_issues = self.detect_duplicate_chunks(validated_chunks)

        total_score = sum(chunk.quality_score for chunk in validated_chunks)
        average_score = round(total_score / len(validated_chunks), 2)

        failed_count = DocumentChunk.objects.filter(
            document=self.document,
            quality_status="failed",
        ).count()

        warning_count = DocumentChunk.objects.filter(
            document=self.document,
            quality_status="warning",
        ).count()

        passed = failed_count == 0 and average_score >= 70

        self.document.chunk_quality_score = average_score
        self.document.status = "chunked" if passed else "validation_warning"
        self.document.save(
            update_fields=["chunk_quality_score", "status", "updated_at"])

        summary = {
            "score": average_score,
            "passed": passed,
            "total_chunks": len(chunks),
            "failed_chunks": failed_count,
            "warning_chunks": warning_count,
            "duplicate_issues": duplicate_issues,
        }

        if passed:
            log_info(
                document=self.document,
                step="chunk_validation_completed",
                message="Chunk quality validation completed successfully.",
                metadata=summary,
            )
        else:
            log_warning(
                document=self.document,
                step="chunk_validation_completed_with_issues",
                message="Chunk quality validation completed with warnings or failures.",
                metadata=summary,
            )

        return summary

    def validate_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Validate one chunk and update its quality fields.
        """

        content = (chunk.content or "").strip()
        issues = []
        score = 100
        
        limits = self.get_chunk_limits()
        min_chars = limits["min_chars"]
        max_chars = limits["max_chars"]
        min_words = limits["min_words"]

        character_count = len(content)
        words = re.findall(r"\b\w+\b", content)
        word_count = len(words)

        if not content:
            issues.append({
                "code": "empty_chunk",
                "message": "Chunk is empty.",
                "severity": "critical",
            })
            score -= 100

        if character_count < min_chars:
            issues.append({
                "code": "chunk_too_short",
                "message": "Chunk is too short and may not provide enough context.",
                "severity": "warning",
                "details": {
                    "character_count": character_count,
                    "minimum_required": min_chars,
                },
            })
            score -= 25

        if character_count > max_chars:
            issues.append({
                "code": "chunk_too_long",
                "message": "Chunk is too long and may reduce retrieval quality.",
                "severity": "warning",
                "details": {
                    "character_count": character_count,
                    "maximum_allowed": max_chars,
                },
            })
            score -= 20

        if word_count < min_words:
            issues.append({
                "code": "low_word_count",
                "message": "Chunk has too few meaningful words.",
                "severity": "warning",
                "details": {
                    "word_count": word_count,
                    "minimum_required": min_words,
                },
            })
            score -= 25

        if self.looks_like_heading_only(content):
            issues.append({
                "code": "heading_only",
                "message": "Chunk appears to contain only a heading.",
                "severity": "warning",
            })
            score -= 15

        quality_score = max(score, 0)

        if quality_score < 50 or any(issue["severity"] == "critical" for issue in issues):
            quality_status = "failed"
        elif issues:
            quality_status = "warning"
        else:
            quality_status = "passed"

        chunk.content = content
        chunk.character_count = character_count
        chunk.token_count = word_count
        chunk.quality_score = quality_score
        chunk.quality_status = quality_status
        chunk.quality_issues = issues
        chunk.save(
            update_fields=[
                "content",
                "character_count",
                "token_count",
                "quality_score",
                "quality_status",
                "quality_issues",
                "updated_at",
            ]
        )

        return chunk

    def detect_duplicate_chunks(self, chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        """
        Detect near-duplicate chunks inside the same document.
        """

        duplicate_issues = []

        for index, first_chunk in enumerate(chunks):
            for second_chunk in chunks[index + 1:]:
                similarity = SequenceMatcher(
                    None,
                    first_chunk.content or "",
                    second_chunk.content or "",
                ).ratio()

                if similarity >= self.DUPLICATE_THRESHOLD:
                    issue = {
                        "code": "duplicate_chunk",
                        "message": f"Chunk {first_chunk.chunk_index} is very similar to chunk {second_chunk.chunk_index}.",
                        "severity": "warning",
                        "details": {
                            "first_chunk_index": first_chunk.chunk_index,
                            "second_chunk_index": second_chunk.chunk_index,
                            "similarity": round(similarity, 4),
                        },
                    }

                    duplicate_issues.append(issue)

                    first_issues = first_chunk.quality_issues or []
                    first_issues.append(issue)

                    first_chunk.quality_issues = first_issues
                    first_chunk.quality_status = "warning"
                    first_chunk.quality_score = max(
                        first_chunk.quality_score - 15, 0)
                    first_chunk.save(
                        update_fields=[
                            "quality_issues",
                            "quality_status",
                            "quality_score",
                            "updated_at",
                        ]
                    )

        if duplicate_issues:
            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="duplicate_chunks_detected",
                check_status="warning",
                severity="warning",
                message="Some chunks are very similar to each other.",
                details={"duplicates": duplicate_issues[:20]},
            )
        else:
            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="duplicate_chunks_valid",
                check_status="passed",
                severity="info",
                message="No duplicate chunks detected.",
                details={},
            )

        return duplicate_issues

    def looks_like_heading_only(self, text: str) -> bool:
        """
        Detect if a chunk looks like a heading only.
        """

        words = text.split()

        if len(words) <= 8 and text.isupper():
            return True

        if len(words) <= 6 and not any(char in text for char in [".", ",", ";", ":"]):
            return True

        return False
