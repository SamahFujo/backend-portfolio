"""
Approved chunks helper.

This file contains the final safety gate for chatbot retrieval.

Only approved, active, embedded, quality-checked chunks should be used
by the chatbot.
"""

from django.db.models import QuerySet

from core.models import DocumentChunk


def get_chatbot_available_chunks() -> QuerySet:
    """
    Return only chunks that are safe for chatbot retrieval.

    A chunk is safe only when:
    - its document is approved
    - its document is active
    - its document is available for chatbot
    - the chunk itself is active
    - the chunk has an embedding
    - the chunk passed quality checks or has acceptable warnings
    """

    return (
        DocumentChunk.objects.filter(
            document__status="approved",
            document__is_active=True,
            document__is_approved=True,
            document__is_available_for_chatbot=True,
            is_active=True,
            has_embedding=True,
            quality_status__in=["passed", "warning"],
        )
        .exclude(embedding__isnull=True)
        .select_related("document")
    )
