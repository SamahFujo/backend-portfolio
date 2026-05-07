"""
Database models for the portfolio chatbot backend.

This module defines:
- ProfileDocument: uploaded documents related to Samah
- DocumentChunk: chunked text extracted from documents
- ChatSession: logical chat session
- ChatMessage: messages exchanged in a session
"""

import uuid
from django.db import models
from pgvector.django import VectorField


class TimeStampedModel(models.Model):
    """
    Abstract base model to track creation and update timestamps.
    """
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class ProfileDocument(TimeStampedModel):
    """
    Stores uploaded profile-related documents such as:
    CVs, certificates, project summaries, recommendation letters, etc.
    """

    STATUS_CHOICES = [
        ("uploaded", "Uploaded"),
        ("processed", "Processed"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="profile_documents/")
    document_type = models.CharField(max_length=100, blank=True, null=True)
    raw_text = models.TextField(blank=True, null=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="uploaded")
    is_active = models.BooleanField(default=True)
    priority = models.PositiveSmallIntegerField(default=5)  # 1 high priority
    tags = models.JSONField(blank=True, null=True)          # list of strings
    source_label = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Optional label such as CV 2026, LinkedIn export, Project Summary, etc."
    )

    def __str__(self):
        return self.title


class DocumentChunk(TimeStampedModel):
    """
    Stores chunked pieces of a document for retrieval.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        ProfileDocument,
        on_delete=models.CASCADE,
        related_name="chunks"
    )
    chunk_index = models.PositiveIntegerField()
    content = models.TextField()
    section_title = models.CharField(max_length=255, blank=True, null=True)
    page_number = models.PositiveIntegerField(blank=True, null=True)

    # 1536 is a good default if you plan to use OpenAI text embeddings.
    # If you later choose a different embedding model, change this before migrating.
    embedding = VectorField(dimensions=1024, blank=True, null=True)

    class Meta:
        ordering = ["document", "chunk_index"]
        unique_together = ("document", "chunk_index")

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"


class ChatSession(TimeStampedModel):
    """
    Represents one chatbot conversation session.

    Each visitor should keep the same session_id while chatting.
    visitor_id helps track anonymous users across sessions without requiring login.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    visitor_email = models.EmailField(
        blank=True,
        null=True,
        db_index=True,
        help_text="Visitor email collected before starting chatbot conversation."
    )

    visitor_id = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        db_index=True,
        help_text="Anonymous visitor identifier stored in frontend localStorage/cookie."
    )

    ip_address = models.GenericIPAddressField(
        blank=True,
        null=True,
        help_text="Visitor IP address for analytics/security tracking."
    )

    user_agent = models.TextField(
        blank=True,
        null=True,
        help_text="Browser/device information."
    )

    referrer = models.TextField(
        blank=True,
        null=True,
        help_text="Page or source that referred the visitor."
    )

    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"ChatSession {self.id} - Visitor {self.visitor_id or 'anonymous'}"


class ChatMessage(TimeStampedModel):
    """
    Stores each message exchanged between visitor and assistant.
    """

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages"
    )

    role = models.CharField(max_length=20, choices=ROLE_CHOICES)

    content = models.TextField()

    citations = models.JSONField(default=list, blank=True)

    confidence_score = models.FloatField(blank=True, null=True)

    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Stores route, mode, model used, retrieval info, UI action, etc."
    )

    def __str__(self):
        return f"{self.role} message in {self.session_id}"
