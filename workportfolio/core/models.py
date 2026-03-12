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
    Represents one chat session in the frontend.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_name = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return str(self.id)


class ChatMessage(TimeStampedModel):
    """
    Stores user and assistant messages for a given session.
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
    citations = models.JSONField(blank=True, null=True)
    confidence_score = models.FloatField(blank=True, null=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role} - {self.session_id}"
