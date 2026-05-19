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
from django.utils import timezone
from datetime import timedelta


class HeroSection(models.Model):
    """
    Stores editable content for the portfolio hero section.

    This allows the admin panel to update the homepage hero content
    without changing the frontend code.
    """

    title_prefix = models.CharField(
        max_length=100,
        default="Hi, I am",
        help_text="Small text before the main name/title.",
    )

    main_title = models.CharField(
        max_length=150,
        default="Samah Fujo",
        help_text="Main hero title, usually the portfolio owner's name.",
    )

    subtitle = models.CharField(
        max_length=255,
        blank=True,
        default="Senior Python/Django Engineer with 5+ years of experience",
        help_text="Short professional headline.",
    )

    description = models.TextField(
        blank=True,
        help_text="Main hero paragraph shown under the title.",
    )

    primary_button_text = models.CharField(
        max_length=80,
        default="Download CV",
    )

    primary_button_url = models.CharField(
        max_length=255,
        blank=True,
        default="/cv.pdf",
        help_text="URL or file path for the primary button.",
    )

    secondary_button_text = models.CharField(
        max_length=80,
        default="Start Project",
    )

    secondary_button_url = models.CharField(
        max_length=255,
        blank=True,
        default="#contact",
        help_text="URL or section anchor for the secondary button.",
    )

    hero_image_dark = models.ImageField(
        upload_to="hero/images/dark/",
        blank=True,
        null=True,
        help_text="Hero image used in dark mode."
    )

    hero_image_light = models.ImageField(
        upload_to="hero/images/light/",
        blank=True,
        null=True,
        help_text="Hero image used in light mode."
    )

    background_image = models.ImageField(
        upload_to="hero/backgrounds/",
        blank=True,
        null=True,
        help_text="Optional background image or brush effect.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only one active hero section should be used on the public website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Hero Section"
        verbose_name_plural = "Hero Sections"
        ordering = ["-updated_at"]
        
    def save(self, *args, **kwargs):
        """
        Ensure only one HeroSection is active at a time.
        """
        if self.is_active:
            HeroSection.objects.exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.main_title} Hero Section"


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

    ADMIN_STATUS_CHOICES = [
        ("open", "Open"),
        ("reviewed", "Reviewed"),
        ("closed", "Closed"),
        ("archived", "Archived"),
    ]

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

    admin_status = models.CharField(
        max_length=20,
        choices=ADMIN_STATUS_CHOICES,
        default="open",
        db_index=True,
        help_text="Admin review workflow status for this chat session.",
    )

    admin_note = models.TextField(
        blank=True,
        default="",
        help_text="Optional internal admin note about this session.",
    )

    reviewed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the admin marked this session as reviewed.",
    )

    closed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the admin closed or archived this session.",
    )

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


class EmailVerificationCode(models.Model):
    """
    Stores temporary email verification codes before allowing chatbot access.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    email = models.EmailField(db_index=True)

    code = models.CharField(max_length=6)

    is_used = models.BooleanField(default=False)

    attempts = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    expires_at = models.DateTimeField()

    def is_expired(self):
        return timezone.now() > self.expires_at

    @classmethod
    def create_code(cls, email: str, code: str, expiry_minutes: int = 10):
        return cls.objects.create(
            email=email.lower().strip(),
            code=code,
            expires_at=timezone.now() + timedelta(minutes=expiry_minutes),
        )


class ContactMessage(TimeStampedModel):
    """
    Stores messages submitted from the Get in Touch form.
    """

    STATUS_CHOICES = [
        ("new", "New"),
        ("read", "Read"),
        ("replied", "Replied"),
        ("archived", "Archived"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    name = models.CharField(max_length=255)
    email = models.EmailField(db_index=True)
    subject = models.CharField(max_length=255, blank=True, null=True)
    message = models.TextField()

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="new",
        db_index=True,
    )

    ip_address = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.TextField(blank=True, null=True)
    referrer = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} - {self.email}"


class ProjectRequest(TimeStampedModel):
    """
    Stores project requests submitted from the Start Project form.
    """

    STATUS_CHOICES = [
        ("new", "New"),
        ("reviewed", "Reviewed"),
        ("contacted", "Contacted"),
        ("accepted", "Accepted"),
        ("rejected", "Rejected"),
        ("archived", "Archived"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    project_name = models.CharField(max_length=255)
    project_type = models.CharField(max_length=100, blank=True, null=True)
    budget_range = models.CharField(max_length=100, blank=True, null=True)
    timeline = models.CharField(max_length=100, blank=True, null=True)
    project_description = models.TextField()

    your_name = models.CharField(max_length=255)
    your_email = models.EmailField(db_index=True)

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="new",
        db_index=True,
    )

    ip_address = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.TextField(blank=True, null=True)
    referrer = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.project_name} - {self.your_email}"
