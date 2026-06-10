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

    eyebrow_text = models.CharField(
        max_length=100,
        default="Hi, I am",
        blank=True,
    )

    full_name = models.CharField(
        max_length=150,
        default="Samah Fujo",
        blank=True,
    )

    headline = models.CharField(
        max_length=255,
        default="Senior Python/Django Engineer",
        blank=True,
    )

    description = models.TextField(
        blank=True,
        default="",
    )

    primary_button_text = models.CharField(
        max_length=80,
        default="Download CV",
        blank=True,
    )

    primary_button_url = models.CharField(
        max_length=255,
        default="/assets/files/Samah-Fujo-CV.pdf",
        blank=True,
    )

    secondary_button_text = models.CharField(
        max_length=80,
        default="Start Project",
        blank=True,
    )

    secondary_button_url = models.CharField(
        max_length=255,
        default="modal:start-project",
        blank=True,
    )

    hero_image_dark = models.ImageField(
        upload_to="hero/images/dark/",
        blank=True,
        null=True,
    )

    hero_image_light = models.ImageField(
        upload_to="hero/images/light/",
        blank=True,
        null=True,
    )

    background_image = models.ImageField(
        upload_to="hero/backgrounds/",
        blank=True,
        null=True,
    )

    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if self.is_active:
            HeroSection.objects.exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Hero Section - {self.full_name}"


class AboutSection(models.Model):
    """
    Stores dynamic content for the public About Me section.

    The website displays the active AboutSection record.
    The custom admin dashboard can update this content without changing code.
    """

    section_title = models.CharField(
        max_length=120,
        default="About me",
        blank=True,
        help_text="Main section heading displayed on the left side.",
    )

    terminal_label = models.CharField(
        max_length=120,
        default="samah.dev",
        blank=True,
        help_text="Small label shown in the terminal/browser window header.",
    )

    welcome_title = models.CharField(
        max_length=120,
        default="✨ Welcome",
        blank=True,
        help_text="Welcome heading inside the terminal content.",
    )

    description = models.TextField(
        blank=True,
        default="",
        help_text="Main About Me paragraph shown inside the terminal window.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only the active About section will be displayed on the website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "About Section"
        verbose_name_plural = "About Sections"
        ordering = ["-updated_at"]

    def save(self, *args, **kwargs):
        """
        Ensure only one AboutSection is active at a time.
        """
        if self.is_active:
            AboutSection.objects.exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return f"About Section - {self.section_title}"


class SkillSection(models.Model):
    """
    Stores dynamic header content for the Skills section.

    The public website displays the active SkillSection record.
    Skill cards are stored separately in SkillItem.
    """

    badge_text = models.CharField(
        max_length=80,
        default="Expertise",
        blank=True,
        help_text="Small badge text shown above the section title.",
    )

    title_line_1 = models.CharField(
        max_length=120,
        default="Skills &",
        blank=True,
        help_text="First line of the Skills section title.",
    )

    title_line_2 = models.CharField(
        max_length=120,
        default="Capabilities.",
        blank=True,
        help_text="Second line of the Skills section title.",
    )

    description = models.TextField(
        blank=True,
        default="A comprehensive toolkit built through years of hands-on experience and continuous learning.",
        help_text="Short description shown on the right side of the section header.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only the active Skills section will be displayed on the website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Skill Section"
        verbose_name_plural = "Skill Sections"
        ordering = ["-updated_at"]

    def save(self, *args, **kwargs):
        """
        Ensure only one SkillSection is active at a time.
        """
        if self.is_active:
            SkillSection.objects.exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Skill Section - {self.title_line_1} {self.title_line_2}"


class SkillItem(models.Model):
    CATEGORY_CHOICES = [
        ("Frontend", "Frontend"),
        ("Backend", "Backend"),
        ("AI / LLM", "AI / LLM"),
        ("Database", "Database"),
        ("DevOps", "DevOps"),
        ("Languages", "Languages"),
        ("UI", "UI"),
    ]

    section = models.ForeignKey(
        SkillSection,
        on_delete=models.CASCADE,
        related_name="items",
    )

    category = models.CharField(
        max_length=50,
        choices=CATEGORY_CHOICES,
        default="Frontend",
    )

    icon = models.CharField(
        max_length=100,
        default="react",
        help_text="TechIcon name, for example react, nextjs, docker, python.",
    )

    label = models.CharField(
        max_length=150,
        help_text="Skill name shown on the website, for example React.",
    )

    level = models.PositiveIntegerField(
        default=7,
        help_text="Skill level from 1 to 10.",
    )

    summary_heading = models.CharField(
        max_length=180,
        blank=True,
        default="Practical technical capability",
    )

    summary_text = models.TextField(
        blank=True,
        default="This skill supports my real project delivery and contributes to building complete, production-oriented solutions.",
    )

    summary_points = models.JSONField(
        blank=True,
        default=list,
        help_text="List of points explaining why this skill matters.",
    )

    sort_order = models.PositiveIntegerField(default=0)

    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Skill Item"
        verbose_name_plural = "Skill Items"
        ordering = ["category", "sort_order", "created_at"]

    def __str__(self):
        return self.label


class ProjectSection(models.Model):
    """
    Stores dynamic header content for the Projects section.

    The public website displays the active ProjectSection record.
    Individual project cards are stored in ProjectItem.
    """

    title = models.CharField(
        max_length=120,
        default="Projects",
        blank=True,
        help_text="Main title shown above the Projects section.",
    )

    description = models.TextField(
        blank=True,
        default="Selected work across AI, automation, analytics, and full-stack product development",
        help_text="Short description shown below the Projects section title.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only the active Projects section will be displayed on the website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Project Section"
        verbose_name_plural = "Project Sections"
        ordering = ["-updated_at"]

    def save(self, *args, **kwargs):
        """
        Ensure only one ProjectSection is active at a time.
        """
        if self.is_active:
            ProjectSection.objects.exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return self.title or "Project Section"


class ProjectItem(models.Model):
    """
    Stores one project displayed inside the Projects section.

    The Request Demo button is handled by the frontend modal,
    so no demo link is required here.
    """

    section = models.ForeignKey(
        ProjectSection,
        on_delete=models.CASCADE,
        related_name="items",
        help_text="The Projects section this item belongs to.",
    )

    title = models.CharField(
        max_length=180,
        help_text="Project title shown on the website.",
    )

    slug = models.SlugField(
        max_length=220,
        unique=True,
        blank=True,
        help_text="Unique project slug used internally and later for project detail pages.",
    )

    short_description = models.TextField(
        blank=True,
        default="",
        help_text="Short text shown on project cards.",
    )

    description = models.TextField(
        blank=True,
        default="",
        help_text="Full project description shown in the selected project detail card.",
    )

    thumbnail_image = models.ImageField(
        upload_to="projects/thumbnails/",
        blank=True,
        null=True,
        help_text="Small image used in project cards.",
    )

    hero_image = models.ImageField(
        upload_to="projects/hero/",
        blank=True,
        null=True,
        help_text="Large image used in the selected project hero preview.",
    )

    alt_text = models.CharField(
        max_length=180,
        blank=True,
        default="",
        help_text="Image alt text for accessibility.",
    )

    category = models.CharField(
        max_length=100,
        blank=True,
        default="AI Project",
        help_text="Project category badge, for example AI Dashboard or LLM Application.",
    )

    tech_stack = models.JSONField(
        blank=True,
        default=list,
        help_text="List of technologies, for example ['Django', 'React', 'PostgreSQL'].",
    )

    sort_order = models.PositiveIntegerField(
        default=0,
        help_text="Controls display order. Lower numbers appear first.",
    )

    is_featured = models.BooleanField(
        default=True,
        help_text="Featured projects are shown in the main Projects section.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only active projects are shown on the public website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Project Item"
        verbose_name_plural = "Project Items"
        ordering = ["sort_order", "created_at"]

    def save(self, *args, **kwargs):
        """
        Auto-generate slug and alt text when missing.
        """
        if not self.slug and self.title:
            from django.utils.text import slugify

            base_slug = slugify(self.title)
            slug = base_slug
            counter = 1

            while ProjectItem.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                counter += 1
                slug = f"{base_slug}-{counter}"

            self.slug = slug

        if not self.alt_text and self.title:
            self.alt_text = self.title

        super().save(*args, **kwargs)

    def __str__(self):
        return self.title


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
    Stores uploaded chatbot knowledge documents such as:
    CVs, certificates, project summaries, recommendation letters, etc.

    A document must pass extraction, validation, chunking, embedding,
    and admin approval before it becomes available for chatbot retrieval.
    """

    STATUS_CHOICES = [
        ("uploaded", "Uploaded"),

        ("extracting", "Extracting"),
        ("extracted", "Extracted"),
        ("extraction_failed", "Extraction Failed"),

        ("validating", "Validating"),
        ("validation_failed", "Validation Failed"),
        ("validation_warning", "Validation Warning"),

        ("chunking", "Chunking"),
        ("chunked", "Chunked"),
        ("chunking_failed", "Chunking Failed"),

        ("embedding", "Generating Embeddings"),
        ("embedded", "Embedded"),
        ("embedding_failed", "Embedding Failed"),

        ("ready_for_review", "Ready for Review"),

        ("approved", "Approved"),
        ("rejected", "Rejected"),
        ("archived", "Archived"),
    ]

    QUALITY_STATUS_CHOICES = [
        ("pending", "Pending"),
        ("passed", "Passed"),
        ("warning", "Warning"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    title = models.CharField(max_length=255)

    file = models.FileField(upload_to="profile_documents/")

    document_type = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Example: CV, certificate, project summary, recommendation letter.",
    )

    raw_text = models.TextField(
        blank=True,
        null=True,
        help_text="Full extracted text from the uploaded document.",
    )

    extracted_text_preview = models.TextField(
        blank=True,
        default="",
        help_text="Short preview of extracted text for admin list/detail display.",
    )

    status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default="uploaded",
        db_index=True,
    )

    quality_status = models.CharField(
        max_length=20,
        choices=QUALITY_STATUS_CHOICES,
        default="pending",
        db_index=True,
    )

    is_active = models.BooleanField(
        default=False,
        help_text="Only active and approved documents can be used by chatbot.",
    )

    is_approved = models.BooleanField(
        default=False,
        help_text="True only after admin approval.",
    )

    is_available_for_chatbot = models.BooleanField(
        default=False,
        help_text="Final safety flag used by chatbot retrieval.",
    )

    priority = models.PositiveSmallIntegerField(
        default=5,
        help_text="1 = highest priority, 10 = lowest priority.",
    )

    tags = models.JSONField(
        blank=True,
        null=True,
        help_text="List of tags, for example ['cv', 'python', 'django'].",
    )

    source_label = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Optional label such as CV 2026, LinkedIn export, Project Summary, etc.",
    )

    original_filename = models.CharField(
        max_length=255,
        blank=True,
        default="",
    )

    file_size = models.PositiveIntegerField(
        default=0,
        help_text="Uploaded file size in bytes.",
    )

    file_hash = models.CharField(
        max_length=128,
        blank=True,
        default="",
        db_index=True,
        help_text="Used to detect duplicate uploaded documents.",
    )

    mime_type = models.CharField(
        max_length=120,
        blank=True,
        default="",
    )

    extraction_score = models.FloatField(default=0)
    chunk_quality_score = models.FloatField(default=0)
    embedding_quality_score = models.FloatField(default=0)
    overall_quality_score = models.FloatField(default=0)

    validation_summary = models.JSONField(
        default=dict,
        blank=True,
        help_text="Stores document validation result summary.",
    )

    processing_metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Stores extraction/chunking/embedding metadata.",
    )

    admin_notes = models.TextField(
        blank=True,
        default="",
        help_text="Internal admin notes about this document.",
    )

    rejection_reason = models.TextField(
        blank=True,
        default="",
        help_text="Required when document is rejected.",
    )
    
    is_reviewed = models.BooleanField(
    default=False,
    help_text="Whether an admin has reviewed the extracted text, chunks, embeddings, and quality checks.",
    )

    reviewed_at = models.DateTimeField(
        blank=True,
        null=True,
        help_text="When the document was reviewed by an admin.",
    )

    review_notes = models.TextField(
        blank=True,
        default="",
        help_text="Admin notes added during document review.",
    )
        

    approved_at = models.DateTimeField(null=True, blank=True)
    rejected_at = models.DateTimeField(null=True, blank=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "Profile Document"
        verbose_name_plural = "Profile Documents"
        ordering = ["priority", "-updated_at"]
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["quality_status"]),
            models.Index(fields=["is_active", "is_approved",
                         "is_available_for_chatbot"]),
            models.Index(fields=["file_hash"]),
            models.Index(fields=["document_type"]),
        ]

    def __str__(self):
        return self.title


class DocumentChunk(TimeStampedModel):
    """
    Stores chunked pieces of a document for retrieval.

    Chunks should only become active when the parent document is approved.
    """

    QUALITY_STATUS_CHOICES = [
        ("pending", "Pending"),
        ("passed", "Passed"),
        ("warning", "Warning"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    document = models.ForeignKey(
        ProfileDocument,
        on_delete=models.CASCADE,
        related_name="chunks",
    )

    chunk_index = models.PositiveIntegerField()

    content = models.TextField()

    section_title = models.CharField(
        max_length=255,
        blank=True,
        null=True,
    )

    page_number = models.PositiveIntegerField(
        blank=True,
        null=True,
    )

    token_count = models.PositiveIntegerField(
        default=0,
        help_text="Approximate number of tokens or words in the chunk.",
    )

    character_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of characters in the chunk.",
    )

    embedding = VectorField(
        dimensions=1024,
        blank=True,
        null=True,
    )

    embedding_model = models.CharField(
        max_length=120,
        blank=True,
        default="",
        help_text="Name of the embedding model used.",
    )

    embedding_dimension = models.PositiveIntegerField(
        default=1024,
    )

    has_embedding = models.BooleanField(
        default=False,
        db_index=True,
    )

    quality_status = models.CharField(
        max_length=20,
        choices=QUALITY_STATUS_CHOICES,
        default="pending",
        db_index=True,
    )

    quality_score = models.FloatField(
        default=0,
    )

    quality_issues = models.JSONField(
        default=list,
        blank=True,
        help_text="List of detected chunk issues.",
    )

    is_active = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Only active chunks are used by chatbot retrieval.",
    )

    class Meta:
        ordering = ["document", "chunk_index"]
        unique_together = ("document", "chunk_index")
        indexes = [
            models.Index(fields=["document", "chunk_index"]),
            models.Index(fields=["has_embedding"]),
            models.Index(fields=["quality_status"]),
            models.Index(fields=["is_active"]),
        ]

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"


class DocumentQualityCheck(TimeStampedModel):
    """
    Stores quality-control checks for uploaded chatbot knowledge documents.

    Each check represents one validation result, such as:
    - extracted text exists
    - document is not duplicated
    - chunking completed
    - embeddings generated
    - OCR text quality is acceptable
    """

    CHECK_STATUS_CHOICES = [
        ("passed", "Passed"),
        ("warning", "Warning"),
        ("failed", "Failed"),
    ]

    SEVERITY_CHOICES = [
        ("info", "Info"),
        ("warning", "Warning"),
        ("error", "Error"),
        ("critical", "Critical"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    document = models.ForeignKey(
        ProfileDocument,
        on_delete=models.CASCADE,
        related_name="quality_checks",
    )

    check_name = models.CharField(max_length=120)

    check_status = models.CharField(
        max_length=20,
        choices=CHECK_STATUS_CHOICES,
        db_index=True,
    )

    severity = models.CharField(
        max_length=20,
        choices=SEVERITY_CHOICES,
        default="info",
        db_index=True,
    )

    message = models.TextField()

    details = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional structured details about the check result.",
    )

    class Meta:
        verbose_name = "Document Quality Check"
        verbose_name_plural = "Document Quality Checks"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["document"]),
            models.Index(fields=["check_status"]),
            models.Index(fields=["severity"]),
        ]

    def __str__(self):
        return f"{self.document.title} - {self.check_name} - {self.check_status}"


class DocumentProcessingLog(TimeStampedModel):
    """
    Stores processing logs for each chatbot knowledge document.

    This is useful for the admin panel timeline and debugging failed documents.
    """

    LEVEL_CHOICES = [
        ("info", "Info"),
        ("warning", "Warning"),
        ("error", "Error"),
        ("critical", "Critical"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    document = models.ForeignKey(
        ProfileDocument,
        on_delete=models.CASCADE,
        related_name="processing_logs",
    )

    step = models.CharField(
        max_length=120,
        help_text="Example: upload_started, extraction_completed, embedding_failed.",
    )

    level = models.CharField(
        max_length=20,
        choices=LEVEL_CHOICES,
        default="info",
        db_index=True,
    )

    message = models.TextField()

    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional structured log metadata.",
    )

    class Meta:
        verbose_name = "Document Processing Log"
        verbose_name_plural = "Document Processing Logs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["document"]),
            models.Index(fields=["level"]),
            models.Index(fields=["step"]),
        ]

    def __str__(self):
        return f"{self.document.title} - {self.step} - {self.level}"


class ProfileDocumentVersion(TimeStampedModel):
    """
    Stores previous versions of a ProfileDocument when a file is replaced.

    This allows safe rollback and prevents the chatbot from losing approved
    knowledge while a new version is still under review.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    document = models.ForeignKey(
        ProfileDocument,
        on_delete=models.CASCADE,
        related_name="versions",
    )

    version_number = models.PositiveIntegerField()

    file = models.FileField(upload_to="profile_documents/versions/")

    raw_text_snapshot = models.TextField(
        blank=True,
        default="",
    )

    status_snapshot = models.CharField(
        max_length=30,
        blank=True,
        default="",
    )

    quality_score_snapshot = models.FloatField(default=0)

    notes = models.TextField(
        blank=True,
        default="",
    )

    class Meta:
        verbose_name = "Profile Document Version"
        verbose_name_plural = "Profile Document Versions"
        ordering = ["-version_number"]
        unique_together = ["document", "version_number"]

    def __str__(self):
        return f"{self.document.title} - Version {self.version_number}"


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


class CertificateSection(models.Model):
    """
    Stores dynamic header content for the Certificates section.

    The public website displays the active CertificateSection record.
    Individual certificates are stored in CertificateItem.
    """

    title = models.CharField(
        max_length=120,
        default="Certificates",
        blank=True,
        help_text="Main title shown above the Certificates section.",
    )

    description = models.TextField(
        blank=True,
        default="Professional certifications and achievements",
        help_text="Short description shown below the Certificates section title.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only the active Certificates section will be displayed on the website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Certificate Section"
        verbose_name_plural = "Certificate Sections"
        ordering = ["-updated_at"]

    def save(self, *args, **kwargs):
        """
        Ensure only one CertificateSection is active at a time.
        """
        if self.is_active:
            CertificateSection.objects.exclude(
                pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return self.title or "Certificate Section"


class CertificateItem(models.Model):
    """
    Stores one certificate displayed inside the Certificates carousel.

    The public UI uses this data for:
    - main hero certificate preview
    - previous/next certificate previews
    - mobile thumbnails
    - preview modal
    - skills obtained badges
    """

    section = models.ForeignKey(
        CertificateSection,
        on_delete=models.CASCADE,
        related_name="items",
        help_text="The Certificates section this item belongs to.",
    )

    title = models.CharField(
        max_length=220,
        help_text="Full certificate title.",
    )

    mobile_title = models.CharField(
        max_length=80,
        blank=True,
        default="",
        help_text="Short title shown in mobile thumbnails and previous/next labels.",
    )

    slug = models.SlugField(
        max_length=240,
        unique=True,
        blank=True,
        help_text="Unique certificate slug used internally.",
    )

    issuer = models.CharField(
        max_length=160,
        blank=True,
        default="",
        help_text="Certificate issuer, for example Udemy, IBM via Coursera, Coursiv.",
    )

    issue_date = models.CharField(
        max_length=80,
        blank=True,
        default="",
        help_text="Display date, for example Jan 25, 2025 or 5 February 2026.",
    )

    certificate_image = models.ImageField(
        upload_to="certificates/images/",
        blank=True,
        null=True,
        help_text="Certificate image used in carousel and preview modal.",
    )

    certificate_file = models.FileField(
        upload_to="certificates/files/",
        blank=True,
        null=True,
        help_text="Optional PDF certificate file.",
    )

    alt_text = models.CharField(
        max_length=220,
        blank=True,
        default="",
        help_text="Image alt text for accessibility.",
    )

    skills = models.JSONField(
        blank=True,
        default=list,
        help_text="List of skills, for example ['React', 'Django', 'REST API'].",
    )

    sort_order = models.PositiveIntegerField(
        default=0,
        help_text="Controls display order. Lower numbers appear first.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only active certificates are shown on the public website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Certificate Item"
        verbose_name_plural = "Certificate Items"
        ordering = ["sort_order", "created_at"]

    def save(self, *args, **kwargs):
        """
        Auto-generate slug, mobile title, and alt text when missing.
        """
        if not self.slug and self.title:
            from django.utils.text import slugify

            base_slug = slugify(self.title)
            slug = base_slug
            counter = 1

            while CertificateItem.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                counter += 1
                slug = f"{base_slug}-{counter}"

            self.slug = slug

        if not self.mobile_title and self.title:
            self.mobile_title = self.title[:70]

        if not self.alt_text and self.title:
            self.alt_text = f"{self.title} certificate"

        super().save(*args, **kwargs)

    def __str__(self):
        return self.title


class ResearchSection(models.Model):
    """
    Stores dynamic header content for the Research section.

    The public website displays the active ResearchSection record.
    Individual research cards are stored in ResearchItem.
    """

    title = models.CharField(
        max_length=120,
        default="Research",
        blank=True,
        help_text="Main title shown above the Research section.",
    )

    description = models.TextField(
        blank=True,
        default="Published papers, articles, and academic contributions",
        help_text="Short description shown below the Research section title.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only the active Research section will be displayed on the website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Research Section"
        verbose_name_plural = "Research Sections"
        ordering = ["-updated_at"]

    def save(self, *args, **kwargs):
        """
        Ensure only one ResearchSection is active at a time.
        """
        if self.is_active:
            ResearchSection.objects.exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return self.title or "Research Section"


class ResearchItem(models.Model):
    """
    Stores one research/publication card displayed in the Research section.

    The public UI uses this data for:
    - research card title
    - type badge such as Conference Paper or Article
    - publish date
    - reads and citations
    - authors list
    - Read More link
    - Share link
    - featured image
    """

    section = models.ForeignKey(
        ResearchSection,
        on_delete=models.CASCADE,
        related_name="items",
        help_text="The Research section this item belongs to.",
    )

    title = models.CharField(
        max_length=500,
        help_text="Research paper/article title.",
    )

    slug = models.SlugField(
        max_length=520,
        unique=True,
        blank=True,
        help_text="Unique slug used internally.",
    )

    research_type = models.CharField(
        max_length=120,
        default="Article",
        help_text="Type badge shown on the card, for example Article or Conference Paper.",
    )

    publish_date = models.CharField(
        max_length=120,
        blank=True,
        default="",
        help_text="Display date, for example November 2022.",
    )

    reads = models.CharField(
        max_length=50,
        blank=True,
        default="0",
        help_text="Display reads count, for example 404.",
    )

    citations = models.CharField(
        max_length=50,
        blank=True,
        default="0",
        help_text="Display citations count, for example 2.",
    )

    authors = models.JSONField(
        blank=True,
        default=list,
        help_text="List of authors, for example ['Moaiad Khder', 'Samah Fujo'].",
    )

    primary_action = models.CharField(
        max_length=80,
        default="Read More",
        blank=True,
        help_text="Main action button text.",
    )

    primary_action_href = models.URLField(
        max_length=1000,
        blank=True,
        default="",
        help_text="Main action link, for example ResearchGate URL.",
    )

    share_href = models.URLField(
        max_length=1000,
        blank=True,
        default="",
        help_text="Share link. Usually same as the ResearchGate/publication URL.",
    )

    image = models.ImageField(
        upload_to="research/images/",
        blank=True,
        null=True,
        help_text="Optional uploaded image for the research card.",
    )

    external_image_url = models.URLField(
        max_length=1000,
        blank=True,
        default="",
        help_text="Optional external image URL, for example Unsplash image.",
    )

    alt_text = models.CharField(
        max_length=300,
        blank=True,
        default="",
        help_text="Image alt text for accessibility.",
    )

    sort_order = models.PositiveIntegerField(
        default=0,
        help_text="Controls display order. Lower numbers appear first.",
    )

    is_active = models.BooleanField(
        default=True,
        help_text="Only active research items are shown on the public website.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Research Item"
        verbose_name_plural = "Research Items"
        ordering = ["sort_order", "created_at"]

    def save(self, *args, **kwargs):
        """
        Auto-generate slug and alt text when missing.
        """
        if not self.slug and self.title:
            from django.utils.text import slugify

            base_slug = slugify(self.title)[:480] or "research-item"
            slug = base_slug
            counter = 1

            while ResearchItem.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                counter += 1
                slug = f"{base_slug}-{counter}"

            self.slug = slug

        if not self.alt_text and self.title:
            self.alt_text = f"{self.title} research image"

        super().save(*args, **kwargs)

    def __str__(self):
        return self.title


class ResearchStatsRefreshLog(models.Model):
    """
    Stores refresh attempts for ResearchGate stats.

    This helps track whether reads/citations were fetched successfully,
    failed, blocked, partially fetched, or unchanged.
    """

    STATUS_CHOICES = [
        ("success", "Success"),
        ("no_change", "No Change"),
        ("partial", "Partial"),
        ("failed", "Failed"),
        ("skipped", "Skipped"),
        ("manual", "Manual Update"),
    ]

    research_item = models.ForeignKey(
        ResearchItem,
        on_delete=models.CASCADE,
        related_name="stats_refresh_logs",
    )

    status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default="failed",
    )

    old_reads = models.CharField(max_length=50, blank=True, default="")
    new_reads = models.CharField(max_length=50, blank=True, default="")

    old_citations = models.CharField(max_length=50, blank=True, default="")
    new_citations = models.CharField(max_length=50, blank=True, default="")

    reads_fetched = models.BooleanField(default=False)
    citations_fetched = models.BooleanField(default=False)

    message = models.TextField(blank=True, default="")
    source_url = models.URLField(max_length=1000, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Research Stats Refresh Log"
        verbose_name_plural = "Research Stats Refresh Logs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.research_item.title} - {self.status} - {self.created_at}"


class FooterSection(models.Model):
    """
    Stores the main footer content.

    This model controls:
    - Follow title
    - Copyright owner/name
    - Active footer version
    """

    follow_title = models.CharField(
        max_length=120,
        default="Follow me",
        blank=True,
    )

    copyright_name = models.CharField(
        max_length=160,
        default="Samah Fujo",
        blank=True,
    )

    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Footer Section"
        verbose_name_plural = "Footer Sections"
        ordering = ["-updated_at"]

    def save(self, *args, **kwargs):
        """
        Ensure only one footer section is active at a time.
        """

        if self.is_active:
            FooterSection.objects.exclude(pk=self.pk).update(is_active=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return self.follow_title or "Footer Section"


class FooterSocialLink(models.Model):
    """
    Stores footer social media links.

    icon_key should match frontend supported icons:
    - linkedin
    - instagram
    - tiktok
    """

    ICON_CHOICES = [
        ("linkedin", "LinkedIn"),
        ("instagram", "Instagram"),
        ("tiktok", "TikTok"),
    ]

    section = models.ForeignKey(
        FooterSection,
        on_delete=models.CASCADE,
        related_name="social_links",
    )

    name = models.CharField(max_length=80)
    icon_key = models.CharField(
        max_length=40,
        choices=ICON_CHOICES,
        default="linkedin",
    )
    url = models.URLField(max_length=1000)

    sort_order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Footer Social Link"
        verbose_name_plural = "Footer Social Links"
        ordering = ["sort_order", "created_at"]

    def __str__(self):
        return self.name


class FooterContactItem(models.Model):
    """
    Stores footer contact information.

    icon_key should match frontend supported icons:
    - email
    - phone
    - location
    """

    ICON_CHOICES = [
        ("email", "Email"),
        ("phone", "Phone"),
        ("location", "Location"),
    ]

    section = models.ForeignKey(
        FooterSection,
        on_delete=models.CASCADE,
        related_name="contact_items",
    )

    label = models.CharField(max_length=80)
    value = models.CharField(max_length=255)
    href = models.CharField(
        max_length=1000,
        blank=True,
        default="",
        help_text="Example: mailto:s.fujo@hotmail.com or tel:+971527929218",
    )
    icon_key = models.CharField(
        max_length=40,
        choices=ICON_CHOICES,
        default="email",
    )

    sort_order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Footer Contact Item"
        verbose_name_plural = "Footer Contact Items"
        ordering = ["sort_order", "created_at"]

    def __str__(self):
        return f"{self.label}: {self.value}"
