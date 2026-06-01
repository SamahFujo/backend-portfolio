from .models import AboutSection
from .models import (
    ProfileDocument,
    DocumentChunk,
    ChatSession,
    ChatMessage,
    ContactMessage,
    ProjectRequest,
    ResearchSection,
    ResearchItem,
    FooterSection,
    FooterSocialLink,
    FooterContactItem,
)

import json
import re
from urllib.parse import urlparse


from rest_framework import serializers
from django.conf import settings
from pathlib import Path


from rest_framework import serializers
from .models import (
    HeroSection,
    AboutSection,
    SkillSection,
    SkillItem,
    ProjectSection,
    ProjectItem,
    CertificateSection,
    CertificateItem,
)


class HeroSectionSerializer(serializers.ModelSerializer):
    """
    Public serializer used by the website frontend.
    """

    hero_image_dark_url = serializers.SerializerMethodField()
    hero_image_light_url = serializers.SerializerMethodField()
    background_image_url = serializers.SerializerMethodField()

    class Meta:
        model = HeroSection
        fields = [
            "id",
            "eyebrow_text",
            "full_name",
            "headline",
            "description",
            "primary_button_text",
            "primary_button_url",
            "secondary_button_text",
            "secondary_button_url",
            "hero_image_dark_url",
            "hero_image_light_url",
            "background_image_url",

            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "hero_image_dark_url",
            "hero_image_light_url",
            "background_image_url",
        ]

    def get_hero_image_dark_url(self, obj):
        request = self.context.get("request")

        if obj.hero_image_dark:
            url = obj.hero_image_dark.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_hero_image_light_url(self, obj):
        request = self.context.get("request")

        if obj.hero_image_light:
            url = obj.hero_image_light.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_background_image_url(self, obj):
        request = self.context.get("request")

        if obj.background_image:
            url = obj.background_image.url
            return request.build_absolute_uri(url) if request else url

        return None


class HeroSectionAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer used for creating and updating Hero content.
    Supports image uploads from the custom admin dashboard.
    """

    hero_image_dark_url = serializers.SerializerMethodField()
    hero_image_light_url = serializers.SerializerMethodField()
    background_image_url = serializers.SerializerMethodField()

    class Meta:
        model = HeroSection
        fields = [
            "id",
            "eyebrow_text",
            "full_name",
            "headline",
            "description",
            "primary_button_text",
            "primary_button_url",
            "secondary_button_text",
            "secondary_button_url",
            "hero_image_dark",
            "hero_image_light",
            "hero_image_dark_url",
            "hero_image_light_url",
            "background_image_url",
            "is_active",
            "created_at",
            "updated_at",
        ]

    read_only_fields = [
        "id",
        "created_at",
        "updated_at",
        "hero_image_dark_url",
        "hero_image_light_url",
        "background_image_url",
    ]

    def get_hero_image_dark_url(self, obj):
        request = self.context.get("request")

        if obj.hero_image_dark:
            url = obj.hero_image_dark.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_hero_image_light_url(self, obj):
        request = self.context.get("request")

        if obj.hero_image_light:
            url = obj.hero_image_light.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_background_image_url(self, obj):
        request = self.context.get("request")

        if obj.background_image:
            url = obj.background_image.url
            return request.build_absolute_uri(url) if request else url

        return None


class AboutSectionSerializer(serializers.ModelSerializer):
    """
    Public serializer used by the website frontend to display
    the active About Me section.
    """

    class Meta:
        model = AboutSection
        fields = [
            "id",
            "section_title",
            "terminal_label",
            "welcome_title",
            "description",
            "is_active",
            "updated_at",
        ]


class AboutSectionAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer used by the custom admin dashboard
    to create and update the About Me section.
    """

    class Meta:
        model = AboutSection
        fields = [
            "id",
            "section_title",
            "terminal_label",
            "welcome_title",
            "description",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
        ]


class SkillItemSerializer(serializers.ModelSerializer):
    """
    Public serializer for one skill card.
    """

    class Meta:
        model = SkillItem
        fields = [
            "id",
            "section",
            "category",
            "icon",
            "label",
            "level",
            "summary_heading",
            "summary_text",
            "summary_points",
            "sort_order",
            "is_active",
            "created_at",
            "updated_at",
        ]


class SkillSectionSerializer(serializers.ModelSerializer):
    """
    Public serializer used by the website frontend to display
    the active Skills section with its active skill cards.
    """

    items = serializers.SerializerMethodField()

    class Meta:
        model = SkillSection
        fields = [
            "id",
            "category",
            "icon",
            "label",
            "level",
            "summary_heading",
            "summary_text",
            "summary_points",
            "sort_order",
            "is_active",
        ]

    def get_items(self, obj):
        items = obj.items.filter(is_active=True).order_by(
            "sort_order", "created_at")
        return SkillItemSerializer(items, many=True).data


class SkillItemSerializer(serializers.ModelSerializer):
    """
    Public serializer for one skill item/card.
    Used by the public portfolio Skills section.
    """

    class Meta:
        model = SkillItem
        fields = [
            "id",
            "category",
            "icon",
            "label",
            "level",
            "summary_heading",
            "summary_text",
            "summary_points",
            "sort_order",
            "is_active",
        ]


class SkillSectionSerializer(serializers.ModelSerializer):
    """
    Public serializer used by the website frontend to display
    the active Skills section with active skill items.
    """

    items = serializers.SerializerMethodField()

    class Meta:
        model = SkillSection
        fields = [
            "id",
            "badge_text",
            "title_line_1",
            "title_line_2",
            "description",
            "items",
            "is_active",
            "updated_at",
        ]

    def get_items(self, obj):
        items = obj.items.filter(is_active=True).order_by(
            "category",
            "sort_order",
            "created_at",
        )
        return SkillItemSerializer(items, many=True).data


class SkillItemAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating skill items.
    Used by the custom admin dashboard.
    """

    class Meta:
        model = SkillItem
        fields = [
            "id",
            "section",
            "category",
            "icon",
            "label",
            "level",
            "summary_heading",
            "summary_text",
            "summary_points",
            "sort_order",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
        ]


class SkillSectionAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for managing the Skills section header
    and returning related skill items.
    """

    items = SkillItemAdminSerializer(many=True, read_only=True)

    class Meta:
        model = SkillSection
        fields = [
            "id",
            "badge_text",
            "title_line_1",
            "title_line_2",
            "description",
            "items",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "items",
            "created_at",
            "updated_at",
        ]


class ProjectItemSerializer(serializers.ModelSerializer):
    """
    Public serializer for one project item.
    Used by the public website Projects section.
    """

    thumbnail_image_url = serializers.SerializerMethodField()
    hero_image_url = serializers.SerializerMethodField()

    class Meta:
        model = ProjectItem
        fields = [
            "id",
            "slug",
            "title",
            "short_description",
            "description",
            "thumbnail_image_url",
            "hero_image_url",
            "alt_text",
            "category",
            "tech_stack",
            "sort_order",
            "is_featured",
            "is_active",
        ]

    def get_thumbnail_image_url(self, obj):
        request = self.context.get("request")

        if obj.thumbnail_image:
            url = obj.thumbnail_image.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_hero_image_url(self, obj):
        request = self.context.get("request")

        if obj.hero_image:
            url = obj.hero_image.url
            return request.build_absolute_uri(url) if request else url

        return None


class ProjectSectionSerializer(serializers.ModelSerializer):
    """
    Public serializer used by the website frontend to display
    the active Projects section with active featured projects.
    """

    items = serializers.SerializerMethodField()

    class Meta:
        model = ProjectSection
        fields = [
            "id",
            "title",
            "description",
            "items",
            "is_active",
            "updated_at",
        ]

    def get_items(self, obj):
        items = obj.items.filter(
            is_active=True,
            is_featured=True,
        ).order_by("sort_order", "created_at")

        return ProjectItemSerializer(
            items,
            many=True,
            context=self.context,
        ).data


class ProjectItemAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating project items.
    Supports image uploads from the custom admin dashboard.
    """

    thumbnail_image_url = serializers.SerializerMethodField()
    hero_image_url = serializers.SerializerMethodField()

    class Meta:
        model = ProjectItem
        fields = [
            "id",
            "section",
            "slug",
            "title",
            "short_description",
            "description",
            "thumbnail_image",
            "hero_image",
            "thumbnail_image_url",
            "hero_image_url",
            "alt_text",
            "category",
            "tech_stack",
            "sort_order",
            "is_featured",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "thumbnail_image_url",
            "hero_image_url",
        ]

    def get_thumbnail_image_url(self, obj):
        request = self.context.get("request")

        if obj.thumbnail_image:
            url = obj.thumbnail_image.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_hero_image_url(self, obj):
        request = self.context.get("request")

        if obj.hero_image:
            url = obj.hero_image.url
            return request.build_absolute_uri(url) if request else url

        return None


class ProjectSectionAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for managing the Projects section header
    and returning related project items.
    """

    items = ProjectItemAdminSerializer(many=True, read_only=True)

    class Meta:
        model = ProjectSection
        fields = [
            "id",
            "title",
            "description",
            "items",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "items",
            "created_at",
            "updated_at",
        ]


"""
Serializers for the public "Start Project" form API endpoint.
"""


class StartProjectRequestSerializer(serializers.Serializer):
    projectName = serializers.CharField(min_length=2, max_length=200)
    projectType = serializers.CharField(
        required=False, allow_blank=True, max_length=100)
    budgetRange = serializers.CharField(
        required=False, allow_blank=True, max_length=100)
    timeline = serializers.CharField(
        required=False, allow_blank=True, max_length=100)
    projectDescription = serializers.CharField(min_length=20)
    yourName = serializers.CharField(min_length=2, max_length=120)
    yourEmail = serializers.EmailField()
    website = serializers.CharField(
        required=False, allow_blank=True, max_length=200)

    def validate(self, attrs):
        if attrs.get("website"):
            raise serializers.ValidationError("Spam detected.")
        return attrs


"""
Serializers for the public "Get in Touch" form API endpoint.
"""


class GetInTouchSerializer(serializers.Serializer):
    name = serializers.CharField(min_length=2, max_length=120)
    message = serializers.CharField(min_length=10, max_length=5000)
    subject = serializers.CharField(
        max_length=200, required=False, allow_blank=True)
    message = serializers.CharField(max_length=5000)

    # Simple “honeypot” anti-bot field (frontend keeps it hidden)
    website = serializers.CharField(
        required=False, allow_blank=True, max_length=200)

    def validate(self, attrs):
        # Honeypot: if filled, likely bot
        if attrs.get("website"):
            raise serializers.ValidationError("Spam detected.")
        return attrs


"""
Serializers for chatbot-related APIs.
"""


class ProfileDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProfileDocument
        fields = "__all__"


class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = "__all__"


class ProfileDocumentUploadSerializer(serializers.ModelSerializer):
    def validate_file(self, value):
        extension = Path(value.name).suffix.lower()
        allowed_extensions = getattr(
            settings,
            "ALLOWED_DOCUMENT_EXTENSIONS",
            {".pdf", ".docx", ".txt"},
        )
        if extension not in allowed_extensions:
            allowed = ", ".join(sorted(allowed_extensions))
            raise serializers.ValidationError(
                f"Unsupported file type. Allowed types: {allowed}."
            )

        max_size = getattr(
            settings, "MAX_DOCUMENT_UPLOAD_SIZE", 5 * 1024 * 1024)
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File too large. Maximum size is {max_size // (1024 * 1024)} MB."
            )

        return value

    class Meta:
        model = ProfileDocument
        fields = ["id", "title", "file", "document_type", "source_label"]


class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = "__all__"


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = "__all__"


class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = "__all__"


class AskQuestionSerializer(serializers.Serializer):
    session_id = serializers.UUIDField(required=False, allow_null=True)
    visitor_id = serializers.CharField(
        required=False, allow_blank=True, allow_null=True)
    message = serializers.CharField()
    visitor_email = serializers.EmailField(required=True)


class AdminChatMessageSerializer(serializers.ModelSerializer):
    """
    Serializer for displaying individual chat messages in the admin panel.
    """

    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "role",
            "content",
            "citations",
            "confidence_score",
            "metadata",
            "created_at",
        ]


class AdminChatSessionListSerializer(serializers.ModelSerializer):
    """
    Serializer for listing chat sessions in the admin panel.
    """

    messages_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = [
            "id",
            "visitor_id",
            "visitor_email",
            "ip_address",
            "user_agent",
            "referrer",
            "is_active",
            "messages_count",
            "last_message",
            "created_at",
            "updated_at",
        ]

    def get_messages_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        last_msg = obj.messages.order_by("-created_at").first()

        if not last_msg:
            return None

        return {
            "role": last_msg.role,
            "content": last_msg.content[:150],
            "created_at": last_msg.created_at,
        }


class AdminChatSessionDetailSerializer(serializers.ModelSerializer):
    """
    Serializer for viewing a full conversation session.
    """

    messages = AdminChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = [
            "id",
            "visitor_id",
            "visitor_email",
            "ip_address",
            "user_agent",
            "referrer",
            "is_active",
            "created_at",
            "updated_at",
            "messages",
        ]


class AdminContactMessageSerializer(serializers.ModelSerializer):
    """
    Serializer for displaying contact form messages in the admin dashboard.
    """

    class Meta:
        model = ContactMessage
        fields = [
            "id",
            "name",
            "email",
            "subject",
            "message",
            "status",
            "ip_address",
            "user_agent",
            "referrer",
            "created_at",
            "updated_at",
        ]


class AdminProjectRequestSerializer(serializers.ModelSerializer):
    """
    Serializer for displaying start-project form requests in the admin dashboard.
    """

    class Meta:
        model = ProjectRequest
        fields = [
            "id",
            "project_name",
            "project_type",
            "budget_range",
            "timeline",
            "project_description",
            "your_name",
            "your_email",
            "status",
            "ip_address",
            "user_agent",
            "referrer",
            "created_at",
            "updated_at",
        ]


class RequestEmailVerificationSerializer(serializers.Serializer):
    email = serializers.EmailField()


class VerifyEmailCodeSerializer(serializers.Serializer):
    email = serializers.EmailField()
    code = serializers.CharField(min_length=6, max_length=6)


class CertificateItemSerializer(serializers.ModelSerializer):
    """
    Public serializer for one certificate item.
    Used by the public website Certificates section.
    """

    certificate_image_url = serializers.SerializerMethodField()
    certificate_file_url = serializers.SerializerMethodField()

    class Meta:
        model = CertificateItem
        fields = [
            "id",
            "slug",
            "title",
            "mobile_title",
            "issuer",
            "issue_date",
            "certificate_image_url",
            "certificate_file_url",
            "alt_text",
            "skills",
            "sort_order",
            "is_active",
        ]

    def get_certificate_image_url(self, obj):
        request = self.context.get("request")

        if obj.certificate_image:
            url = obj.certificate_image.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_certificate_file_url(self, obj):
        request = self.context.get("request")

        if obj.certificate_file:
            url = obj.certificate_file.url
            return request.build_absolute_uri(url) if request else url

        return None


class CertificateSectionSerializer(serializers.ModelSerializer):
    """
    Public serializer used by the website frontend to display
    the active Certificates section with active certificate items.
    """

    items = serializers.SerializerMethodField()

    class Meta:
        model = CertificateSection
        fields = [
            "id",
            "title",
            "description",
            "items",
            "is_active",
            "updated_at",
        ]

    def get_items(self, obj):
        items = obj.items.filter(is_active=True).order_by(
            "sort_order",
            "created_at",
        )

        return CertificateItemSerializer(
            items,
            many=True,
            context=self.context,
        ).data


class CertificateItemAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating certificate items.
    Supports image and PDF uploads from the custom admin dashboard.
    """

    certificate_image_url = serializers.SerializerMethodField()
    certificate_file_url = serializers.SerializerMethodField()

    class Meta:
        model = CertificateItem
        fields = [
            "id",
            "section",
            "slug",
            "title",
            "mobile_title",
            "issuer",
            "issue_date",
            "certificate_image",
            "certificate_file",
            "certificate_image_url",
            "certificate_file_url",
            "alt_text",
            "skills",
            "sort_order",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "certificate_image_url",
            "certificate_file_url",
        ]

    def get_certificate_image_url(self, obj):
        request = self.context.get("request")

        if obj.certificate_image:
            url = obj.certificate_image.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_certificate_file_url(self, obj):
        request = self.context.get("request")

        if obj.certificate_file:
            url = obj.certificate_file.url
            return request.build_absolute_uri(url) if request else url

        return None


class CertificateSectionAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for managing the Certificates section header
    and returning related certificate items.
    """

    items = CertificateItemAdminSerializer(many=True, read_only=True)

    class Meta:
        model = CertificateSection
        fields = [
            "id",
            "title",
            "description",
            "items",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "items",
            "created_at",
            "updated_at",
        ]


class ResearchItemSerializer(serializers.ModelSerializer):
    """
    Public serializer for one research item.
    Used by the public website Research section.
    """

    image_url = serializers.SerializerMethodField()

    class Meta:
        model = ResearchItem
        fields = [
            "id",
            "slug",
            "title",
            "research_type",
            "publish_date",
            "reads",
            "citations",
            "authors",
            "primary_action",
            "primary_action_href",
            "share_href",
            "image_url",
            "external_image_url",
            "alt_text",
            "sort_order",
            "is_active",
        ]

    def get_image_url(self, obj):
        request = self.context.get("request")

        if obj.image:
            url = obj.image.url
            return request.build_absolute_uri(url) if request else url

        if obj.external_image_url:
            return obj.external_image_url

        return None


class ResearchSectionSerializer(serializers.ModelSerializer):
    """
    Public serializer used by the website frontend to display
    the active Research section with active research items.
    """

    items = serializers.SerializerMethodField()

    class Meta:
        model = ResearchSection
        fields = [
            "id",
            "title",
            "description",
            "items",
            "is_active",
            "updated_at",
        ]

    def get_items(self, obj):
        items = obj.items.filter(is_active=True).order_by(
            "sort_order",
            "created_at",
        )

        return ResearchItemSerializer(
            items,
            many=True,
            context=self.context,
        ).data


class ResearchItemAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating research cards.

    This serializer provides user-friendly validation for:
    - Required fields
    - Valid URLs
    - Numeric reads/citations
    - Authors format
    - Uploaded image or external image URL
    - Duplicate slug
    - Sort order
    """

    image_url = serializers.SerializerMethodField()
    latest_refresh_log = serializers.SerializerMethodField()

    class Meta:
        model = ResearchItem
        fields = [
            "id",
            "section",
            "slug",
            "title",
            "research_type",
            "publish_date",
            "reads",
            "citations",
            "authors",
            "primary_action",
            "primary_action_href",
            "share_href",
            "image",
            "image_url",
            "external_image_url",
            "alt_text",
            "sort_order",
            "is_active",
            "latest_refresh_log",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "image_url",
            "latest_refresh_log",
        ]

        extra_kwargs = {
            "title": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the research title.",
                    "required": "Research title is required.",
                },
            },
            "research_type": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the research type, for example Article or Conference Paper.",
                    "required": "Research type is required.",
                },
            },
            "publish_date": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the publish date, for example November 2022.",
                    "required": "Publish date is required.",
                },
            },
            "primary_action": {
                "required": False,
                "allow_blank": True,
            },
            "primary_action_href": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the ResearchGate or publication URL.",
                    "required": "Publication URL is required.",
                    "invalid": "Please enter a valid publication URL.",
                },
            },
            "share_href": {
                "required": False,
                "allow_blank": True,
                "error_messages": {
                    "invalid": "Please enter a valid share URL.",
                },
            },
            "external_image_url": {
                "required": False,
                "allow_blank": True,
                "error_messages": {
                    "invalid": "Please enter a valid image URL.",
                },
            },
            "alt_text": {
                "required": False,
                "allow_blank": True,
            },
        }

    def get_image_url(self, obj):
        request = self.context.get("request")

        if obj.image:
            url = obj.image.url
            return request.build_absolute_uri(url) if request else url

        if obj.external_image_url:
            return obj.external_image_url

        return None

    def get_latest_refresh_log(self, obj):
        """
        Return the latest ResearchGate refresh attempt for this item.
        """

        latest_log = obj.stats_refresh_logs.order_by("-created_at").first()

        if not latest_log:
            return None

        return {
            "status": latest_log.status,
            "old_reads": latest_log.old_reads,
            "new_reads": latest_log.new_reads,
            "old_citations": latest_log.old_citations,
            "new_citations": latest_log.new_citations,
            "reads_fetched": latest_log.reads_fetched,
            "citations_fetched": latest_log.citations_fetched,
            "message": latest_log.message,
            "source_url": latest_log.source_url,
            "created_at": latest_log.created_at,
        }

    def validate_slug(self, value):
        """
        Slug is optional because the model can auto-generate it.
        If provided, validate format and uniqueness.
        """

        if not value:
            return value

        clean_value = value.strip().lower()

        if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", clean_value):
            raise serializers.ValidationError(
                "Slug can only contain lowercase letters, numbers, and hyphens. Example: my-research-paper"
            )

        existing_qs = ResearchItem.objects.filter(slug=clean_value)

        if self.instance:
            existing_qs = existing_qs.exclude(pk=self.instance.pk)

        if existing_qs.exists():
            raise serializers.ValidationError(
                "This slug is already used by another research item. Please choose a different slug."
            )

        return clean_value

    def validate_title(self, value):
        value = value.strip()

        if len(value) < 10:
            raise serializers.ValidationError(
                "Research title is too short. Please enter the full publication title."
            )

        if len(value) > 500:
            raise serializers.ValidationError(
                "Research title is too long. Please keep it under 500 characters."
            )

        return value

    def validate_research_type(self, value):
        value = value.strip()

        allowed_types = {
            "Article",
            "Conference Paper",
            "Journal Article",
            "Book Chapter",
            "Preprint",
            "Thesis",
            "Report",
        }

        if value not in allowed_types:
            raise serializers.ValidationError(
                "Please choose one of: Article, Conference Paper, Journal Article, Book Chapter, Preprint, Thesis, or Report."
            )

        return value

    def validate_reads(self, value):
        value = str(value).strip()

        if not value:
            return "0"

        if not re.match(r"^\d+$", value):
            raise serializers.ValidationError(
                "Reads must be a whole number only. Example: 404"
            )

        return value

    def validate_citations(self, value):
        value = str(value).strip()

        if not value:
            return "0"

        if not re.match(r"^\d+$", value):
            raise serializers.ValidationError(
                "Citations must be a whole number only. Example: 2"
            )

        return value

    def validate_sort_order(self, value):
        if value is None:
            return 0

        if value < 0:
            raise serializers.ValidationError(
                "Sort order cannot be negative. Use 1 for the first item, 2 for the second, and so on."
            )

        return value

    def validate_authors(self, value):
        """
        Handles:
        - JSON list from application/json
        - JSON string from multipart/form-data
        """

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise serializers.ValidationError(
                    "Authors must be a valid list. Add one author per line in the admin form."
                )

        if not isinstance(value, list):
            raise serializers.ValidationError(
                "Authors must be a list of names.")

        cleaned_authors = []

        for author in value:
            if not isinstance(author, str):
                raise serializers.ValidationError(
                    "Each author must be written as text."
                )

            clean_author = author.strip()

            if clean_author:
                cleaned_authors.append(clean_author)

        if not cleaned_authors:
            raise serializers.ValidationError(
                "Please add at least one author.")

        if len(cleaned_authors) > 20:
            raise serializers.ValidationError(
                "Too many authors. Please keep the list under 20 authors."
            )

        return cleaned_authors

    def validate_primary_action_href(self, value):
        return self._validate_url(
            value=value,
            field_name="Publication URL",
            required=True,
        )

    def validate_share_href(self, value):
        return self._validate_url(
            value=value,
            field_name="Share URL",
            required=False,
        )

    def validate_external_image_url(self, value):
        value = self._validate_url(
            value=value,
            field_name="External image URL",
            required=False,
        )

        if value:
            image_extensions = (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".gif",
                ".avif",
            )

            parsed_url = urlparse(value)
            path = parsed_url.path.lower()

            has_image_extension = path.endswith(image_extensions)
            is_unsplash_or_remote_image = "images.unsplash.com" in parsed_url.netloc

            if not has_image_extension and not is_unsplash_or_remote_image:
                raise serializers.ValidationError(
                    "External image URL should point to an image file such as .jpg, .png, .webp, or an allowed image service like Unsplash."
                )

        return value

    def validate_image(self, value):
        if not value:
            return value

        max_size_mb = 5
        max_size_bytes = max_size_mb * 1024 * 1024

        if value.size > max_size_bytes:
            raise serializers.ValidationError(
                f"Image is too large. Please upload an image smaller than {max_size_mb}MB."
            )

        allowed_content_types = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
        ]

        content_type = getattr(value, "content_type", "")

        if content_type not in allowed_content_types:
            raise serializers.ValidationError(
                "Unsupported image type. Please upload JPG, PNG, WEBP, or GIF."
            )

        return value

    def validate(self, attrs):
        """
        Cross-field validation.
        """

        image = attrs.get("image", getattr(self.instance, "image", None))
        external_image_url = attrs.get(
            "external_image_url",
            getattr(self.instance, "external_image_url", ""),
        )

        primary_action = attrs.get(
            "primary_action",
            getattr(self.instance, "primary_action", ""),
        )

        primary_action_href = attrs.get(
            "primary_action_href",
            getattr(self.instance, "primary_action_href", ""),
        )

        share_href = attrs.get(
            "share_href",
            getattr(self.instance, "share_href", ""),
        )

        if not image and not external_image_url:
            raise serializers.ValidationError(
                {
                    "external_image_url": [
                        "Please upload an image or provide an external image URL."
                    ],
                    "image": [
                        "Please upload an image or provide an external image URL."
                    ],
                }
            )

        if not primary_action:
            attrs["primary_action"] = "Read More"

        if not share_href and primary_action_href:
            attrs["share_href"] = primary_action_href

        if not attrs.get("alt_text"):
            title = attrs.get("title") or getattr(self.instance, "title", "")
            attrs["alt_text"] = f"{title} research image"

        return attrs

    def _validate_url(self, value, field_name: str, required: bool = False):
        value = (value or "").strip()

        if not value:
            if required:
                raise serializers.ValidationError(f"{field_name} is required.")
            return ""

        parsed_url = urlparse(value)

        if parsed_url.scheme not in ["http", "https"] or not parsed_url.netloc:
            raise serializers.ValidationError(
                f"{field_name} must be a valid URL starting with http:// or https://."
            )

        return value


class ResearchSectionAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for managing the Research section header
    and returning related research items.
    """

    items = ResearchItemAdminSerializer(many=True, read_only=True)

    class Meta:
        model = ResearchSection
        fields = [
            "id",
            "title",
            "description",
            "items",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "items",
            "created_at",
            "updated_at",
        ]

        extra_kwargs = {
            "title": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the Research section title.",
                    "required": "Research section title is required.",
                },
            },
            "description": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter a short description for the Research section.",
                    "required": "Research section description is required.",
                },
            },
        }

    def validate_title(self, value):
        value = value.strip()

        if len(value) < 3:
            raise serializers.ValidationError(
                "Section title is too short. Example: Research"
            )

        if len(value) > 120:
            raise serializers.ValidationError(
                "Section title is too long. Please keep it under 120 characters."
            )

        return value

    def validate_description(self, value):
        value = value.strip()

        if len(value) < 10:
            raise serializers.ValidationError(
                "Section description is too short. Please write a clear description."
            )

        if len(value) > 500:
            raise serializers.ValidationError(
                "Section description is too long. Please keep it under 500 characters."
            )

        return value


class FooterSocialLinkSerializer(serializers.ModelSerializer):
    """
    Public serializer for footer social links.
    Used by the website footer.
    """

    class Meta:
        model = FooterSocialLink
        fields = [
            "id",
            "name",
            "icon_key",
            "url",
            "sort_order",
            "is_active",
        ]


class FooterContactItemSerializer(serializers.ModelSerializer):
    """
    Public serializer for footer contact items.
    Used by the website footer.
    """

    class Meta:
        model = FooterContactItem
        fields = [
            "id",
            "label",
            "value",
            "href",
            "icon_key",
            "sort_order",
            "is_active",
        ]


class FooterSectionSerializer(serializers.ModelSerializer):
    """
    Public serializer for the active footer section.
    Only active social links and contact items are returned.
    """

    social_links = serializers.SerializerMethodField()
    contact_items = serializers.SerializerMethodField()

    class Meta:
        model = FooterSection
        fields = [
            "id",
            "follow_title",
            "copyright_name",
            "social_links",
            "contact_items",
            "is_active",
            "created_at",
            "updated_at",
        ]

    def get_social_links(self, obj):
        social_links = obj.social_links.filter(is_active=True).order_by(
            "sort_order",
            "created_at",
        )

        return FooterSocialLinkSerializer(
            social_links,
            many=True,
            context=self.context,
        ).data

    def get_contact_items(self, obj):
        contact_items = obj.contact_items.filter(is_active=True).order_by(
            "sort_order",
            "created_at",
        )

        return FooterContactItemSerializer(
            contact_items,
            many=True,
            context=self.context,
        ).data


class FooterSocialLinkAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating footer social links.

    Validates:
    - Social platform name
    - Supported icon key
    - Valid URL
    - Sort order
    """

    class Meta:
        model = FooterSocialLink
        fields = [
            "id",
            "section",
            "name",
            "icon_key",
            "url",
            "sort_order",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
        ]

        extra_kwargs = {
            "name": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the social platform name.",
                    "required": "Social platform name is required.",
                },
            },
            "icon_key": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please choose a social icon.",
                    "required": "Social icon is required.",
                    "invalid_choice": "Please choose a supported icon: LinkedIn, Instagram, or TikTok.",
                },
            },
            "url": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the social profile URL.",
                    "required": "Social profile URL is required.",
                    "invalid": "Please enter a valid social profile URL.",
                },
            },
        }

    def validate_name(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError(
                "Social platform name is too short. Example: LinkedIn"
            )

        if len(value) > 80:
            raise serializers.ValidationError(
                "Social platform name is too long. Please keep it under 80 characters."
            )

        allowed_names = {
            "linkedin",
            "instagram",
            "tiktok",
        }

        if value.lower() not in allowed_names:
            raise serializers.ValidationError(
                "Please use one of the supported social platforms: LinkedIn, Instagram, or TikTok."
            )

        return value

    def validate_icon_key(self, value):
        value = value.strip().lower()

        allowed_icons = {
            "linkedin",
            "instagram",
            "tiktok",
        }

        if value not in allowed_icons:
            raise serializers.ValidationError(
                "Unsupported icon. Please choose linkedin, instagram, or tiktok."
            )

        return value

    def validate_url(self, value):
        value = value.strip()

        self._validate_http_url(
            value=value,
            field_name="Social profile URL",
        )

        icon_key = self.initial_data.get("icon_key", "")
        icon_key = str(icon_key).strip().lower()

        parsed_url = urlparse(value)
        domain = parsed_url.netloc.lower()

        if icon_key == "linkedin" and "linkedin.com" not in domain:
            raise serializers.ValidationError(
                "LinkedIn URL must be from linkedin.com."
            )

        if icon_key == "instagram" and "instagram.com" not in domain:
            raise serializers.ValidationError(
                "Instagram URL must be from instagram.com."
            )

        if icon_key == "tiktok" and "tiktok.com" not in domain:
            raise serializers.ValidationError(
                "TikTok URL must be from tiktok.com."
            )

        return value

    def validate_sort_order(self, value):
        if value is None:
            return 0

        if value < 0:
            raise serializers.ValidationError(
                "Sort order cannot be negative. Use 1 for the first icon, 2 for the second, and so on."
            )

        return value

    def validate(self, attrs):
        name = attrs.get("name", getattr(self.instance, "name", ""))
        icon_key = attrs.get("icon_key", getattr(
            self.instance, "icon_key", ""))

        if name and icon_key:
            normalized_name = name.strip().lower()
            normalized_icon = icon_key.strip().lower()

            expected_pairs = {
                "linkedin": "linkedin",
                "instagram": "instagram",
                "tiktok": "tiktok",
            }

            expected_icon = expected_pairs.get(normalized_name)

            if expected_icon and expected_icon != normalized_icon:
                raise serializers.ValidationError(
                    {
                        "icon_key": [
                            f"The selected icon does not match {name}. Please choose {expected_icon}."
                        ]
                    }
                )

        return attrs

    def _validate_http_url(self, value, field_name):
        parsed_url = urlparse(value)

        if parsed_url.scheme not in ["http", "https"] or not parsed_url.netloc:
            raise serializers.ValidationError(
                f"{field_name} must be a valid URL starting with http:// or https://."
            )


class FooterContactItemAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating footer contact items.

    Validates:
    - Email format
    - Phone format
    - Location text
    - Correct href format
    - Matching icon key
    - Sort order
    """

    class Meta:
        model = FooterContactItem
        fields = [
            "id",
            "section",
            "label",
            "value",
            "href",
            "icon_key",
            "sort_order",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
        ]

        extra_kwargs = {
            "label": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the contact label.",
                    "required": "Contact label is required.",
                },
            },
            "value": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the contact value.",
                    "required": "Contact value is required.",
                },
            },
            "icon_key": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please choose a contact icon.",
                    "required": "Contact icon is required.",
                    "invalid_choice": "Please choose a supported icon: Email, Phone, or Location.",
                },
            },
            "href": {
                "required": False,
                "allow_blank": True,
            },
        }

    def validate_label(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError(
                "Contact label is too short. Example: Email"
            )

        if len(value) > 80:
            raise serializers.ValidationError(
                "Contact label is too long. Please keep it under 80 characters."
            )

        allowed_labels = {
            "email",
            "phone",
            "location",
        }

        if value.lower() not in allowed_labels:
            raise serializers.ValidationError(
                "Please use one of the supported contact labels: Email, Phone, or Location."
            )

        return value

    def validate_icon_key(self, value):
        value = value.strip().lower()

        allowed_icons = {
            "email",
            "phone",
            "location",
        }

        if value not in allowed_icons:
            raise serializers.ValidationError(
                "Unsupported contact icon. Please choose email, phone, or location."
            )

        return value

    def validate_value(self, value):
        value = value.strip()

        label = str(self.initial_data.get("label", "")).strip().lower()
        icon_key = str(self.initial_data.get("icon_key", "")).strip().lower()
        contact_type = icon_key or label

        if contact_type == "email":
            self._validate_email(value)

        elif contact_type == "phone":
            self._validate_phone(value)

        elif contact_type == "location":
            self._validate_location(value)

        else:
            if len(value) < 2:
                raise serializers.ValidationError(
                    "Contact value is too short."
                )

        return value

    def validate_href(self, value):
        value = (value or "").strip()

        label = str(self.initial_data.get("label", "")).strip().lower()
        icon_key = str(self.initial_data.get("icon_key", "")).strip().lower()
        contact_type = icon_key or label

        if not value:
            return ""

        if contact_type == "email":
            if not value.startswith("mailto:"):
                raise serializers.ValidationError(
                    "Email link must start with mailto:. Example: mailto:s.fujo@hotmail.com"
                )

            email_part = value.replace("mailto:", "", 1).strip()
            self._validate_email(email_part)

        elif contact_type == "phone":
            if not value.startswith("tel:"):
                raise serializers.ValidationError(
                    "Phone link must start with tel:. Example: tel:+971527929218"
                )

            phone_part = value.replace("tel:", "", 1).strip()
            self._validate_phone(phone_part)

        elif contact_type == "location":
            parsed_url = urlparse(value)

            if parsed_url.scheme not in ["http", "https"] or not parsed_url.netloc:
                raise serializers.ValidationError(
                    "Location link must be a valid URL starting with http:// or https://, or leave it empty."
                )

        return value

    def validate_sort_order(self, value):
        if value is None:
            return 0

        if value < 0:
            raise serializers.ValidationError(
                "Sort order cannot be negative. Use 1 for the first contact item, 2 for the second, and so on."
            )

        return value

    def validate(self, attrs):
        label = attrs.get("label", getattr(self.instance, "label", ""))
        icon_key = attrs.get("icon_key", getattr(
            self.instance, "icon_key", ""))
        value = attrs.get("value", getattr(self.instance, "value", ""))
        href = attrs.get("href", getattr(self.instance, "href", ""))

        normalized_label = label.strip().lower() if label else ""
        normalized_icon = icon_key.strip().lower() if icon_key else ""

        if normalized_label and normalized_icon and normalized_label != normalized_icon:
            raise serializers.ValidationError(
                {
                    "icon_key": [
                        f"The selected icon does not match {label}. Please choose {normalized_label}."
                    ]
                }
            )

        if normalized_icon == "email" and not href:
            attrs["href"] = f"mailto:{value}"

        if normalized_icon == "phone" and not href:
            clean_phone = re.sub(r"\s+", "", value)
            attrs["href"] = f"tel:{clean_phone}"

        if normalized_icon == "location" and href is None:
            attrs["href"] = ""

        return attrs

    def _validate_email(self, value):
        pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

        if not re.match(pattern, value):
            raise serializers.ValidationError(
                "Please enter a valid email address. Example: s.fujo@hotmail.com"
            )

    def _validate_phone(self, value):
        clean_value = value.strip()

        pattern = r"^\+?[0-9\s\-()]{7,20}$"

        if not re.match(pattern, clean_value):
            raise serializers.ValidationError(
                "Please enter a valid phone number. Example: +971 527 929 218"
            )

    def _validate_location(self, value):
        if len(value) < 3:
            raise serializers.ValidationError(
                "Location is too short. Example: Dubai, United Arab Emirates"
            )

        if len(value) > 255:
            raise serializers.ValidationError(
                "Location is too long. Please keep it under 255 characters."
            )


class FooterSectionAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for managing the footer section.

    Includes nested social links and contact items for admin preview.
    """

    social_links = FooterSocialLinkAdminSerializer(many=True, read_only=True)
    contact_items = FooterContactItemAdminSerializer(many=True, read_only=True)

    class Meta:
        model = FooterSection
        fields = [
            "id",
            "follow_title",
            "copyright_name",
            "social_links",
            "contact_items",
            "is_active",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "social_links",
            "contact_items",
            "created_at",
            "updated_at",
        ]

        extra_kwargs = {
            "follow_title": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the footer follow title.",
                    "required": "Footer follow title is required.",
                },
            },
            "copyright_name": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "blank": "Please enter the copyright name.",
                    "required": "Copyright name is required.",
                },
            },
        }

    def validate_follow_title(self, value):
        value = value.strip()

        if len(value) < 3:
            raise serializers.ValidationError(
                "Footer follow title is too short. Example: Follow me"
            )

        if len(value) > 120:
            raise serializers.ValidationError(
                "Footer follow title is too long. Please keep it under 120 characters."
            )

        return value

    def validate_copyright_name(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError(
                "Copyright name is too short. Example: Samah Fujo"
            )

        if len(value) > 160:
            raise serializers.ValidationError(
                "Copyright name is too long. Please keep it under 160 characters."
            )

        if re.search(r"https?://|www\.", value, flags=re.IGNORECASE):
            raise serializers.ValidationError(
                "Copyright name should be a plain name, not a URL."
            )

        return value
