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
)
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
    Supports uploaded image and external image URL.
    """

    image_url = serializers.SerializerMethodField()

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
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "image_url",
        ]

    def get_image_url(self, obj):
        request = self.context.get("request")

        if obj.image:
            url = obj.image.url
            return request.build_absolute_uri(url) if request else url

        if obj.external_image_url:
            return obj.external_image_url

        return None


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
