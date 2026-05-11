from .models import (
    ProfileDocument,
    DocumentChunk,
    ChatSession,
    ChatMessage,
    ContactMessage,
    ProjectRequest,
)
from rest_framework import serializers
from django.conf import settings
from pathlib import Path

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
