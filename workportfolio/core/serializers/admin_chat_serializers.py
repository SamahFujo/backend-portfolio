from rest_framework import serializers

from ..models import (
    ChatSession,
    ChatMessage,
    ContactMessage,
    ProjectRequest,
)


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
