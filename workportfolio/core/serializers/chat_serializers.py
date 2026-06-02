from rest_framework import serializers

from ..models import ChatSession, ChatMessage


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


class RequestEmailVerificationSerializer(serializers.Serializer):
    email = serializers.EmailField()


class VerifyEmailCodeSerializer(serializers.Serializer):
    email = serializers.EmailField()
    code = serializers.CharField(min_length=6, max_length=6)
