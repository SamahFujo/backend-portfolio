from .models import ProfileDocument, DocumentChunk, ChatSession, ChatMessage
from rest_framework import serializers

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



"""
Serializers for the public "Get in Touch" form API endpoint.
"""
class GetInTouchSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=120)
    email = serializers.EmailField(max_length=255)
    subject = serializers.CharField(max_length=200, required=False, allow_blank=True)
    message = serializers.CharField(max_length=5000)

    # Simple “honeypot” anti-bot field (frontend keeps it hidden)
    website = serializers.CharField(required=False, allow_blank=True, max_length=200)

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
    """
    Input serializer for the chat endpoint.
    """
    session_id = serializers.UUIDField(required=False)
    message = serializers.CharField()
