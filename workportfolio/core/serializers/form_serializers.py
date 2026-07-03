from rest_framework import serializers
import json 

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


class GetInTouchSerializer(serializers.Serializer):
    name = serializers.CharField(min_length=2, max_length=120)
    email = serializers.EmailField()
    subject = serializers.CharField(
        max_length=200,
        required=False,
        allow_blank=True,
    )
    message = serializers.CharField(min_length=10, max_length=5000)

    # Simple honeypot anti-bot field
    website = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=200,
    )

    def validate(self, attrs):
        # Honeypot: if filled, likely bot
        if attrs.get("website"):
            raise serializers.ValidationError("Spam detected.")
        return attrs


class WebsiteVisitTrackSerializer(serializers.Serializer):
    visitor_id = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=100,
    )
    session_key = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=100,
    )

    path = serializers.CharField(max_length=255)

    page_title = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=255,
    )

    event_type = serializers.ChoiceField(
        choices=[
            "page_view",
            "cta_click",
            "custom",
            "engagement",
            "frontend_error",
            "chatbot_error",
            "form_error",
        ],
        required=False,
        default="page_view",
    )

    event_name = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=100,
    )

    referrer = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=2000,
    )

    source_label = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=100,
    )

    utm_source = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=150,
    )
    utm_medium = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=150,
    )
    utm_campaign = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=150,
    )
    utm_term = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=150,
    )
    utm_content = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=150,
    )

    metadata = serializers.JSONField(required=False)

    def validate_path(self, value):
        value = (value or "").strip()
        if not value:
            raise serializers.ValidationError("Path is required.")

        if not value.startswith("/"):
            raise serializers.ValidationError("Path must start with '/'.")

        return value[:255]

    def validate_metadata(self, value):
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise serializers.ValidationError("Metadata must be an object.")

        # Prevent very large analytics payloads.
        encoded = json.dumps(value, default=str)
        if len(encoded) > 5000:
            raise serializers.ValidationError("Metadata is too large.")

        return value
