import json
import re
from urllib.parse import urlparse

from rest_framework import serializers

from ..models import (
    HeroSection,
    AboutSection,
    SkillSection,
    SkillItem,
    ProjectSection,
    ProjectItem,
    CertificateSection,
    CertificateItem,
    ResearchSection,
    ResearchItem,
    FooterSection,
    FooterSocialLink,
    FooterContactItem,
)


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


class SkillItemAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating skill items.

    Validation covered:
    - Required business fields
    - Controlled category list
    - Controlled icon list
    - Label, summary heading, and summary text length
    - Level range from 1 to 10
    - Summary points list validation
    - Sort order range
    """

    ALLOWED_CATEGORIES = {
        "Frontend",
        "Backend",
        "AI / LLM",
        "Database",
        "DevOps",
        "Languages",
        "UI",
    }

    ALLOWED_ICONS = {
        "react",
        "nextjs",
        "tailwind",
        "javascript",
        "typescript",
        "drf",
        "fastapi",
        "flask",
        "rest",
        "swagger",
        "openapi",
        "rbac",
        "jwt",
        "nodejs",
        "transformers",
        "huggingface",
        "llms",
        "rag",
        "promptengineering",
        "gemini",
        "pdfparsing",
        "ocr",
        "ollama",
        "openwebui",
        "langchain",
        "langfuse",
        "postgresql",
        "mongodb",
        "mysql",
        "sqlserver",
        "oracle",
        "docker",
        "nginx",
        "gunicorn",
        "postman",
        "python",
        "dotnet",
        "java",
        "cpp",
        "php",
        "streamlit",
        "figma",
    }

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

        extra_kwargs = {
            "section": {
                "required": False,
                "allow_null": True,
            },
            "category": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Skill category is required.",
                    "blank": "Skill category is required.",
                },
            },
            "icon": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Skill icon is required.",
                    "blank": "Skill icon is required.",
                },
            },
            "label": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Skill label is required.",
                    "blank": "Skill label is required.",
                },
            },
            "summary_heading": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Summary heading is required.",
                    "blank": "Summary heading is required.",
                },
            },
            "summary_text": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Summary text is required.",
                    "blank": "Summary text is required.",
                },
            },
        }

    def validate_category(self, value):
        value = value.strip()

        if value not in self.ALLOWED_CATEGORIES:
            raise serializers.ValidationError(
                "Please choose a valid skill category."
            )

        return value

    def validate_icon(self, value):
        value = value.strip()

        if value not in self.ALLOWED_ICONS:
            raise serializers.ValidationError(
                "Please choose a valid icon from the supported icon list."
            )

        return value

    def validate_label(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError("Skill label is too short.")

        if len(value) > 60:
            raise serializers.ValidationError(
                "Skill label is too long. Please keep it under 60 characters."
            )

        return value

    def validate_level(self, value):
        if value is None:
            raise serializers.ValidationError("Skill level is required.")

        if value < 1 or value > 10:
            raise serializers.ValidationError(
                "Skill level must be between 1 and 10."
            )

        return value

    def validate_summary_heading(self, value):
        value = value.strip()

        if len(value) < 5:
            raise serializers.ValidationError(
                "Summary heading is too short. Please write at least 5 characters."
            )

        if len(value) > 120:
            raise serializers.ValidationError(
                "Summary heading is too long. Please keep it under 120 characters."
            )

        return value

    def validate_summary_text(self, value):
        value = value.strip()

        if len(value) < 20:
            raise serializers.ValidationError(
                "Summary text is too short. Please write at least 20 characters."
            )

        if len(value) > 500:
            raise serializers.ValidationError(
                "Summary text is too long. Please keep it under 500 characters."
            )

        return value

    def validate_summary_points(self, value):
        """
        Supports:
        - JSON list from application/json
        - JSON string from multipart/form-data if needed later
        """

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise serializers.ValidationError(
                    "Summary points must be a valid list. Add one point per line."
                )

        if not isinstance(value, list):
            raise serializers.ValidationError(
                "Summary points must be a list of text points."
            )

        cleaned_points = []

        for point in value:
            if not isinstance(point, str):
                raise serializers.ValidationError(
                    "Each summary point must be written as text."
                )

            clean_point = point.strip()

            if clean_point:
                cleaned_points.append(clean_point)

        if not cleaned_points:
            raise serializers.ValidationError(
                "Please add at least one summary point."
            )

        if len(cleaned_points) > 8:
            raise serializers.ValidationError(
                "Too many summary points. Please keep the list under 8 points."
            )

        for point in cleaned_points:
            if len(point) > 160:
                raise serializers.ValidationError(
                    "Each summary point must be under 160 characters."
                )

        unique_points = []
        seen = set()

        for point in cleaned_points:
            key = point.lower()

            if key not in seen:
                seen.add(key)
                unique_points.append(point)

        return unique_points

    def validate_sort_order(self, value):
        if value is None:
            return 0

        if value < 0:
            raise serializers.ValidationError("Sort order cannot be negative.")

        if value > 999:
            raise serializers.ValidationError(
                "Sort order is too large. Please use a value between 0 and 999."
            )

        return value


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

        extra_kwargs = {
            "badge_text": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Badge text is required.",
                    "blank": "Badge text is required.",
                },
            },
            "title_line_1": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Title line 1 is required.",
                    "blank": "Title line 1 is required.",
                },
            },
            "title_line_2": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Title line 2 is required.",
                    "blank": "Title line 2 is required.",
                },
            },
            "description": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Skills section description is required.",
                    "blank": "Skills section description is required.",
                },
            },
        }

    def validate_badge_text(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError("Badge text is too short.")

        if len(value) > 80:
            raise serializers.ValidationError(
                "Badge text is too long. Please keep it under 80 characters."
            )

        return value

    def validate_title_line_1(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError("Title line 1 is too short.")

        if len(value) > 80:
            raise serializers.ValidationError(
                "Title line 1 is too long. Please keep it under 80 characters."
            )

        return value

    def validate_title_line_2(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError("Title line 2 is too short.")

        if len(value) > 80:
            raise serializers.ValidationError(
                "Title line 2 is too long. Please keep it under 80 characters."
            )

        return value

    def validate_description(self, value):
        value = value.strip()

        if len(value) < 20:
            raise serializers.ValidationError(
                "Section description is too short. Please write at least 20 characters."
            )

        if len(value) > 500:
            raise serializers.ValidationError(
                "Section description is too long. Please keep it under 500 characters."
            )

        return value


class ProjectItemAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating project items.
    Provides professional validation for admin dashboard input.
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

        extra_kwargs = {
            "section": {
                "required": False,
                "allow_null": True,
            },
            "slug": {
                "required": False,
                "allow_blank": True,
            },
            "title": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Project title is required.",
                    "blank": "Project title is required.",
                },
            },
            "category": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Project category is required.",
                    "blank": "Project category is required.",
                },
            },
            "short_description": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Short description is required.",
                    "blank": "Short description is required.",
                },
            },
            "description": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Full project description is required.",
                    "blank": "Full project description is required.",
                },
            },
            "alt_text": {
                "required": False,
                "allow_blank": True,
            },
            "thumbnail_image": {
                "required": False,
                "allow_null": True,
            },
            "hero_image": {
                "required": False,
                "allow_null": True,
            },
        }

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

    def validate_slug(self, value):
        if not value:
            return value

        clean_value = value.strip().lower()

        if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", clean_value):
            raise serializers.ValidationError(
                "Slug can only contain lowercase letters, numbers, and hyphens. Example: ai-property-chatbot"
            )

        existing_qs = ProjectItem.objects.filter(slug=clean_value)

        if self.instance:
            existing_qs = existing_qs.exclude(pk=self.instance.pk)

        if existing_qs.exists():
            raise serializers.ValidationError(
                "This slug is already used by another project. Please choose a different slug."
            )

        return clean_value

    def validate_title(self, value):
        value = value.strip()

        if len(value) < 5:
            raise serializers.ValidationError(
                "Project title is too short. Please enter a clear project name."
            )

        if len(value) > 180:
            raise serializers.ValidationError(
                "Project title is too long. Please keep it under 180 characters."
            )

        return value

    def validate_category(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError("Project category is too short.")

        if len(value) > 80:
            raise serializers.ValidationError(
                "Project category is too long. Please keep it under 80 characters."
            )

        return value

    def validate_short_description(self, value):
        value = value.strip()

        if len(value) < 20:
            raise serializers.ValidationError(
                "Short description is too short. Please write at least 20 characters."
            )

        if len(value) > 280:
            raise serializers.ValidationError(
                "Short description is too long. Please keep it under 280 characters."
            )

        return value

    def validate_description(self, value):
        value = value.strip()

        if len(value) < 50:
            raise serializers.ValidationError(
                "Full description is too short. Please write at least 50 characters."
            )

        if len(value) > 2000:
            raise serializers.ValidationError(
                "Full description is too long. Please keep it under 2000 characters."
            )

        return value

    def validate_alt_text(self, value):
        value = (value or "").strip()

        if value and len(value) > 180:
            raise serializers.ValidationError(
                "Alt text is too long. Please keep it under 180 characters."
            )

        return value

    def validate_sort_order(self, value):
        if value is None:
            return 0

        if value < 0:
            raise serializers.ValidationError(
                "Sort order cannot be negative."
            )

        if value > 999:
            raise serializers.ValidationError(
                "Sort order is too large. Please use a value between 0 and 999."
            )

        return value

    def validate_tech_stack(self, value):
        """
        Accepts:
        - JSON list from application/json
        - JSON string from multipart/form-data
        """

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise serializers.ValidationError(
                    "Tech stack must be a valid list. Add one technology per line."
                )

        if not isinstance(value, list):
            raise serializers.ValidationError(
                "Tech stack must be a list of technologies."
            )

        cleaned_items = []

        for item in value:
            if not isinstance(item, str):
                raise serializers.ValidationError(
                    "Each technology must be written as text."
                )

            clean_item = item.strip()

            if clean_item:
                cleaned_items.append(clean_item)

        if not cleaned_items:
            raise serializers.ValidationError(
                "Please add at least one technology."
            )

        if len(cleaned_items) > 20:
            raise serializers.ValidationError(
                "Too many technologies. Please keep the tech stack under 20 items."
            )

        unique_items = []
        seen = set()

        for item in cleaned_items:
            key = item.lower()

            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return unique_items

    def validate_thumbnail_image(self, value):
        return self._validate_image(
            value=value,
            field_name="Thumbnail image",
            max_size_mb=5,
        )

    def validate_hero_image(self, value):
        return self._validate_image(
            value=value,
            field_name="Hero image",
            max_size_mb=8,
        )

    def validate(self, attrs):
        thumbnail_image = attrs.get(
            "thumbnail_image",
            getattr(self.instance, "thumbnail_image", None),
        )

        hero_image = attrs.get(
            "hero_image",
            getattr(self.instance, "hero_image", None),
        )

        title = attrs.get("title") or getattr(self.instance, "title", "")

        errors = {}

        if not thumbnail_image:
            errors["thumbnail_image"] = [
                "Please upload a thumbnail image for the project card."
            ]

        if not hero_image:
            errors["hero_image"] = [
                "Please upload a hero image for the project preview."
            ]

        if errors:
            raise serializers.ValidationError(errors)

        if not attrs.get("alt_text"):
            attrs["alt_text"] = f"{title} project image"

        return attrs

    def _validate_image(self, value, field_name, max_size_mb):
        if not value:
            return value

        max_size_bytes = max_size_mb * 1024 * 1024

        if value.size > max_size_bytes:
            raise serializers.ValidationError(
                f"{field_name} is too large. Please upload an image smaller than {max_size_mb}MB."
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
                f"Unsupported {field_name.lower()} type. Please upload JPG, PNG, WEBP, or GIF."
            )

        return value


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

        extra_kwargs = {
            "title": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Projects section title is required.",
                    "blank": "Projects section title is required.",
                },
            },
            "description": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Projects section description is required.",
                    "blank": "Projects section description is required.",
                },
            },
        }

    def validate_title(self, value):
        value = value.strip()

        if len(value) < 3:
            raise serializers.ValidationError(
                "Section title is too short. Example: Projects"
            )

        if len(value) > 120:
            raise serializers.ValidationError(
                "Section title is too long. Please keep it under 120 characters."
            )

        return value

    def validate_description(self, value):
        value = value.strip()

        if len(value) < 20:
            raise serializers.ValidationError(
                "Section description is too short. Please write at least 20 characters."
            )

        if len(value) > 500:
            raise serializers.ValidationError(
                "Section description is too long. Please keep it under 500 characters."
            )

        return value


class CertificateItemAdminSerializer(serializers.ModelSerializer):
    """
    Admin serializer for creating and updating certificate items.

    Industrial validation covered:
    - Required business fields
    - Slug format and uniqueness
    - Title, issuer, issue date, and mobile title length
    - Skills list validation
    - Sort order validation
    - Certificate image validation
    - Certificate PDF validation
    - Required files on create
    - Existing files accepted on update
    - Safe default alt text
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

        extra_kwargs = {
            "section": {
                "required": False,
                "allow_null": True,
            },
            "slug": {
                "required": False,
                "allow_blank": True,
            },
            "title": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Certificate title is required.",
                    "blank": "Certificate title is required.",
                },
            },
            "mobile_title": {
                "required": False,
                "allow_blank": True,
            },
            "issuer": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Certificate issuer is required.",
                    "blank": "Certificate issuer is required.",
                },
            },
            "issue_date": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Issue date is required.",
                    "blank": "Issue date is required.",
                },
            },
            "alt_text": {
                "required": False,
                "allow_blank": True,
            },
            "certificate_image": {
                "required": False,
                "allow_null": True,
            },
            "certificate_file": {
                "required": False,
                "allow_null": True,
            },
        }

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

    def validate_slug(self, value):
        if not value:
            return value

        clean_value = value.strip().lower()

        if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", clean_value):
            raise serializers.ValidationError(
                "Slug can only contain lowercase letters, numbers, and hyphens. Example: master-of-chatgpt"
            )

        existing_qs = CertificateItem.objects.filter(slug=clean_value)

        if self.instance:
            existing_qs = existing_qs.exclude(pk=self.instance.pk)

        if existing_qs.exists():
            raise serializers.ValidationError(
                "This slug is already used by another certificate. Please choose a different slug."
            )

        return clean_value

    def validate_title(self, value):
        value = value.strip()

        if len(value) < 3:
            raise serializers.ValidationError(
                "Certificate title is too short. Please enter a clear certificate name."
            )

        if len(value) > 180:
            raise serializers.ValidationError(
                "Certificate title is too long. Please keep it under 180 characters."
            )

        return value

    def validate_mobile_title(self, value):
        value = (value or "").strip()

        if value and len(value) > 60:
            raise serializers.ValidationError(
                "Mobile title is too long. Please keep it under 60 characters."
            )

        return value

    def validate_issuer(self, value):
        value = value.strip()

        if len(value) < 2:
            raise serializers.ValidationError("Issuer name is too short.")

        if len(value) > 120:
            raise serializers.ValidationError(
                "Issuer name is too long. Please keep it under 120 characters."
            )

        return value

    def validate_issue_date(self, value):
        value = value.strip()

        if len(value) > 80:
            raise serializers.ValidationError(
                "Issue date is too long. Please keep it under 80 characters."
            )

        return value

    def validate_alt_text(self, value):
        value = (value or "").strip()

        if value and len(value) > 180:
            raise serializers.ValidationError(
                "Alt text is too long. Please keep it under 180 characters."
            )

        return value

    def validate_sort_order(self, value):
        if value is None:
            return 0

        if value < 0:
            raise serializers.ValidationError("Sort order cannot be negative.")

        if value > 999:
            raise serializers.ValidationError(
                "Sort order is too large. Please use a value between 0 and 999."
            )

        return value

    def validate_skills(self, value):
        """
        Supports:
        - JSON list from application/json
        - JSON string from multipart/form-data
        """

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise serializers.ValidationError(
                    "Skills must be a valid list. Add one skill per line."
                )

        if not isinstance(value, list):
            raise serializers.ValidationError(
                "Skills must be a list of skill names."
            )

        cleaned_skills = []

        for skill in value:
            if not isinstance(skill, str):
                raise serializers.ValidationError(
                    "Each skill must be written as text."
                )

            clean_skill = skill.strip()

            if clean_skill:
                cleaned_skills.append(clean_skill)

        if not cleaned_skills:
            raise serializers.ValidationError(
                "Please add at least one skill."
            )

        if len(cleaned_skills) > 20:
            raise serializers.ValidationError(
                "Too many skills. Please keep the skills list under 20 items."
            )

        for skill in cleaned_skills:
            if len(skill) > 60:
                raise serializers.ValidationError(
                    f"'{skill}' is too long. Each skill should be under 60 characters."
                )

        unique_skills = []
        seen = set()

        for skill in cleaned_skills:
            key = skill.lower()

            if key not in seen:
                seen.add(key)
                unique_skills.append(skill)

        return unique_skills

    def validate_certificate_image(self, value):
        return self._validate_certificate_image(value)

    def validate_certificate_file(self, value):
        return self._validate_certificate_pdf(value)

    def validate(self, attrs):
        certificate_image = attrs.get(
            "certificate_image",
            getattr(self.instance, "certificate_image", None),
        )

        certificate_file = attrs.get(
            "certificate_file",
            getattr(self.instance, "certificate_file", None),
        )

        title = attrs.get("title") or getattr(self.instance, "title", "")

        errors = {}

        if not certificate_image:
            errors["certificate_image"] = [
                "Please upload a certificate image."
            ]

        if not certificate_file:
            errors["certificate_file"] = [
                "Please upload the certificate PDF file."
            ]

        if errors:
            raise serializers.ValidationError(errors)

        if not attrs.get("alt_text"):
            attrs["alt_text"] = f"{title} certificate image"

        return attrs

    def _validate_certificate_image(self, value):
        if not value:
            return value

        max_size_mb = 5
        max_size_bytes = max_size_mb * 1024 * 1024

        if value.size > max_size_bytes:
            raise serializers.ValidationError(
                f"Certificate image is too large. Please upload an image smaller than {max_size_mb}MB."
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
                "Unsupported certificate image type. Please upload JPG, PNG, WEBP, or GIF."
            )

        return value

    def _validate_certificate_pdf(self, value):
        if not value:
            return value

        max_size_mb = 10
        max_size_bytes = max_size_mb * 1024 * 1024

        if value.size > max_size_bytes:
            raise serializers.ValidationError(
                f"Certificate PDF is too large. Please upload a PDF smaller than {max_size_mb}MB."
            )

        content_type = getattr(value, "content_type", "")

        if content_type != "application/pdf":
            raise serializers.ValidationError(
                "Unsupported certificate file type. Please upload a PDF file only."
            )

        return value


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

        extra_kwargs = {
            "title": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Certificates section title is required.",
                    "blank": "Certificates section title is required.",
                },
            },
            "description": {
                "required": True,
                "allow_blank": False,
                "error_messages": {
                    "required": "Certificates section description is required.",
                    "blank": "Certificates section description is required.",
                },
            },
        }

    def validate_title(self, value):
        value = value.strip()

        if len(value) < 3:
            raise serializers.ValidationError(
                "Section title is too short. Example: Certificates"
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
                "Section description is too short. Please write at least 10 characters."
            )

        if len(value) > 500:
            raise serializers.ValidationError(
                "Section description is too long. Please keep it under 500 characters."
            )

        return value


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
