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
