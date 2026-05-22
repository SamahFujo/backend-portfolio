from django.contrib import admin

# Register your models here.

from django.contrib import admin
from .models import (
    HeroSection, AboutSection, SkillItem,
    SkillSection, ProjectSection, ProjectItem, CertificateSection,
    CertificateItem,
    ResearchSection,
    ResearchItem,
)


@admin.register(HeroSection)
class HeroSectionAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing the homepage Hero section.
    """

    list_display = (
        "full_name",
        "headline",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "is_active",
    )

    search_fields = (
        "eyebrow_text",
        "full_name",
        "headline",
        "description",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        ("Hero Text Content", {
            "fields": (
                "eyebrow_text",
                "full_name",
                "headline",
                "description",
            )
        }),
        ("Buttons", {
            "fields": (
                "primary_button_text",
                "primary_button_url",
                "secondary_button_text",
                "secondary_button_url",
            )
        }),
        ("Theme Images", {
            "fields": (
                "hero_image_dark",
                "hero_image_light",
                "background_image",
            )
        }),
        ("Status", {
            "fields": (
                "is_active",
            )
        }),
        ("Timestamps", {
            "fields": (
                "created_at",
                "updated_at",
            )
        }),
    )


@admin.register(AboutSection)
class AboutSectionAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing the About Me section.
    """

    list_display = (
        "section_title",
        "terminal_label",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "is_active",
    )

    search_fields = (
        "section_title",
        "terminal_label",
        "welcome_title",
        "description",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        ("About Section Content", {
            "fields": (
                "section_title",
                "terminal_label",
                "welcome_title",
                "description",
            )
        }),
        ("Status", {
            "fields": (
                "is_active",
            )
        }),
        ("Timestamps", {
            "fields": (
                "created_at",
                "updated_at",
            )
        }),
    )


class SkillItemInline(admin.TabularInline):
    """
    Allows managing skill cards directly inside the Skill Section admin page.
    """

    model = SkillItem
    extra = 1

    fields = (
        "sort_order",
        "category",
        "icon",
        "label",
        "level",
        "is_active",
    )

    ordering = (
        "category",
        "sort_order",
    )


@admin.register(SkillSection)
class SkillSectionAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing the Skills section.
    """

    list_display = (
        "title_line_1",
        "title_line_2",
        "badge_text",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "is_active",
    )

    search_fields = (
        "badge_text",
        "title_line_1",
        "title_line_2",
        "description",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        ("Skills Section Header", {
            "fields": (
                "badge_text",
                "title_line_1",
                "title_line_2",
                "description",
            )
        }),
        ("Status", {
            "fields": (
                "is_active",
            )
        }),
        ("Timestamps", {
            "fields": (
                "created_at",
                "updated_at",
            )
        }),
    )

    inlines = [SkillItemInline]


@admin.register(SkillItem)
class SkillItemAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing individual skill cards
    used in the interactive Skills section.
    """

    list_display = (
        "label",
        "section",
        "category",
        "icon",
        "level",
        "sort_order",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "is_active",
        "section",
        "category",
    )

    search_fields = (
        "label",
        "icon",
        "category",
        "summary_heading",
        "summary_text",
    )

    list_editable = (
        "level",
        "sort_order",
        "is_active",
    )

    ordering = (
        "category",
        "sort_order",
        "created_at",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        ("Skill Basic Info", {
            "fields": (
                "section",
                "category",
                "label",
                "icon",
                "level",
                "sort_order",
                "is_active",
            )
        }),
        ("Skill Summary", {
            "fields": (
                "summary_heading",
                "summary_text",
                "summary_points",
            )
        }),
        ("Timestamps", {
            "fields": (
                "created_at",
                "updated_at",
            )
        }),
    )


class ProjectItemInline(admin.TabularInline):
    """
    Allows managing project items directly inside the Project Section admin page.
    """

    model = ProjectItem
    extra = 1

    fields = (
        "sort_order",
        "title",
        "category",
        "is_featured",
        "is_active",
    )

    ordering = (
        "sort_order",
        "created_at",
    )


@admin.register(ProjectSection)
class ProjectSectionAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing the Projects section header.
    """

    list_display = (
        "title",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "is_active",
    )

    search_fields = (
        "title",
        "description",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        ("Projects Section Header", {
            "fields": (
                "title",
                "description",
            )
        }),
        ("Status", {
            "fields": (
                "is_active",
            )
        }),
        ("Timestamps", {
            "fields": (
                "created_at",
                "updated_at",
            )
        }),
    )

    inlines = [ProjectItemInline]


@admin.register(ProjectItem)
class ProjectItemAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing individual portfolio projects.
    """

    list_display = (
        "title",
        "category",
        "sort_order",
        "is_featured",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "category",
        "is_featured",
        "is_active",
        "section",
    )

    search_fields = (
        "title",
        "short_description",
        "description",
        "category",
    )

    list_editable = (
        "sort_order",
        "is_featured",
        "is_active",
    )

    prepopulated_fields = {
        "slug": ("title",),
    }

    readonly_fields = (
        "created_at",
        "updated_at",
    )


fieldsets = (
    ("Project Basic Info", {
        "fields": (
            "section",
            "title",
            "slug",
            "category",
            "sort_order",
            "is_featured",
            "is_active",
        )
    }),
    ("Project Content", {
        "fields": (
            "short_description",
            "description",
            "tech_stack",
        )
    }),
    ("Project Images", {
        "fields": (
            "thumbnail_image",
            "hero_image",
            "alt_text",
        )
    }),
    ("Timestamps", {
        "fields": (
            "created_at",
            "updated_at",
        )
    }),
)


class CertificateItemInline(admin.TabularInline):
    """
    Allows simple certificate item management inside the Certificate Section admin page.

    Keep this inline lightweight because certificate items include images,
    PDF files, and skills JSON. Full editing should be done from CertificateItemAdmin.
    """

    model = CertificateItem
    extra = 1

    fields = (
        "sort_order",
        "title",
        "mobile_title",
        "issuer",
        "issue_date",
        "is_active",
    )

    ordering = (
        "sort_order",
        "created_at",
    )


@admin.register(CertificateSection)
class CertificateSectionAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing the Certificates section header.
    """

    list_display = (
        "title",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "is_active",
    )

    search_fields = (
        "title",
        "description",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        (
            "Certificates Section Header",
            {
                "fields": (
                    "title",
                    "description",
                )
            },
        ),
        (
            "Status",
            {
                "fields": (
                    "is_active",
                )
            },
        ),
        (
            "Timestamps",
            {
                "fields": (
                    "created_at",
                    "updated_at",
                )
            },
        ),
    )

    inlines = [CertificateItemInline]


@admin.register(CertificateItem)
class CertificateItemAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing individual certificates.
    """

    list_display = (
        "title",
        "mobile_title",
        "issuer",
        "issue_date",
        "sort_order",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "issuer",
        "is_active",
        "section",
    )

    search_fields = (
        "title",
        "mobile_title",
        "issuer",
        "issue_date",
        "skills",
    )

    list_editable = (
        "sort_order",
        "is_active",
    )

    prepopulated_fields = {
        "slug": ("title",),
    }

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        (
            "Certificate Basic Info",
            {
                "fields": (
                    "section",
                    "title",
                    "mobile_title",
                    "slug",
                    "issuer",
                    "issue_date",
                    "sort_order",
                    "is_active",
                )
            },
        ),
        (
            "Certificate Files",
            {
                "fields": (
                    "certificate_image",
                    "certificate_file",
                    "alt_text",
                )
            },
        ),
        (
            "Skills Obtained",
            {
                "fields": (
                    "skills",
                )
            },
        ),
        (
            "Timestamps",
            {
                "fields": (
                    "created_at",
                    "updated_at",
                )
            },
        ),
    )


class ResearchItemInline(admin.TabularInline):
    """
    Lightweight inline editor for research items inside the Research Section page.
    Full research item editing should be done from ResearchItemAdmin.
    """

    model = ResearchItem
    extra = 1

    fields = (
        "sort_order",
        "title",
        "research_type",
        "publish_date",
        "reads",
        "citations",
        "is_active",
    )

    ordering = (
        "sort_order",
        "created_at",
    )


@admin.register(ResearchSection)
class ResearchSectionAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing the Research section header.
    """

    list_display = (
        "title",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "is_active",
    )

    search_fields = (
        "title",
        "description",
    )

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        (
            "Research Section Header",
            {
                "fields": (
                    "title",
                    "description",
                )
            },
        ),
        (
            "Status",
            {
                "fields": (
                    "is_active",
                )
            },
        ),
        (
            "Timestamps",
            {
                "fields": (
                    "created_at",
                    "updated_at",
                )
            },
        ),
    )

    inlines = [ResearchItemInline]


@admin.register(ResearchItem)
class ResearchItemAdmin(admin.ModelAdmin):
    """
    Admin configuration for managing individual research cards.
    """

    list_display = (
        "title",
        "research_type",
        "publish_date",
        "reads",
        "citations",
        "sort_order",
        "is_active",
        "updated_at",
    )

    list_filter = (
        "research_type",
        "is_active",
        "section",
    )

    search_fields = (
        "title",
        "research_type",
        "publish_date",
        "reads",
        "citations",
        "authors",
    )

    list_editable = (
        "sort_order",
        "is_active",
    )

    prepopulated_fields = {
        "slug": ("title",),
    }

    readonly_fields = (
        "created_at",
        "updated_at",
    )

    fieldsets = (
        (
            "Research Basic Info",
            {
                "fields": (
                    "section",
                    "title",
                    "slug",
                    "research_type",
                    "publish_date",
                    "reads",
                    "citations",
                    "sort_order",
                    "is_active",
                )
            },
        ),
        (
            "Authors",
            {
                "fields": (
                    "authors",
                ),
                "description": (
                    "Enter authors as valid JSON, for example: "
                    '["Moaiad Khder", "Samah Fujo"]'
                ),
            },
        ),
        (
            "Actions and Links",
            {
                "fields": (
                    "primary_action",
                    "primary_action_href",
                    "share_href",
                )
            },
        ),
        (
            "Research Image",
            {
                "fields": (
                    "image",
                    "external_image_url",
                    "alt_text",
                ),
                "description": (
                    "Use either uploaded image or external image URL. "
                    "Uploaded image will be preferred later in the serializer."
                ),
            },
        ),
        (
            "Timestamps",
            {
                "fields": (
                    "created_at",
                    "updated_at",
                )
            },
        ),
    )
