from django.contrib import admin

# Register your models here.

from django.contrib import admin
from .models import HeroSection


@admin.register(HeroSection)
class HeroSectionAdmin(admin.ModelAdmin):
    list_display = (
        "main_title",
        "subtitle",
        "is_active",
        "updated_at",
    )

    list_filter = ("is_active", "created_at", "updated_at")

    search_fields = (
        "main_title",
        "subtitle",
        "description",
    )

    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        (
            "Hero Text",
            {
                "fields": (
                    "title_prefix",
                    "main_title",
                    "subtitle",
                    "description",
                )
            },
        ),
        (
            "Buttons",
            {
                "fields": (
                    "primary_button_text",
                    "primary_button_url",
                    "secondary_button_text",
                    "secondary_button_url",
                )
            },
        ),
        (
            "Media",
            {
                "fields": (
                    "hero_image",
                    "background_image",
                )
            },
        ),
        (
            "Status",
            {
                "fields": (
                    "is_active",
                    "created_at",
                    "updated_at",
                )
            },
        ),
    )
