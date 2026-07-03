import uuid

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0025_profiledocument_is_reviewed_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="WebsiteVisit",
            fields=[
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("visitor_id", models.CharField(blank=True, db_index=True, help_text="Anonymous visitor identifier shared by the frontend.", max_length=100, null=True)),
                ("session_key", models.CharField(blank=True, db_index=True, help_text="Frontend session key if available.", max_length=100, null=True)),
                ("path", models.CharField(db_index=True, help_text="Visited path such as /, /projects, or /contact.", max_length=255)),
                ("page_title", models.CharField(blank=True, default="", help_text="Optional frontend page title for display in analytics.", max_length=255)),
                ("event_type", models.CharField(choices=[("page_view", "Page View"), ("cta_click", "CTA Click"), ("custom", "Custom")], db_index=True, default="page_view", max_length=20)),
                ("referrer", models.TextField(blank=True, help_text="Referring page or source URL.", null=True)),
                ("ip_address", models.GenericIPAddressField(blank=True, help_text="Visitor IP address for analytics/security tracking.", null=True)),
                ("user_agent", models.TextField(blank=True, help_text="Browser/device information for analytics.", null=True)),
                ("source_label", models.CharField(blank=True, default="", help_text="Optional frontend-defined source label.", max_length=100)),
                ("metadata", models.JSONField(blank=True, default=dict, help_text="Optional extra frontend analytics metadata.")),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
