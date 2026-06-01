from django.core.management.base import BaseCommand

from core.models import FooterSection, FooterSocialLink, FooterContactItem


class Command(BaseCommand):
    help = "Seed footer section, social links, and contact items."

    def handle(self, *args, **options):
        FooterSection.objects.update(is_active=False)

        footer = FooterSection.objects.create(
            follow_title="Follow me",
            copyright_name="Samah Fujo",
            is_active=True,
        )

        social_links = [
            {
                "name": "LinkedIn",
                "icon_key": "linkedin",
                "url": "https://www.linkedin.com/in/samah-fujo-885a3b207/",
                "sort_order": 1,
            },
            {
                "name": "Instagram",
                "icon_key": "instagram",
                "url": "https://www.instagram.com/samah.ai.engineer?igsh=cGVjYWtrcWpyZjlp&utm_source=qr",
                "sort_order": 2,
            },
            {
                "name": "TikTok",
                "icon_key": "tiktok",
                "url": "https://www.tiktok.com/@samah.ai.engineer?is_from_webapp=1&sender_device=pc",
                "sort_order": 3,
            },
        ]

        for social in social_links:
            FooterSocialLink.objects.create(
                section=footer,
                is_active=True,
                **social,
            )

        contact_items = [
            {
                "label": "Email",
                "value": "s.fujo@hotmail.com",
                "href": "mailto:s.fujo@hotmail.com",
                "icon_key": "email",
                "sort_order": 1,
            },
            {
                "label": "Phone",
                "value": "+971 527 929 218",
                "href": "tel:+971527929218",
                "icon_key": "phone",
                "sort_order": 2,
            },
            {
                "label": "Location",
                "value": "Dubai, United Arab Emirates",
                "href": "",
                "icon_key": "location",
                "sort_order": 3,
            },
        ]

        for contact in contact_items:
            FooterContactItem.objects.create(
                section=footer,
                is_active=True,
                **contact,
            )

        self.stdout.write(
            self.style.SUCCESS("Footer section seeded successfully.")
        )
