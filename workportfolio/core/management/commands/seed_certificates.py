import os
from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import CertificateSection, CertificateItem


CERTIFICATES = [
    {
        "slug": "react-django-fullstack",
        "title": "React & Django Full Stack: web app, backend API, mobile apps",
        "mobile_title": "React & Django",
        "issuer": "Udemy",
        "issue_date": "Jan 25, 2025",
        "image_path": "assets/certificates/react-django.jpg",
        "file_path": "assets/certificates/Samah Cirtificate- React & Django Full Stack Web App, Backend API, Mobile Apps,.pdf",
        "alt_text": "React & Django Full Stack certificate",
        "skills": [
            "React",
            "Django",
            "REST API",
            "Full-Stack Development",
            "Web Apps",
            "Mobile Apps",
        ],
        "sort_order": 1,
    },
    {
        "slug": "building-ai-agents",
        "title": "Fundamentals of Building AI Agents",
        "mobile_title": "AI Agents",
        "issuer": "IBM via Coursera",
        "issue_date": "Nov 16, 2025",
        "image_path": "assets/certificates/ai-agents.jpg",
        "file_path": "assets/certificates/Fundamentals of Building AI Agents - Coursera OJBRRU0XILKH.pdf",
        "alt_text": "Fundamentals of Building AI Agents certificate",
        "skills": [
            "AI Agents",
            "Agentic AI",
            "LLMs",
            "AI Workflows",
            "IBM",
            "Coursera",
        ],
        "sort_order": 2,
    },
    {
        "slug": "master-of-chatgpt",
        "title": "Master of ChatGPT",
        "mobile_title": "ChatGPT",
        "issuer": "Coursiv",
        "issue_date": "5 February 2026",
        "image_path": "assets/certificates/chatgpt.jpg",
        "file_path": "assets/certificates/Master of chat gpt.pdf",
        "alt_text": "Master of ChatGPT certificate",
        "skills": [
            "ChatGPT",
            "Prompt Engineering",
            "AI Productivity",
            "LLM Workflows",
            "Generative AI",
        ],
        "sort_order": 3,
    },
    {
        "slug": "master-of-claude",
        "title": "Master of Claude",
        "mobile_title": "Claude",
        "issuer": "Coursiv",
        "issue_date": "16 February 2026",
        "image_path": "assets/certificates/claude.jpg",
        "file_path": "assets/certificates/Master of Claude.pdf",
        "alt_text": "Master of Claude certificate",
        "skills": [
            "Claude",
            "Prompt Engineering",
            "AI Productivity",
            "LLM Workflows",
            "AI Assistance",
        ],
        "sort_order": 4,
    },
    {
        "slug": "master-of-lovable",
        "title": "Master of Lovable",
        "mobile_title": "Lovable",
        "issuer": "Coursiv",
        "issue_date": "2 March 2026",
        "image_path": "assets/certificates/lovable.jpg",
        "file_path": "assets/certificates/Master of Lovabe.pdf",
        "alt_text": "Master of Lovable certificate",
        "skills": [
            "Lovable",
            "AI Tools",
            "Prompting",
            "Productivity",
            "Digital Skills",
        ],
        "sort_order": 5,
    },
]


class Command(BaseCommand):
    help = "Seed certificate section and existing certificate items from static website data."

    @transaction.atomic
    def handle(self, *args, **options):
        section, _ = CertificateSection.objects.update_or_create(
            is_active=True,
            defaults={
                "title": "Certificates",
                "description": "Professional certifications and achievements",
            },
        )

        created_count = 0
        updated_count = 0

        for cert_data in CERTIFICATES:
            certificate, created = CertificateItem.objects.update_or_create(
                slug=cert_data["slug"],
                defaults={
                    "section": section,
                    "title": cert_data["title"],
                    "mobile_title": cert_data["mobile_title"],
                    "issuer": cert_data["issuer"],
                    "issue_date": cert_data["issue_date"],
                    "alt_text": cert_data["alt_text"],
                    "skills": cert_data["skills"],
                    "sort_order": cert_data["sort_order"],
                    "is_active": True,
                },
            )

            self.attach_file_if_exists(
                certificate=certificate,
                field_name="certificate_image",
                relative_path=cert_data["image_path"],
            )

            self.attach_file_if_exists(
                certificate=certificate,
                field_name="certificate_file",
                relative_path=cert_data["file_path"],
            )

            certificate.save()

            if created:
                created_count += 1
            else:
                updated_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Certificates seeded successfully. Created {created_count}, updated {updated_count}."
            )
        )

    def attach_file_if_exists(self, certificate, field_name, relative_path):
        """
        Attach a local static file to the matching ImageField/FileField.

        Expected path example:
        frontend/public/assets/certificates/chatgpt.jpg

        Adjust STATIC_CERTIFICATE_BASE below if your frontend public folder
        is in a different location.
        """

        # Your Django folder:
        # C:/Users/Office/Downloads/backend_workportfolio/workportfolio
        #
        # If your frontend project is beside backend_workportfolio, adjust this path.
        STATIC_CERTIFICATE_BASE = os.path.abspath(
            os.path.join(
                settings.BASE_DIR,
                "..",
                "..",
                "frontend",
                "public",
            )
        )

        full_path = os.path.join(STATIC_CERTIFICATE_BASE, relative_path)

        if not os.path.exists(full_path):
            self.stdout.write(
                self.style.WARNING(
                    f"File not found for {field_name}: {full_path}"
                )
            )
            return

        current_file = getattr(certificate, field_name)

        # Avoid replacing if already uploaded.
        if current_file:
            return

        with open(full_path, "rb") as file_obj:
            django_file = File(file_obj)
            filename = os.path.basename(full_path)
            getattr(certificate, field_name).save(
                filename, django_file, save=False)
