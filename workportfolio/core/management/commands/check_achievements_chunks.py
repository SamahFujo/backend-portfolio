from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Achievements and Impact document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="achievements", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR(
                "❌ No active achievements document found."))
            return

        full_text = doc.raw_text or ""
        lower_text = full_text.lower()

        self.stdout.write(self.style.SUCCESS(f"DOCUMENT FOUND: {doc.title}"))
        self.stdout.write(f"RAW TEXT LENGTH: {len(full_text)}")

        # ---------------------------------------------------------
        # 1. Parser / raw text validation
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("RAW TEXT VALIDATION")
        self.stdout.write("#" * 80)

        required_raw_keywords = {
            "Document Identity": [
                "achievements and impact",
                "overview",
                "summary",
            ],
            "AI and Automation Impact": [
                "ai and automation",
                "reducing manual work",
                "structured outputs",
                "business processes",
            ],
            "Backend / Full-Stack Impact": [
                "backend",
                "full-stack",
                "apis",
                "user-friendly interfaces",
            ],
            "Business Value": [
                "business-focused",
                "dashboards",
                "classification",
                "decision support",
            ],
            "Leadership Impact": [
                "ai team lead",
                "ownership",
                "technical direction",
                "stakeholders",
            ],
            "Customer / Stakeholder Impact": [
                "customers",
                "stakeholders",
                "requirement clarity",
                "business need",
            ],
            "Outcome Areas": [
                "workflow automation",
                "smarter access to information",
                "operational visibility",
                "ai-enabled business tools",
            ],
        }

        failed_checks = []

        for group_name, keywords in required_raw_keywords.items():
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(group_name)

            for keyword in keywords:
                if keyword.lower() in lower_text:
                    self.stdout.write(
                        self.style.SUCCESS(f"✅ Found: {keyword}"))
                else:
                    self.stdout.write(self.style.ERROR(
                        f"❌ Missing: {keyword}"))
                    failed_checks.append(
                        f"RAW TEXT -> {group_name} -> {keyword}")

        # ---------------------------------------------------------
        # 2. Chunk the document
        # ---------------------------------------------------------
        chunks = ChunkService.chunk_generic(full_text)

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        self.stdout.write("#" * 80)

        for i, chunk in enumerate(chunks[:30]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(chunk[:1200])

        # ---------------------------------------------------------
        # 3. Required chunk coverage checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED CHUNK COVERAGE CHECKS")
        self.stdout.write("#" * 80)

        combined_chunks = "\n".join(chunks).lower()

        required_chunk_keywords = [
            "practical, production-oriented",
            "reducing manual work",
            "structured outputs",
            "strong backend apis",
            "full-stack systems",
            "dashboards and analytics",
            "classification or review tasks",
            "ai team lead",
            "technical direction",
            "stakeholders and customers",
            "workflow automation",
            "operational visibility",
            "structured digital solutions",
        ]

        for keyword in required_chunk_keywords:
            if keyword.lower() in combined_chunks:
                self.stdout.write(self.style.SUCCESS(
                    f"✅ Chunk coverage found: {keyword}"))
            else:
                self.stdout.write(self.style.ERROR(
                    f"❌ Chunk coverage missing: {keyword}"))
                failed_checks.append(f"CHUNKS -> {keyword}")

        # ---------------------------------------------------------
        # 4. Semantic area checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("SEMANTIC AREA CHECKS")
        self.stdout.write("#" * 80)

        semantic_checks = {
            "achievement / impact questions": [
                "impact",
                "value",
                "outcomes",
                "stands out",
            ],
            "automation questions": [
                "automation",
                "reducing manual work",
                "business processes",
            ],
            "backend / full-stack questions": [
                "backend apis",
                "full-stack systems",
                "maintainable project structures",
            ],
            "leadership questions": [
                "ownership",
                "technical direction",
                "implementation planning",
            ],
            "stakeholder questions": [
                "customers",
                "stakeholders",
                "requirement clarity",
            ],
        }

        for area, keywords in semantic_checks.items():
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHECKING AREA: {area}")

            area_passed = any(
                keyword.lower() in combined_chunks for keyword in keywords)

            if area_passed:
                self.stdout.write(self.style.SUCCESS(
                    f"✅ Area covered: {area}"))
            else:
                self.stdout.write(self.style.ERROR(
                    f"❌ Area not covered: {area}"))
                failed_checks.append(f"SEMANTIC AREA -> {area}")

        # ---------------------------------------------------------
        # 5. Final verdict
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FINAL ACHIEVEMENTS DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: Achievements document validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: Achievements document is parsed and chunked correctly."))
