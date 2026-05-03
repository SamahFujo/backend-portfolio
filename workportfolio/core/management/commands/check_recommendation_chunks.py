from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Recommendation Letter document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="recommendation", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR("❌ No active recommendation document found."))
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
                "to whom it may concern",
                "letter of recommendation",
                "12-08-2025",
            ],
            "Recommended Person / Role": [
                "samah fujo",
                "ai team leader",
                "artificial intelligence research and development",
                "nasser centre for science & technology",
            ],
            "Professional Strengths": [
                "professionalism",
                "reliability",
                "commitment",
                "integrity",
                "attention to detail",
                "accountability",
            ],
            "Collaboration / Attitude": [
                "collaborate effectively",
                "adapt to new challenges",
                "positive and proactive attitude",
                "indispensable asset",
            ],
            "Interpersonal / Organizational Skills": [
                "interpersonal skills",
                "collaborative relationships",
                "internal teams",
                "external partners",
                "organizational abilities",
            ],
            "Recommendation Statement": [
                "strongly recommend",
                "future role",
                "dedication",
                "professionalism",
                "excellence",
            ],
            "Issuer": [
                "aysha abdulrahman al fadhel",
                "director of human resources & service",
            ],
        }

        failed_checks = []

        for group_name, keywords in required_raw_keywords.items():
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(group_name)

            for keyword in keywords:
                if keyword.lower() in lower_text:
                    self.stdout.write(self.style.SUCCESS(f"✅ Found: {keyword}"))
                else:
                    self.stdout.write(self.style.ERROR(f"❌ Missing: {keyword}"))
                    failed_checks.append(f"RAW TEXT -> {group_name} -> {keyword}")

        # ---------------------------------------------------------
        # 2. Chunk the document using the same route as ingestion
        # ---------------------------------------------------------
        chunks = ChunkService.chunk_document(
            raw_text=full_text,
            document_type="recommendation",
            title=doc.title,
        )

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        self.stdout.write("#" * 80)

        for i, chunk in enumerate(chunks[:20]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(chunk[:1500])

        combined_chunks = "\n".join(chunks).lower()

        # ---------------------------------------------------------
        # 3. Required chunk coverage checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED CHUNK COVERAGE CHECKS")
        self.stdout.write("#" * 80)

        required_chunk_keywords = [
            "letter of recommendation",
            "samah fujo",
            "ai team leader",
            "professionalism",
            "reliability",
            "commitment",
            "integrity",
            "attention to detail",
            "accountability",
            "collaborate effectively",
            "adapt to new challenges",
            "positive and proactive attitude",
            "interpersonal skills",
            "organizational abilities",
            "strongly recommend",
            "aysha abdulrahman al fadhel",
            "director of human resources & service",
        ]

        for keyword in required_chunk_keywords:
            if keyword.lower() in combined_chunks:
                self.stdout.write(self.style.SUCCESS(f"✅ Chunk coverage found: {keyword}"))
            else:
                self.stdout.write(self.style.ERROR(f"❌ Chunk coverage missing: {keyword}"))
                failed_checks.append(f"CHUNKS -> {keyword}")

        # ---------------------------------------------------------
        # 4. Semantic area checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("SEMANTIC AREA CHECKS")
        self.stdout.write("#" * 80)

        semantic_checks = {
            "recommendation / reference questions": [
                "letter of recommendation",
                "strongly recommend",
                "future role",
            ],
            "professionalism questions": [
                "professionalism",
                "reliability",
                "commitment",
                "integrity",
            ],
            "work style / attitude questions": [
                "attention to detail",
                "accountability",
                "positive and proactive attitude",
            ],
            "collaboration questions": [
                "collaborate effectively",
                "interpersonal skills",
                "collaborative relationships",
            ],
            "organization / delivery questions": [
                "organizational abilities",
                "completed efficiently",
                "highest standards",
            ],
            "issuer / HR questions": [
                "aysha abdulrahman al fadhel",
                "director of human resources & service",
            ],
        }

        for area, keywords in semantic_checks.items():
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHECKING AREA: {area}")

            area_passed = any(keyword.lower() in combined_chunks for keyword in keywords)

            if area_passed:
                self.stdout.write(self.style.SUCCESS(f"✅ Area covered: {area}"))
            else:
                self.stdout.write(self.style.ERROR(f"❌ Area not covered: {area}"))
                failed_checks.append(f"SEMANTIC AREA -> {area}")

        # ---------------------------------------------------------
        # 5. Final verdict
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FINAL RECOMMENDATION DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR("❌ FAIL: Recommendation document validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS("✅ PASS: Recommendation document is parsed and chunked correctly."))