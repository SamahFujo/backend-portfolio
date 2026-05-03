from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Career Timeline document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="career_timeline", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR(
                "❌ No active career timeline document found."))
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
                "career timeline",
                "overview",
                "summary",
            ],
            "Early Technical Experience": [
                "early technical",
                "customer-facing",
                "troubleshooting",
                "system support",
            ],
            "Customer / Requirement Experience": [
                "customer interaction",
                "requirement gathering",
                "stakeholders",
                "business needs",
            ],
            "Backend / Full-Stack Growth": [
                "full-stack",
                "backend development",
                "python",
                "django",
                "react",
                "next.js",
            ],
            "AI / ML Growth": [
                "ai",
                "machine learning",
                "nlp",
                "ocr",
                "llm integration",
                "embeddings",
                "retrieval-based systems",
            ],
            "Business-Focused AI Delivery": [
                "business-focused",
                "applied ai delivery",
                "intelligent automation",
                "production-focused engineering",
            ],
            "Leadership": [
                "leadership progression",
                "ai team lead",
                "technical direction",
                "solution design",
            ],
            "Current Profile / Direction": [
                "current professional profile",
                "career direction",
                "python backend development",
                "django and api design",
                "technical leadership",
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
        # 2. Chunk the document using the same route as ingestion
        # ---------------------------------------------------------
        chunks = ChunkService.chunk_document(
            raw_text=full_text,
            document_type="career_timeline",
            title=doc.title,
        )

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        self.stdout.write("#" * 80)

        for i, chunk in enumerate(chunks[:30]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(chunk[:1200])

        combined_chunks = "\n".join(chunks).lower()

        # ---------------------------------------------------------
        # 3. Required chunk coverage checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED CHUNK COVERAGE CHECKS")
        self.stdout.write("#" * 80)

        required_chunk_keywords = [
            "technical support",
            "troubleshooting technical issues",
            "customer interaction",
            "requirement gathering",
            "python development",
            "django and django rest framework",
            "react and next.js",
            "ai-powered classification systems",
            "document understanding and ocr workflows",
            "llm integration",
            "embeddings and retrieval-based systems",
            "applied ai delivery",
            "ai team lead",
            "current professional profile",
            "career direction",
            "technical leadership",
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
            "career progression questions": [
                "career has progressed",
                "role progression",
                "areas of growth",
            ],
            "early experience questions": [
                "technical support",
                "system support",
                "customer communication",
            ],
            "customer / stakeholder questions": [
                "customers and stakeholders",
                "requirement gathering",
                "business needs",
            ],
            "backend / full-stack questions": [
                "python development",
                "django and django rest framework",
                "react and next.js",
            ],
            "ai / llm questions": [
                "machine learning",
                "llm integration",
                "embeddings and retrieval-based systems",
            ],
            "leadership questions": [
                "leadership role",
                "ai team lead",
                "technical direction",
            ],
            "career direction questions": [
                "career direction",
                "backend engineering",
                "technical leadership",
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
        self.stdout.write("FINAL CAREER TIMELINE DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: Career Timeline document validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: Career Timeline document is parsed and chunked correctly."))
