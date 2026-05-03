from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Compensation and Availability document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="compensation", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR(
                "❌ No active compensation document found."))
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
                "compensation and availability",
                "overview",
                "summary",
            ],
            "Salary / Compensation": [
                "expected salary range",
                "target compensation range",
                "fair and competitive compensation package",
                "role scope",
                "seniority level",
            ],
            "Work Type": [
                "preferred work type",
                "full-time employment",
                "contract-based roles",
                "freelance projects",
                "consulting",
            ],
            "Work Arrangement": [
                "preferred work arrangement",
                "remote work",
                "hybrid work",
                "on-site work",
            ],
            "Geographic Preference": [
                "geographic preference",
                "dubai",
                "abu dhabi",
                "uae",
                "remote opportunities",
            ],
            "Availability": [
                "availability for opportunities",
                "open to hearing",
                "ai",
                "backend engineering",
                "full-stack development",
                "intelligent automation",
            ],
            "Freelance / Project-Based": [
                "freelance and project-based work",
                "backend apis",
                "ai chatbot systems",
                "rag-based assistants",
                "document processing",
                "ocr workflows",
                "django and next.js-based solutions",
            ],
            "Role Fit": [
                "role fit consideration",
                "hands-on technical implementation",
                "backend and ai system design",
                "direct business impact",
                "production-oriented delivery",
            ],
            "Discussion Style": [
                "compensation discussion style",
                "professional and transparent",
                "market competitiveness",
                "ownership level",
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
            document_type="compensation",
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
            "expected salary range",
            "target compensation range",
            "fair and competitive compensation package",
            "full-time employment",
            "contract-based roles",
            "freelance projects",
            "consulting or solution-based technical work",
            "remote work",
            "hybrid work",
            "on-site work",
            "dubai",
            "abu dhabi",
            "remote opportunities",
            "availability for opportunities",
            "ai or llm-enabled systems",
            "django or python-based development",
            "freelance or project-based work",
            "backend apis",
            "rag-based assistants",
            "document processing and ocr workflows",
            "django and next.js-based solutions",
            "role fit consideration",
            "hands-on technical implementation",
            "backend and ai system design",
            "professional and transparent compensation discussion",
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
            "salary / compensation questions": [
                "expected salary range",
                "target compensation range",
                "fair and competitive compensation package",
            ],
            "full-time / contract questions": [
                "full-time employment",
                "contract-based roles",
                "consulting or solution-based technical work",
            ],
            "freelance questions": [
                "freelance projects",
                "freelance or project-based work",
                "rag-based assistants",
            ],
            "remote / hybrid / onsite questions": [
                "remote work",
                "hybrid work",
                "on-site work",
            ],
            "location questions": [
                "dubai",
                "abu dhabi",
                "uae",
                "remote opportunities",
            ],
            "role fit questions": [
                "backend and ai system design",
                "hands-on technical implementation",
                "direct business impact",
            ],
            "discussion style questions": [
                "professional and transparent",
                "market competitiveness",
                "ownership level",
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
        self.stdout.write("FINAL COMPENSATION DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: Compensation document validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: Compensation document is parsed and chunked correctly."))
