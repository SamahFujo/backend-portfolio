from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Certificates Portfolio document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="certificates", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR("❌ No active certificates document found."))
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
                "certificates portfolio",
                "certificate register",
                "professional relevance",
            ],
            "Master of ChatGPT": [
                "master of chatgpt",
                "coursiv",
                "5 february 2026",
                "prompt engineering",
                "structured assistant usage",
            ],
            "Master of Claude": [
                "master of claude",
                "coursiv",
                "16 february 2026",
                "multi-model ai fluency",
                "generative ai systems",
            ],
            "React and Django Certificate": [
                "react & django full stack",
                "udemy",
                "25 january 2025",
                "full-stack development",
                "django and react",
            ],
            "AI Agents Certificate": [
                "fundamentals of building ai agents",
                "ibm via coursera",
                "16 november 2025",
                "agentic ai systems",
                "workflow agents",
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
            document_type="certificates",
            title=doc.title,
        )

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        self.stdout.write("#" * 80)

        for i, chunk in enumerate(chunks[:30]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(chunk[:1500])

        combined_chunks = "\n".join(chunks).lower()

        # ---------------------------------------------------------
        # 3. Certificate structure checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("CERTIFICATE STRUCTURE CHECKS")
        self.stdout.write("#" * 80)

        expected_certificate_titles = [
            "Certificate: Master of ChatGPT",
            "Certificate: Master of Claude",
            "Certificate: React & Django Full Stack: Web App, Backend API, Mobile Apps",
            "Certificate: Fundamentals of Building AI Agents",
        ]

        if len(chunks) == 4:
            self.stdout.write(self.style.SUCCESS("✅ Certificate chunk count is correct: 4"))
        else:
            self.stdout.write(self.style.ERROR(f"❌ Expected 4 certificate chunks, got {len(chunks)}"))
            failed_checks.append(f"STRUCTURE -> expected 4 chunks, got {len(chunks)}")

        for title in expected_certificate_titles:
            if title.lower() in combined_chunks:
                self.stdout.write(self.style.SUCCESS(f"✅ Found certificate chunk: {title}"))
            else:
                self.stdout.write(self.style.ERROR(f"❌ Missing certificate chunk: {title}"))
                failed_checks.append(f"STRUCTURE -> missing {title}")

        malformed_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()

            has_certificate = "certificate:" in chunk_lower
            has_issuer = "issuer:" in chunk_lower
            has_date = "date:" in chunk_lower
            has_focus = "focus:" in chunk_lower
            has_why = "why it matters:" in chunk_lower

            if has_certificate and has_issuer and has_date and has_focus and has_why:
                self.stdout.write(self.style.SUCCESS(f"✅ Chunk {i} has full certificate structure."))
            else:
                self.stdout.write(self.style.ERROR(f"❌ Chunk {i} is missing certificate fields."))
                malformed_chunks.append(i)

        if malformed_chunks:
            failed_checks.append(f"STRUCTURE -> malformed chunks: {malformed_chunks}")

        # ---------------------------------------------------------
        # 4. Required chunk coverage checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED CHUNK COVERAGE CHECKS")
        self.stdout.write("#" * 80)

        required_chunk_keywords = [
            "master of chatgpt",
            "issuer: coursiv",
            "date: 5 february 2026",
            "prompting, advanced usage patterns",
            "master of claude",
            "date: 16 february 2026",
            "multi-model ai fluency",
            "react & django full stack",
            "issuer: udemy",
            "date: 25 january 2025",
            "full-stack development spanning react, django",
            "fundamentals of building ai agents",
            "issuer: ibm via coursera",
            "date: 16 november 2025",
            "agentic ai systems",
            "workflow agents",
        ]

        for keyword in required_chunk_keywords:
            if keyword.lower() in combined_chunks:
                self.stdout.write(self.style.SUCCESS(f"✅ Chunk coverage found: {keyword}"))
            else:
                self.stdout.write(self.style.ERROR(f"❌ Chunk coverage missing: {keyword}"))
                failed_checks.append(f"CHUNKS -> {keyword}")

        # ---------------------------------------------------------
        # 5. Semantic area checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("SEMANTIC AREA CHECKS")
        self.stdout.write("#" * 80)

        semantic_checks = {
            "certificate list questions": [
                "master of chatgpt",
                "master of claude",
                "react & django full stack",
                "fundamentals of building ai agents",
            ],
            "issuer questions": [
                "issuer: coursiv",
                "issuer: udemy",
                "issuer: ibm via coursera",
            ],
            "date questions": [
                "date: 5 february 2026",
                "date: 16 february 2026",
                "date: 25 january 2025",
                "date: 16 november 2025",
            ],
            "ai / llm certificate questions": [
                "prompt engineering",
                "generative ai systems",
                "agentic ai systems",
            ],
            "full-stack certificate questions": [
                "react & django full stack",
                "full-stack development",
                "django and react",
            ],
            "professional relevance questions": [
                "why it matters:",
                "solution delivery",
                "workflow agents",
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
        # 6. Final verdict
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FINAL CERTIFICATES DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR("❌ FAIL: Certificates document validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS("✅ PASS: Certificates document is parsed and chunked correctly."))