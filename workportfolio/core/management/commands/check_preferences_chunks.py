from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Preferences and Work Style document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="preferences", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR(
                "❌ No active preferences document found."))
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
                "preferences and work style",
                "overview",
                "summary",
            ],
            "Favorite Language": [
                "favorite programming language",
                "python",
                "backend development",
                "ai workflows",
            ],
            "Backend Preference": [
                "preferred backend framework",
                "django",
                "django rest framework",
                "fastapi",
                "flask",
            ],
            "Frontend Preference": [
                "preferred frontend stack",
                "next.js",
                "react",
                "tailwind css",
            ],
            "Preferred Work Type": [
                "preferred type of work",
                "backend engineering",
                "ai or llm integration",
                "full-stack solution delivery",
            ],
            "Enjoyed Areas": [
                "areas i enjoy most",
                "ai-powered web applications",
                "chatbots and intelligent assistants",
                "document-based ai solutions",
                "dashboards",
            ],
            "Backend vs Frontend vs AI": [
                "backend vs frontend vs ai",
                "backend development and ai-focused work",
                "frontend development",
                "solution design",
            ],
            "Project Style": [
                "preferred project style",
                "production-oriented",
                "scalable",
                "secure",
                "business needs",
            ],
            "Technical Environment": [
                "preferred technical environment",
                "clean and organized codebases",
                "enterprise-style architecture",
                "api-first thinking",
            ],
            "Working / Collaboration Style": [
                "working style",
                "structured",
                "detail-oriented",
                "solution-focused",
                "collaboration style",
                "stakeholders",
            ],
            "Role Fit": [
                "types of roles that fit me best",
                "ai engineer",
                "backend engineer",
                "senior python developer",
                "django developer",
                "technical lead",
            ],
            "Preferred Technologies": [
                "technologies i prefer working with",
                "postgresql",
                "mongodb",
                "hugging face transformers",
                "google gemini",
                "ollama",
                "langchain",
            ],
            "Problem Solving": [
                "type of problems i like solving",
                "unstructured input into structured results",
                "automate repetitive or manual tasks",
                "smart interfaces",
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
            document_type="preferences",
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
            "favorite programming language",
            "python is the language",
            "preferred backend framework",
            "django rest framework",
            "fastapi and flask",
            "preferred frontend stack",
            "next.js with react and tailwind css",
            "backend engineering",
            "ai or llm integration",
            "full-stack solution delivery",
            "ai-powered web applications",
            "chatbots and intelligent assistants",
            "document-based ai solutions",
            "backend development and ai-focused work",
            "preferred project style",
            "production-oriented",
            "enterprise-style architecture",
            "working style",
            "structured",
            "detail-oriented",
            "collaboration style",
            "types of roles that fit me best",
            "senior python developer",
            "technical lead or ai team lead",
            "technologies i prefer working with",
            "hugging face transformers",
            "type of problems i like solving",
            "unstructured input into structured results",
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
            "favorite language questions": [
                "favorite programming language",
                "python is the language",
            ],
            "backend framework questions": [
                "preferred backend framework",
                "django rest framework",
                "fastapi and flask",
            ],
            "frontend stack questions": [
                "preferred frontend stack",
                "next.js with react and tailwind css",
            ],
            "work preference questions": [
                "backend engineering",
                "ai or llm integration",
                "full-stack solution delivery",
            ],
            "project preference questions": [
                "preferred project style",
                "production-oriented",
                "business needs",
            ],
            "working style questions": [
                "working style",
                "structured",
                "detail-oriented",
                "solution-focused",
            ],
            "collaboration questions": [
                "collaboration style",
                "stakeholders directly",
                "cross-functional teams",
            ],
            "role fit questions": [
                "types of roles that fit me best",
                "ai engineer",
                "backend engineer",
                "technical lead or ai team lead",
            ],
            "technology preference questions": [
                "technologies i prefer working with",
                "postgresql",
                "google gemini",
                "langchain",
            ],
            "problem-solving preference questions": [
                "type of problems i like solving",
                "unstructured input into structured results",
                "automate repetitive or manual tasks",
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
        self.stdout.write("FINAL PREFERENCES DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: Preferences document validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: Preferences document is parsed and chunked correctly."))
