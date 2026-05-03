from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Project Portfolio document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="projects", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR(
                "❌ No active projects document found."))
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
                "project portfolio",
                "compiled summary",
                "samah fujo",
                "closing note",
            ],
            "Electricity Dashboard": [
                "live smart electricity dashboard",
                "power consumption",
                "python",
                "dash",
                "oracle database",
                "deep learning",
                "forecasting",
            ],
            "Payroll Auditing": [
                "auditing employee payroll system",
                "workflow automation",
                "payroll",
                "salary-related anomalies",
                "manual auditing effort",
            ],
            "Spend Analysis": [
                "spend analysis dashboard",
                "unspsc classification",
                "django",
                "django rest framework",
                "react",
                "next.js",
                "bert",
                "roberta",
                "azure ad",
            ],
            "Property Chatbot": [
                "ai property search chatbot",
                "natural language property requests",
                "gemini",
                "ollama",
                "rapidfuzz",
                "follow-up questions",
                "excel-based property data",
            ],
            "Portfolio Website": [
                "samah.ai interactive portfolio website",
                "next.js",
                "typescript",
                "tailwind css",
                "chatbot-focused section",
                "personal branding",
            ],
            "Project Structure Fields": [
                "status",
                "technology stack",
                "overview",
                "business need",
                "your contribution",
                "key features",
                "outcome / business value",
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
            document_type="projects",
            title=doc.title,
        )

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        self.stdout.write("#" * 80)

        for i, chunk in enumerate(chunks[:40]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(chunk[:1500])

        combined_chunks = "\n".join(chunks).lower()

        # ---------------------------------------------------------
        # 3. Project chunk structure checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("PROJECT STRUCTURE CHECKS")
        self.stdout.write("#" * 80)

        expected_project_prefixes = [
            "Project: Live Smart Electricity Dashboard for Power Consumption",
            "Project: Auditing Employee Payroll System",
            "Project: Spend Analysis Dashboard with AI-based UNSPSC Classification",
            "Project: AI Property Search Chatbot",
            "Project: Samah.ai Interactive Portfolio Website",
        ]

        for project_prefix in expected_project_prefixes:
            project_chunks = [
                chunk for chunk in chunks
                if chunk.lower().startswith(project_prefix.lower())
            ]

            if project_chunks:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"✅ Found chunks for project: {project_prefix}"
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR(
                        f"❌ Missing chunks for project: {project_prefix}"
                    )
                )
                failed_checks.append(
                    f"STRUCTURE -> missing project chunks: {project_prefix}")

        # ---------------------------------------------------------
        # 4. Required chunk coverage checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED CHUNK COVERAGE CHECKS")
        self.stdout.write("#" * 80)

        required_chunk_keywords = [
            "live smart electricity dashboard",
            "oracle database",
            "forecasting future electricity consumption",
            "cnn-based prediction workflows",
            "auditing employee payroll system",
            "detect inconsistencies",
            "salary-related anomalies",
            "spend analysis dashboard",
            "ai-based unspsc classification",
            "django rest framework",
            "azure ad",
            "bert and roberta",
            "agreement leakage",
            "ai property search chatbot",
            "natural language query understanding",
            "gemini and ollama",
            "dynamic step-by-step refinement",
            "excel-based data lookup",
            "samah.ai interactive portfolio website",
            "next.js, react, typescript, tailwind css",
            "interactive projects showcase",
            "bot/chat-about-me concept",
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
        # 5. Semantic area checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("SEMANTIC AREA CHECKS")
        self.stdout.write("#" * 80)

        semantic_checks = {
            "dashboard project questions": [
                "live smart electricity dashboard",
                "interactive dashboards",
                "forecasting capabilities",
            ],
            "payroll automation questions": [
                "auditing employee payroll system",
                "workflow automation",
                "payroll mismatches",
            ],
            "procurement / UNSPSC questions": [
                "spend analysis dashboard",
                "unspsc categorization",
                "agreement leakage",
            ],
            "property chatbot questions": [
                "ai property search chatbot",
                "structured filters",
                "follow-up questions",
            ],
            "portfolio website questions": [
                "samah.ai interactive portfolio website",
                "interactive projects showcase",
                "professional branding",
            ],
            "technology stack questions": [
                "technology stack",
                "django",
                "next.js",
                "python",
                "react",
            ],
            "business value questions": [
                "outcome / business value",
                "manual burden",
                "operational planning",
                "decision-making",
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
        # 6. Final verdict
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FINAL PROJECTS DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: Project Portfolio validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: Project Portfolio is parsed and chunked correctly."))
