from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check What I Can Help With / Capabilities document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="capabilities", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR(
                "❌ No active capabilities document found."))
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
                "what i can help with",
                "overview",
                "summary",
            ],
            "Core Profile": [
                "ai team lead",
                "ai/ml engineer",
                "full-stack development",
                "python backend engineering",
                "django-based api development",
            ],
            "AI and LLM Solutions": [
                "ai and llm solutions",
                "ai-powered applications",
                "retrieval-augmented generation",
                "rag",
                "prompt engineering",
                "ocr and document understanding",
                "llm-based information extraction",
            ],
            "Backend Development": [
                "backend development",
                "django and django rest framework",
                "api design",
                "rbac",
                "jwt",
                "admin panels",
                "structured backend architecture",
            ],
            "Full-Stack Web Applications": [
                "full-stack web applications",
                "django backend and next.js frontend",
                "responsive dashboards",
                "interactive chatbot interfaces",
                "role-based web applications",
            ],
            "Business Systems": [
                "data and ai-powered business systems",
                "procurement and spend analysis",
                "unspsc",
                "monitoring dashboards",
                "payroll auditing",
                "property search",
            ],
            "Frontend / UX": [
                "frontend and user experience",
                "next.js and react",
                "tailwind css",
                "chatbot user experiences",
                "admin dashboard interfaces",
            ],
            "Leadership": [
                "technical leadership and delivery",
                "translating business requirements",
                "leading ai or software development initiatives",
                "communicating with clients",
                "supporting junior developers",
            ],
            "Project Types": [
                "types of projects i can build",
                "ai-powered chatbots",
                "rag-based question-answering systems",
                "dashboard and analytics platforms",
                "document analysis and ocr systems",
                "workflow automation systems",
            ],
            "Technologies": [
                "technologies i use professionally",
                "python",
                "javascript",
                "typescript",
                "django",
                "react",
                "next.js",
                "hugging face transformers",
                "postgresql",
                "docker",
            ],
            "Ramp-Up / Not Specialized": [
                "work i can do with some ramp-up",
                "areas i do not primarily specialize in",
                "native ios development",
                "native android development",
                "heavy game development",
            ],
            "Problem Solving": [
                "the kind of problems i am best at solving",
                "manual business workflows",
                "unstructured user input",
                "structured business actions",
                "secure backend logic",
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
            document_type="capabilities",
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
        # 3. Required chunk coverage checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED CHUNK COVERAGE CHECKS")
        self.stdout.write("#" * 80)

        required_chunk_keywords = [
            "ai and llm solutions",
            "ai chatbots for customer support",
            "retrieval-augmented generation",
            "prompt engineering and structured output pipelines",
            "ocr and document understanding workflows",
            "backend development",
            "django and django rest framework backend development",
            "authentication and authorization using rbac and jwt",
            "full-stack web applications",
            "django backend and next.js frontend",
            "responsive dashboards",
            "data and ai-powered business systems",
            "procurement and spend analysis systems",
            "classification solutions using business taxonomies such as unspsc",
            "frontend and user experience",
            "next.js and react user interfaces",
            "technical leadership and delivery",
            "translating business requirements into technical solutions",
            "types of projects i can build",
            "portfolio chatbots grounded on personal or business documents",
            "technologies i use professionally",
            "hugging face transformers",
            "embeddings and vector search",
            "work i can do with some ramp-up",
            "areas i do not primarily specialize in",
            "native ios development",
            "the kind of problems i am best at solving",
            "turning manual business workflows into intelligent digital systems",
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
            "general capability questions": [
                "what i can help with",
                "types of work i can confidently support",
                "production-ready ai solutions",
            ],
            "ai / llm questions": [
                "ai and llm solutions",
                "retrieval-augmented generation",
                "llm-based information extraction",
            ],
            "backend api questions": [
                "backend development",
                "django and django rest framework backend development",
                "api design and implementation",
            ],
            "full-stack questions": [
                "full-stack web applications",
                "django backend and next.js frontend",
                "interactive chatbot interfaces",
            ],
            "business system questions": [
                "procurement and spend analysis systems",
                "monitoring dashboards",
                "payroll auditing",
            ],
            "frontend / ux questions": [
                "frontend and user experience",
                "next.js and react user interfaces",
                "tailwind css-based responsive design",
            ],
            "leadership questions": [
                "technical leadership and delivery",
                "leading ai or software development initiatives",
                "communicating with clients",
            ],
            "technology stack questions": [
                "technologies i use professionally",
                "django rest framework",
                "google gemini",
                "postgresql",
            ],
            "not specialized questions": [
                "areas i do not primarily specialize in",
                "native ios development",
                "heavy game development",
            ],
            "problem-solving questions": [
                "the kind of problems i am best at solving",
                "manual business workflows",
                "unstructured user input",
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
        self.stdout.write("FINAL CAPABILITIES DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: Capabilities document validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: Capabilities document is parsed and chunked correctly."))
