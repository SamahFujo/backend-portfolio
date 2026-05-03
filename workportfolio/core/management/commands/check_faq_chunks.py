from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Samah FAQ document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="faq", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(self.style.ERROR("❌ No active FAQ document found."))
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
                "samah faq",
                "overview",
                "summary",
            ],
            "Core Profile": [
                "what does samah do?",
                "ai team lead",
                "ai/ml engineer",
                "full-stack development",
            ],
            "Technical Strengths": [
                "strongest technical areas",
                "python backend development",
                "django rest framework",
                "ai and llm integration",
                "workflow automation",
            ],
            "Technology Preferences": [
                "favorite programming language",
                "python",
                "preferred backend framework",
                "django",
                "next.js",
                "react",
                "tailwind css",
            ],
            "AI / RAG / OCR": [
                "ai-powered chatbots",
                "rag-based systems",
                "retrieval-augmented generation",
                "ocr",
                "document-processing",
                "tesseract",
                "easyocr",
                "opencv",
            ],
            "Projects / Business Systems": [
                "business dashboards",
                "live smart electricity dashboard",
                "payroll system",
                "spend analysis dashboard",
                "ai property search chatbot",
            ],
            "Leadership / Stakeholders": [
                "leadership experience",
                "ai team lead",
                "clients or stakeholders",
                "gather requirements",
                "present solutions",
            ],
            "Availability / Role Fit": [
                "full-time work",
                "freelance or contract work",
                "remote work",
                "dubai",
                "abu dhabi",
                "uae",
                "roles fit samah best",
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
            document_type="faq",
            title=doc.title,
        )

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        self.stdout.write("#" * 80)

        for i, chunk in enumerate(chunks[:40]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(chunk[:1200])

        combined_chunks = "\n".join(chunks).lower()

        # ---------------------------------------------------------
        # 3. FAQ structure checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FAQ STRUCTURE CHECKS")
        self.stdout.write("#" * 80)

        if len(chunks) >= 20:
            self.stdout.write(self.style.SUCCESS(f"✅ FAQ produced enough Q&A chunks: {len(chunks)}"))
        else:
            self.stdout.write(self.style.ERROR(f"❌ FAQ produced too few chunks: {len(chunks)}"))
            failed_checks.append(f"FAQ STRUCTURE -> too few chunks: {len(chunks)}")

        malformed_chunks = []

        for i, chunk in enumerate(chunks):
            has_question = "FAQ Question:" in chunk
            has_answer = "FAQ Answer:" in chunk

            if has_question and has_answer:
                self.stdout.write(self.style.SUCCESS(f"✅ Chunk {i} has question and answer format."))
            else:
                self.stdout.write(self.style.ERROR(f"❌ Chunk {i} is malformed."))
                malformed_chunks.append(i)

        if malformed_chunks:
            failed_checks.append(f"FAQ STRUCTURE -> malformed chunks: {malformed_chunks}")

        # ---------------------------------------------------------
        # 4. Required Q&A coverage checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED Q&A COVERAGE CHECKS")
        self.stdout.write("#" * 80)

        required_chunk_keywords = [
            "FAQ Question: What does Samah do?",
            "FAQ Answer: Samah is an AI Team Lead and AI/ML Engineer",
            "FAQ Question: What are Samah’s strongest technical areas?",
            "python backend development",
            "django and django rest framework",
            "FAQ Question: What is Samah’s favorite programming language?",
            "samah’s favorite programming language is python",
            "FAQ Question: What backend framework does Samah prefer most?",
            "preferred backend framework is django",
            "FAQ Question: Can Samah build AI-powered chatbots?",
            "document-grounded chatbots",
            "FAQ Question: Can Samah build RAG-based systems?",
            "retrieval-augmented generation systems",
            "FAQ Question: Can Samah work on OCR and document-processing solutions?",
            "tesseract",
            "easyocr",
            "opencv",
            "FAQ Question: What types of projects has Samah built?",
            "live smart electricity dashboard",
            "spend analysis dashboard",
            "ai property search chatbot",
            "FAQ Question: Has Samah worked with AI and LLM technologies?",
            "hugging face transformers",
            "gemini",
            "ollama",
            "langchain",
            "FAQ Question: Does Samah have leadership experience?",
            "leadership experience as an ai team lead",
            "FAQ Question: Is Samah open to freelance or contract work?",
            "freelance, contract-based, consulting, and project-based opportunities",
            "FAQ Question: Which locations is Samah especially open to?",
            "dubai",
            "abu dhabi",
            "uae",
            "FAQ Question: What kinds of business problems is Samah best at solving?",
            "automating manual workflows",
            "unstructured data into useful structured outputs",
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
            "identity / summary questions": [
                "what does samah do?",
                "ai team lead and ai/ml engineer",
            ],
            "technical strength questions": [
                "strongest technical areas",
                "python backend development",
                "workflow automation",
            ],
            "preference questions": [
                "favorite programming language",
                "preferred backend framework",
                "preferred frontend stack",
            ],
            "ai / rag / ocr questions": [
                "ai-powered chatbots",
                "rag-based systems",
                "ocr and document-processing",
            ],
            "project questions": [
                "types of projects has samah built",
                "live smart electricity dashboard",
                "spend analysis dashboard",
            ],
            "leadership / stakeholder questions": [
                "leadership experience",
                "clients or stakeholders",
                "gather requirements",
            ],
            "availability questions": [
                "full-time opportunities",
                "freelance, contract-based",
                "remote work",
            ],
            "location questions": [
                "dubai",
                "abu dhabi",
                "uae",
            ],
            "role fit questions": [
                "roles that fit samah best",
                "ai engineer",
                "backend engineer",
                "technical lead",
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
        self.stdout.write("FINAL FAQ DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR("❌ FAIL: FAQ document validation failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS("✅ PASS: FAQ document is parsed and chunked correctly."))