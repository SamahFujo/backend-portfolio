from django.core.management.base import BaseCommand

from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):
    help = "Check Security Deployment document parsing and chunking"

    def handle(self, *args, **options):
        doc = (
            ProfileDocument.objects
            .filter(document_type="security_deployment", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not doc:
            self.stdout.write(
                self.style.ERROR(
                    "❌ No active security_deployment document found.")
            )
            return

        full_text = doc.raw_text or ""
        lower_text = full_text.lower()

        self.stdout.write(self.style.SUCCESS(f"DOCUMENT FOUND: {doc.title}"))
        self.stdout.write(f"DOCUMENT TYPE: {doc.document_type}")
        self.stdout.write(f"STATUS: {doc.status}")
        self.stdout.write(f"RAW TEXT LENGTH: {len(full_text)}")

        # ---------------------------------------------------------
        # 1. Parser / raw text validation
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("RAW TEXT VALIDATION")
        self.stdout.write("#" * 80)

        required_raw_keywords = {
            "Document Identity": [
                "security skills",
                "deployment & devops skills",
                "representative project evidence",
                "professional positioning statement",
            ],
            "Security Skills": [
                "authentication",
                "authorization",
                "jwt",
                "rbac",
                "microsoft entra",
                "azure ad",
                "cors",
                "csrf",
                "rate throttling",
                "secure secret management",
                "prompt-injection",
                "auditability",
                "https",
                "ssl",
                "nginx",
            ],
            "Deployment and DevOps Skills": [
                "django",
                "react",
                "next.js",
                "aws ec2",
                "gunicorn",
                "nginx",
                "supervisor",
                "postgresql",
                "rds",
                "oracle",
                "docker",
                "ollama",
                "logging",
                "monitoring",
                "backups",
                "postman",
                "pinned dependencies",
            ],
            "Representative Project Evidence": [
                "spend analysis dashboard",
                "samah.ai portfolio chatbot",
                "ai property search chatbot",
                "smart electricity monitoring dashboard",
                "azure ad",
                "rbac",
                "rate limits",
                "injection-aware",
                "database connectivity",
            ],
            "Professional Positioning": [
                "software engineering",
                "ai/ml implementation",
                "security-aware backend design",
                "secure apis",
                "interactive frontend interfaces",
                "production deployment",
            ],
        }

        failed_checks = []

        for group_name, keywords in required_raw_keywords.items():
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(group_name)

            for keyword in keywords:
                if keyword.lower() in lower_text:
                    self.stdout.write(
                        self.style.SUCCESS(f"✅ Found: {keyword}")
                    )
                else:
                    self.stdout.write(
                        self.style.ERROR(f"❌ Missing: {keyword}")
                    )
                    failed_checks.append(
                        f"RAW TEXT -> {group_name} -> {keyword}"
                    )

        # ---------------------------------------------------------
        # 2. Chunk the document
        # ---------------------------------------------------------
        chunks = ChunkService.chunk_document(
            raw_text=full_text,
            document_type="security_deployment",
            title=doc.title,
        )

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        self.stdout.write("#" * 80)

        if not chunks:
            failed_checks.append("CHUNKS -> No chunks generated")

        for i, chunk in enumerate(chunks[:30]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(f"LENGTH: {len(chunk)}")
            self.stdout.write("-" * 80)
            self.stdout.write(chunk[:1500])

        # ---------------------------------------------------------
        # 3. Required chunk coverage checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED CHUNK COVERAGE CHECKS")
        self.stdout.write("#" * 80)

        combined_chunks = "\n".join(chunks).lower()

        required_chunk_keywords = [
            "authentication and authorization",
            "jwt",
            "rbac",
            "microsoft entra",
            "azure ad",
            "permission classes",
            "internal api keys",
            "cors",
            "csrf",
            "rate throttling",
            "environment variables",
            "aws secrets manager",
            "prompt-injection",
            "grounded responses",
            "safe fallback behavior",
            "user activity logs",
            "access logs",
            "https",
            "ssl",
            "nginx reverse proxy",
            "gunicorn",
            "supervisor",
            "aws ec2",
            "postgresql",
            "rds",
            "oracle data warehouse",
            "docker",
            "ollama",
            "logging",
            "monitoring",
            "backups",
            "postman",
            "next.js",
            "production deployment",
        ]

        for keyword in required_chunk_keywords:
            if keyword.lower() in combined_chunks:
                self.stdout.write(
                    self.style.SUCCESS(f"✅ Chunk coverage found: {keyword}")
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f"❌ Chunk coverage missing: {keyword}")
                )
                failed_checks.append(f"CHUNKS -> {keyword}")

        # ---------------------------------------------------------
        # 4. Section prefix checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("SECTION PREFIX CHECKS")
        self.stdout.write("#" * 80)

        expected_section_markers = [
            "Security Deployment Section: SECURITY SKILLS",
            "Security Deployment Section: DEPLOYMENT & DEVOPS SKILLS",
            "Security Deployment Section: REPRESENTATIVE PROJECT EVIDENCE",
            "Security Deployment Section: PROFESSIONAL POSITIONING STATEMENT",
        ]

        for marker in expected_section_markers:
            if marker.lower() in combined_chunks:
                self.stdout.write(
                    self.style.SUCCESS(f"✅ Section found: {marker}")
                )
            else:
                self.stdout.write(
                    self.style.WARNING(
                        f"⚠️ Section prefix not found: {marker}")
                )
                failed_checks.append(f"SECTION PREFIX -> {marker}")

        # ---------------------------------------------------------
        # 5. Semantic area checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("SEMANTIC AREA CHECKS")
        self.stdout.write("#" * 80)

        semantic_checks = {
            "security questions": [
                "authentication",
                "authorization",
                "jwt",
                "rbac",
                "cors",
                "csrf",
                "rate throttling",
                "secure cookies",
            ],
            "deployment questions": [
                "aws ec2",
                "gunicorn",
                "nginx",
                "supervisor",
                "ssl",
                "https",
                "production deployment",
            ],
            "database deployment questions": [
                "postgresql",
                "rds",
                "oracle",
                "migrations",
                "data synchronization",
            ],
            "devops and production readiness questions": [
                "docker",
                "logging",
                "monitoring",
                "backups",
                "pinned dependencies",
                "postman",
            ],
            "llm security questions": [
                "prompt-injection",
                "grounded responses",
                "safe fallback",
                "llm",
                "chatbot",
            ],
            "project evidence questions": [
                "spend analysis dashboard",
                "samah.ai portfolio chatbot",
                "ai property search chatbot",
                "smart electricity monitoring dashboard",
            ],
        }

        for area, keywords in semantic_checks.items():
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHECKING AREA: {area}")

            area_passed = any(
                keyword.lower() in combined_chunks for keyword in keywords
            )

            if area_passed:
                self.stdout.write(
                    self.style.SUCCESS(f"✅ Area covered: {area}")
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f"❌ Area not covered: {area}")
                )
                failed_checks.append(f"SEMANTIC AREA -> {area}")

        # ---------------------------------------------------------
        # 6. Saved database chunk checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("DATABASE SAVED CHUNK CHECKS")
        self.stdout.write("#" * 80)

        saved_chunks = doc.chunks.order_by("chunk_index")
        saved_chunk_count = saved_chunks.count()

        self.stdout.write(f"SAVED DB CHUNKS: {saved_chunk_count}")

        if saved_chunk_count == 0:
            self.stdout.write(
                self.style.ERROR("❌ No saved chunks found in database.")
            )
            failed_checks.append("DATABASE -> No saved chunks")
        else:
            for chunk in saved_chunks[:30]:
                has_embedding = chunk.embedding is not None

                self.stdout.write("\n" + "=" * 80)
                self.stdout.write(f"DB CHUNK INDEX: {chunk.chunk_index}")
                self.stdout.write(f"CONTENT LENGTH: {len(chunk.content)}")
                self.stdout.write(f"HAS EMBEDDING: {has_embedding}")

                if has_embedding:
                    self.stdout.write(
                        self.style.SUCCESS("✅ Embedding exists")
                    )
                else:
                    self.stdout.write(
                        self.style.ERROR("❌ Missing embedding")
                    )
                    failed_checks.append(
                        f"DATABASE -> Chunk {chunk.chunk_index} missing embedding"
                    )

                self.stdout.write("-" * 80)
                self.stdout.write(chunk.content[:800])

        # ---------------------------------------------------------
        # 7. Final verdict
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FINAL SECURITY DEPLOYMENT DOCUMENT VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(
                self.style.ERROR(
                    "❌ FAIL: Security Deployment document validation failed."
                )
            )

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "✅ PASS: Security Deployment document is parsed, chunked, embedded, and saved correctly."
                )
            )
