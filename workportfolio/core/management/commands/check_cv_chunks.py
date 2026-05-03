# This is a Django management command to check the CV chunking logic.
# It loads the most recent CV document, extracts sections based on headings,
# and then applies the chunking logic to see how the CV is being split.
# To run this command, use: python manage.py check_cv_chunks
# This is a Django management command to check the CV chunking logic.
# To run this command, use:
# python manage.py check_cv_chunks

from django.core.management.base import BaseCommand
from core.models import ProfileDocument
from core.services.documents.chunk_service import ChunkService


class Command(BaseCommand):

    help = "Check CV section extraction and chunking"

    def handle(self, *args, **options):
        cv = (
            ProfileDocument.objects
            .filter(document_type="cv", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not cv:
            self.stdout.write(self.style.ERROR("No active CV found."))
            return

        full_text = cv.raw_text or ""

        self.stdout.write(self.style.SUCCESS(f"CV FOUND: {cv.title}"))
        self.stdout.write(f"RAW TEXT LENGTH: {len(full_text)}")

        # ---------------------------------------------------------
        # 1. Check extracted sections
        # ---------------------------------------------------------
        sections = ChunkService._extract_heading_sections(full_text)

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"SECTION COUNT: {len(sections)}")
        self.stdout.write("#" * 80)

        education_sections = []

        for i, sec in enumerate(sections[:30], 1):
            heading = sec.get("heading") or ""
            body = sec.get("body") or ""

            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"SECTION {i}")
            self.stdout.write(f"HEADING: {repr(heading)}")
            self.stdout.write("BODY PREVIEW:")
            self.stdout.write(body[:800])

            if "education" in heading.lower():
                education_sections.append(sec)

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("EDUCATION SECTION CHECK")
        self.stdout.write("#" * 80)

        if education_sections:
            self.stdout.write(self.style.SUCCESS(
                "✅ Education section was detected."))

            for sec in education_sections:
                body = sec.get("body") or ""
                self.stdout.write("\n--- EDUCATION SECTION BODY ---")
                self.stdout.write(body[:1500])

                if "master" in body.lower():
                    self.stdout.write(self.style.SUCCESS(
                        "✅ Master degree found inside Education section."))
                else:
                    self.stdout.write(self.style.ERROR(
                        "❌ Master degree NOT found inside Education section."))

                if "b.sc" in body.lower() or "bsc" in body.lower() or "bachelor" in body.lower():
                    self.stdout.write(self.style.SUCCESS(
                        "✅ Bachelor degree found inside Education section."))
                else:
                    self.stdout.write(self.style.ERROR(
                        "❌ Bachelor degree NOT found inside Education section."))
        else:
            self.stdout.write(self.style.ERROR(
                "❌ Education section was NOT detected by _extract_heading_sections()."))

        # ---------------------------------------------------------
        # 2. Check final resume chunks
        # ---------------------------------------------------------
        chunks = ChunkService.chunk_resume(full_text)

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        self.stdout.write("#" * 80)

        education_chunks = []

        for i, ch in enumerate(chunks[:30]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(ch[:1200])

            ch_lower = ch.lower()

            if (
                "education" in ch_lower
                or "master" in ch_lower
                or "b.sc" in ch_lower
                or "bsc" in ch_lower
                or "bachelor" in ch_lower
                or "ahlia university" in ch_lower
                or "applied science university" in ch_lower
            ):
                education_chunks.append((i, ch))

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("EDUCATION CHUNK CHECK")
        self.stdout.write("#" * 80)

        if not education_chunks:
            self.stdout.write(self.style.ERROR(
                "❌ No chunk contains Education / Master / Bachelor keywords."))
            self.stdout.write(self.style.ERROR(
                "This means your education content is not being chunked or is being lost."))
            return

        self.stdout.write(self.style.SUCCESS(
            f"✅ Found {len(education_chunks)} possible education-related chunk(s)."))

        for index, ch in education_chunks:
            ch_lower = ch.lower()

            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"POSSIBLE EDUCATION CHUNK INDEX: {index}")
            self.stdout.write("=" * 80)
            self.stdout.write(ch[:2000])

            has_education_header = "resume section: education" in ch_lower or "education" in ch_lower
            has_master = "master" in ch_lower
            has_bachelor = (
                "b.sc" in ch_lower
                or "bsc" in ch_lower
                or "bachelor" in ch_lower
            )
            has_work_experience = (
                "work experience" in ch_lower
                or "professional experience" in ch_lower
            )

            if has_education_header:
                self.stdout.write(self.style.SUCCESS(
                    "✅ Chunk contains Education heading."))
            else:
                self.stdout.write(self.style.WARNING(
                    "⚠️ Chunk contains degree keywords but no clear Education heading."))

            if has_master:
                self.stdout.write(self.style.SUCCESS(
                    "✅ Chunk contains Master degree."))
            else:
                self.stdout.write(self.style.ERROR(
                    "❌ Chunk does not contain Master degree."))

            if has_bachelor:
                self.stdout.write(self.style.SUCCESS(
                    "✅ Chunk contains Bachelor degree."))
            else:
                self.stdout.write(self.style.ERROR(
                    "❌ Chunk does not contain Bachelor degree."))

            if has_work_experience:
                self.stdout.write(self.style.WARNING(
                    "⚠️ Education chunk may be mixed with Work Experience."))
            else:
                self.stdout.write(self.style.SUCCESS(
                    "✅ Education chunk is not mixed with Work Experience."))

        # ---------------------------------------------------------
        # 3. Final verdict
        # ---------------------------------------------------------
        clean_education_chunks = []

        for index, ch in education_chunks:
            ch_lower = ch.lower()

            is_clean = (
                "education" in ch_lower
                and "master" in ch_lower
                and (
                    "b.sc" in ch_lower
                    or "bsc" in ch_lower
                    or "bachelor" in ch_lower
                )
                and "work experience" not in ch_lower
                and "professional experience" not in ch_lower
            )

            if is_clean:
                clean_education_chunks.append((index, ch))

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FINAL VERDICT")
        self.stdout.write("#" * 80)

        if clean_education_chunks:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: CV parser/chunker creates a clean Education chunk."))
            self.stdout.write(
                self.style.SUCCESS(
                    f"Best education chunk index: {clean_education_chunks[0][0]}"
                )
            )
        else:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: No clean Education chunk was created."))
            self.stdout.write(
                self.style.ERROR(
                    "Fix needed in ChunkService._extract_heading_sections() or ChunkService.chunk_resume()."
                )
            )

        # ---------------------------------------------------------
        # 4. Required section quality checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("REQUIRED SECTION QUALITY CHECKS")
        self.stdout.write("#" * 80)

        required_section_checks = {
            "Resume Header": [
                "+971",
                "s.fujo@hotmail.com",
                "Dubai",
            ],
            "Resume Section: SKILLS": [
                "Django REST Framework",
                "RAG",
                "PostgreSQL",
            ],
            "Resume Section: ABOUT ME": [
                "AI Team Lead",
                "Senior Data Scientist",
            ],
            "Resume Section: WORK EXPERIENCE": [
                "Nasser Artificial Intelligence",
                "Senior Al Data Scientist",
                "Part-Time Lecturer",
                "Integration Engine Specialist",
                "Research Assistant",
            ],
            "Resume Section: EDUCATION": [
                "Master",
                "Ahlia University",
                "B.Sc.",
                "Applied Science University",
            ],
            "Resume Section: PERSONAL DETAILS": [
                "Residency",
                "Golden Visa",
            ],
            "Resume Section: EXTRA-CURRICULAR ACTIVITIES": [
                "workshops",
                "University of Bahrain",
                "Tamkeen",
            ],
        }

        failed_checks = []

        for required_prefix, keywords in required_section_checks.items():
            matching_chunks = [
                chunk for chunk in chunks
                if chunk.startswith(required_prefix)
            ]

            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHECKING: {required_prefix}")

            if not matching_chunks:
                self.stdout.write(self.style.ERROR(
                    f"❌ Missing section chunk: {required_prefix}"))
                failed_checks.append(required_prefix)
                continue

            combined_text = "\n".join(matching_chunks)
            combined_lower = combined_text.lower()

            for keyword in keywords:
                if keyword.lower() in combined_lower:
                    self.stdout.write(self.style.SUCCESS(
                        f"✅ Found keyword: {keyword}"))
                else:
                    self.stdout.write(self.style.ERROR(
                        f"❌ Missing keyword: {keyword}"))
                    failed_checks.append(f"{required_prefix} -> {keyword}")

        # ---------------------------------------------------------
        # 5. Cross-section pollution checks
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("CROSS-SECTION POLLUTION CHECKS")
        self.stdout.write("#" * 80)

        pollution_rules = [
            {
                "section_prefix": "Resume Section: EDUCATION",
                "forbidden_keywords": [
                    "Part-Time Lecturer",
                    "Integration Engine Specialist",
                    "Research Assistant",
                    "Lead cross-functional teams",
                ],
            },
            {
                "section_prefix": "Resume Section: SKILLS",
                "forbidden_keywords": [
                    "Part-Time Lecturer",
                    "Integration Engine Specialist",
                    "Research Assistant",
                    "Master",
                    "B.Sc.",
                ],
            },
            {
                "section_prefix": "Resume Header",
                "forbidden_keywords": [
                    "Work Experience",
                    "Master",
                    "Part-Time Lecturer",
                    "Research Assistant",
                ],
            },

            {
                "section_prefix": "Resume Section: WORK EXPERIENCE",
                "forbidden_keywords": [
                    "Master's in Information Technology",
                    "B.Sc. in Computer Science",
                    "Ahlia University / Bahrain / 2021",
                    "PERSONAL DETAILS",
                    "EXTRA-CURRICULAR ACTIVITIES",
                ],
            },
        ]

        for rule in pollution_rules:
            section_prefix = rule["section_prefix"]
            forbidden_keywords = rule["forbidden_keywords"]

            matching_chunks = [
                chunk for chunk in chunks
                if chunk.startswith(section_prefix)
            ]

            combined_text = "\n".join(matching_chunks)
            combined_lower = combined_text.lower()

            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"POLLUTION CHECK: {section_prefix}")

            for keyword in forbidden_keywords:
                if keyword.lower() in combined_lower:
                    self.stdout.write(
                        self.style.ERROR(
                            f"❌ Pollution found in {section_prefix}: {keyword}"
                        )
                    )
                    failed_checks.append(
                        f"{section_prefix} polluted by {keyword}")
                else:
                    self.stdout.write(self.style.SUCCESS(
                        f"✅ Not polluted by: {keyword}"))

        # ---------------------------------------------------------
        # 6. Final required-section verdict
        # ---------------------------------------------------------
        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FINAL REQUIRED SECTION VERDICT")
        self.stdout.write("#" * 80)

        if failed_checks:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: Some required section checks failed."))

            for item in failed_checks:
                self.stdout.write(self.style.ERROR(f"- {item}"))
        else:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: Required CV sections are chunked correctly."))
