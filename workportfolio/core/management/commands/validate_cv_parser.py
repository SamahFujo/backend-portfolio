from django.core.management.base import BaseCommand
from core.models import ProfileDocument


class Command(BaseCommand):
    help = "Validate whether the latest active CV raw_text contains key CV sections and facts."

    def handle(self, *args, **options):
        cv = (
            ProfileDocument.objects
            .filter(document_type="cv", is_active=True)
            .order_by("-created_at")
            .first()
        )

        if not cv:
            self.stdout.write(self.style.ERROR("❌ No active CV found."))
            return

        full_text = cv.raw_text or ""
        lower_text = full_text.lower()

        self.stdout.write(self.style.SUCCESS(f"✅ CV FOUND: {cv.title}"))
        self.stdout.write(f"RAW TEXT LENGTH: {len(full_text)}")

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("RAW TEXT PREVIEW")
        self.stdout.write("#" * 80)
        self.stdout.write(full_text[:3000])

        required_keywords = {
            "Header / Contact": [
                "samah",
                "dubai",
                "s.fujo@hotmail.com",
                "+971",
            ],
            "About / Summary": [
                "about me",
                "ai team lead",
                "senior data scientist",
            ],
            "Skills": [
                "skills",
                "django",
                "react",
                "rag",
                "postgresql",
            ],
            "Work Experience": [
                "work experience",
                "nasser artificial intelligence",
                "part-time lecturer",
                "integration engine specialist",
                "research assistant",
            ],
            "Education": [
                "education",
                "master",
                "ahlia",
                "b.sc",
                "applied science university",
            ],
            "Languages": [
                "languages",
                "arabic",
                "english",
            ],
        }

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("KEYWORD VALIDATION")
        self.stdout.write("#" * 80)

        missing_by_group = {}

        for group_name, keywords in required_keywords.items():
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(group_name)
            self.stdout.write("=" * 80)

            missing = []

            for keyword in keywords:
                exists = keyword.lower() in lower_text

                if exists:
                    self.stdout.write(
                        self.style.SUCCESS(f"✅ Found: {keyword}"))
                else:
                    self.stdout.write(self.style.ERROR(
                        f"❌ Missing: {keyword}"))
                    missing.append(keyword)

            if missing:
                missing_by_group[group_name] = missing

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("SECTION POSITION CHECK")
        self.stdout.write("#" * 80)

        section_markers = [
            "about me",
            "skills",
            "work experience",
            "education",
            "languages",
            "certifications",
            "projects",
            "research",
            "additional information",
        ]

        positions = []

        for marker in section_markers:
            pos = lower_text.find(marker)
            positions.append((marker, pos))

            if pos == -1:
                self.stdout.write(self.style.WARNING(
                    f"⚠️ Not found: {marker}"))
            else:
                self.stdout.write(self.style.SUCCESS(
                    f"✅ {marker}: position {pos}"))

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("EDUCATION CONTEXT CHECK")
        self.stdout.write("#" * 80)

        education_pos = lower_text.find("education")

        if education_pos == -1:
            self.stdout.write(self.style.ERROR(
                "❌ Cannot show Education context because 'education' was not found."))
        else:
            start = max(0, education_pos - 500)
            end = min(len(full_text), education_pos + 1500)

            self.stdout.write(self.style.SUCCESS(
                "✅ Education keyword found. Context below:"))
            self.stdout.write("\n" + "-" * 80)
            self.stdout.write(full_text[start:end])
            self.stdout.write("-" * 80)

        self.stdout.write("\n" + "#" * 80)
        self.stdout.write("FINAL PARSER VERDICT")
        self.stdout.write("#" * 80)

        if not missing_by_group:
            self.stdout.write(self.style.SUCCESS(
                "✅ PASS: Parser raw_text contains the key CV information."))
            self.stdout.write(
                self.style.SUCCESS(
                    "Next step: implement section-aware chunking safely."
                )
            )
        else:
            self.stdout.write(self.style.ERROR(
                "❌ FAIL: Parser raw_text is missing important CV information."))

            for group_name, missing in missing_by_group.items():
                self.stdout.write(
                    self.style.ERROR(
                        f"{group_name} missing keywords: {', '.join(missing)}"
                    )
                )

            self.stdout.write(
                self.style.ERROR(
                    "Fix parser extraction or re-upload a cleaner CV before changing chunking."
                )
            )
