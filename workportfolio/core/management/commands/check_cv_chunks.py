# This is a Django management command to check the CV chunking logic.
# It loads the most recent CV document, extracts sections based on headings,    
# and then applies the chunking logic to see how the CV is being split.
# To run this command, use: python manage.py check_cv_chunks
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

        sections = ChunkService._extract_heading_sections(full_text)

        self.stdout.write(f"SECTION COUNT: {len(sections)}")
        for i, sec in enumerate(sections[:20], 1):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"SECTION {i}")
            self.stdout.write(f"HEADING: {repr(sec.get('heading'))}")
            self.stdout.write("BODY PREVIEW:")
            self.stdout.write((sec.get("body") or "")[:800])

        # Then chunk the full CV text and print the first few chunks
        chunks = ChunkService.chunk_resume(full_text)

        self.stdout.write(f"CHUNK COUNT: {len(chunks)}")
        for i, ch in enumerate(chunks[:20]):
            self.stdout.write("\n" + "=" * 80)
            self.stdout.write(f"CHUNK {i}")
            self.stdout.write(ch[:1200])