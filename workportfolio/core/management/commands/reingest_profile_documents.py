## How to run it ##
## Re-ingest all profile documents : python manage.py reingest_profile_documents
## Re-ingest only project documents : python manage.py reingest_profile_documents --document-type projects

## to test output ###
# from core.models import ProfileDocument

# doc = ProfileDocument.objects.get(title="Certificates")
# for c in doc.chunks.order_by("chunk_index")[:10]:
#     print("\n" + "=" * 80)
#     print("chunk_index:", c.chunk_index)
#     print(c.content[:1500])
    
    
from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import ProfileDocument
from core.services.documents.ingestion_service import IngestionService


class Command(BaseCommand):
    help = "Re-ingest profile documents to refresh raw_text, document_type, chunks, and embeddings."

    def add_arguments(self, parser):
        parser.add_argument(
            "--title",
            type=str,
            help="Re-ingest only documents whose title contains this value.",
        )
        parser.add_argument(
            "--document-type",
            type=str,
            help="Re-ingest only documents with this current document_type.",
        )
        parser.add_argument(
            "--status",
            type=str,
            help="Re-ingest only documents with this current status.",
        )
        parser.add_argument(
            "--only-failed",
            action="store_true",
            help="Re-ingest only documents whose status is failed.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Limit number of documents to process.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show which documents would be processed without actually re-ingesting them.",
        )

    def handle(self, *args, **options):
        qs = ProfileDocument.objects.all().order_by("title", "created_at")

        if options.get("title"):
            qs = qs.filter(title__icontains=options["title"])

        if options.get("document_type"):
            qs = qs.filter(document_type=options["document_type"])

        if options.get("status"):
            qs = qs.filter(status=options["status"])

        if options.get("only_failed"):
            qs = qs.filter(status="failed")

        if options.get("limit"):
            qs = qs[: options["limit"]]

        docs = list(qs)

        if not docs:
            self.stdout.write(self.style.WARNING(
                "No matching documents found."))
            return

        self.stdout.write(self.style.NOTICE(
            f"Matched {len(docs)} document(s)."))

        if options.get("dry_run"):
            for doc in docs:
                self.stdout.write(
                    f"[DRY RUN] {doc.id} | {doc.title} | type={doc.document_type} | status={doc.status}"
                )
            return

        success_count = 0
        failed_count = 0

        for index, doc in enumerate(docs, start=1):
            self.stdout.write(
                f"\n[{index}/{len(docs)}] Re-ingesting: "
                f"{doc.title} | current_type={doc.document_type} | current_status={doc.status}"
            )

            try:
                IngestionService.process_document(doc)
                doc.refresh_from_db()

                success_count += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Done: {doc.title} | "
                        f"status={doc.status} | "
                        f"document_type={doc.document_type} | "
                        f"chunk_count={getattr(doc, 'chunk_count', 'n/a')}"
                    )
                )

            except Exception as exc:
                failed_count += 1
                self.stderr.write(
                    self.style.ERROR(
                        f"FAILED: {doc.title} | error={exc}"
                    )
                )

        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(
            self.style.SUCCESS(f"Successful: {success_count}")
        )
        self.stdout.write(
            self.style.ERROR(f"Failed: {failed_count}")
        )
        self.stdout.write("=" * 80)
