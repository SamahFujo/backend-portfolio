"""
Refresh ResearchGate reads and citations for ResearchItem records.

Scenario:
- Try ResearchGate only.
- If ResearchGate blocks the request, keep old values.
- Log every attempt.
- Do not use OpenAlex or Semantic Scholar because their citation counts
  may not match ResearchGate.

Usage:
    python manage.py refresh_research_stats

Dry run:
    python manage.py refresh_research_stats --dry-run

Limit:
    python manage.py refresh_research_stats --limit 1
"""

import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from django.core.management.base import BaseCommand

from core.models import ResearchItem, ResearchStatsRefreshLog


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def normalize_number(value: str) -> Optional[str]:
    if not value:
        return None

    raw_value = value.strip().replace(",", "").replace(" ", "")

    match = re.match(r"^(\d+(?:\.\d+)?)([kKmM]?)$", raw_value)
    if not match:
        return None

    number = float(match.group(1))
    suffix = match.group(2).lower()

    if suffix == "k":
        number *= 1000
    elif suffix == "m":
        number *= 1_000_000

    return str(int(number))


def extract_stat_from_text(text: str, label: str) -> Optional[str]:
    compact_number = r"(\d+(?:[,.]\d+)?(?:\.\d+)?\s*[kKmM]?)"

    patterns = [
        rf"{compact_number}\s+{label}",
        rf"{label}\s+{compact_number}",
        rf"{label.lower()}\s+{compact_number}",
        rf"{compact_number}\s+{label.lower()}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_number(match.group(1))

    return None


def fetch_researchgate_stats(url: str, timeout: int = 20) -> dict:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    response = requests.get(url, headers=headers, timeout=timeout)

    if response.status_code == 403:
        return {
            "blocked": True,
            "reads": None,
            "citations": None,
            "message": (
                "ResearchGate blocked the automated request with 403 Forbidden. "
                "Old database values were kept."
            ),
        }

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    page_text = soup.get_text(" ", strip=True)
    page_text = re.sub(r"\s+", " ", page_text)

    reads = extract_stat_from_text(page_text, "Reads")
    citations = extract_stat_from_text(page_text, "Citations")

    return {
        "blocked": False,
        "reads": reads,
        "citations": citations,
        "message": "ResearchGate page fetched successfully.",
    }


def create_refresh_log(
    item: ResearchItem,
    status: str,
    old_reads: str,
    new_reads: Optional[str],
    old_citations: str,
    new_citations: Optional[str],
    message: str,
) -> None:
    ResearchStatsRefreshLog.objects.create(
        research_item=item,
        status=status,
        old_reads=old_reads or "",
        new_reads=new_reads or "",
        old_citations=old_citations or "",
        new_citations=new_citations or "",
        reads_fetched=new_reads is not None,
        citations_fetched=new_citations is not None,
        message=message,
        source_url=item.primary_action_href or "",
    )


class Command(BaseCommand):
    help = "Refresh ResearchGate reads and citations only."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch stats but do not save updates.",
        )

        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit number of research items.",
        )

        parser.add_argument(
            "--sleep",
            type=float,
            default=2.0,
            help="Seconds to wait between requests.",
        )

        parser.add_argument(
            "--timeout",
            type=int,
            default=20,
            help="HTTP timeout in seconds.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        limit = options["limit"]
        sleep_seconds = options["sleep"]
        timeout = options["timeout"]

        queryset = ResearchItem.objects.filter(
            is_active=True,
            primary_action_href__icontains="researchgate.net",
        ).order_by("sort_order", "created_at")

        if limit:
            queryset = queryset[:limit]

        total = queryset.count() if not limit else len(queryset)

        updated_count = 0
        no_change_count = 0
        failed_count = 0
        skipped_count = 0

        self.stdout.write(
            self.style.WARNING(
                f"Refreshing ResearchGate stats for {total} research item(s)."
            )
        )

        for item in queryset:
            self.stdout.write("")
            self.stdout.write(f"Checking: {item.title}")

            old_reads = item.reads or ""
            old_citations = item.citations or ""

            if not item.primary_action_href:
                skipped_count += 1

                create_refresh_log(
                    item=item,
                    status="skipped",
                    old_reads=old_reads,
                    new_reads=None,
                    old_citations=old_citations,
                    new_citations=None,
                    message="Skipped because no ResearchGate URL was found.",
                )

                self.stdout.write(self.style.WARNING("Skipped: No URL."))
                continue

            try:
                stats = fetch_researchgate_stats(
                    item.primary_action_href,
                    timeout=timeout,
                )

                new_reads = stats.get("reads")
                new_citations = stats.get("citations")

                reads_fetched = new_reads is not None
                citations_fetched = new_citations is not None

                if not reads_fetched and not citations_fetched:
                    failed_count += 1

                    create_refresh_log(
                        item=item,
                        status="failed",
                        old_reads=old_reads,
                        new_reads=None,
                        old_citations=old_citations,
                        new_citations=None,
                        message=stats.get(
                            "message",
                            "Failed to fetch reads and citations from ResearchGate.",
                        ),
                    )

                    self.stdout.write(
                        self.style.ERROR(
                            "Failed: ResearchGate values were not fetched. Old values kept."
                        )
                    )

                    time.sleep(sleep_seconds)
                    continue

                should_update = False

                if reads_fetched and new_reads != old_reads:
                    item.reads = new_reads
                    should_update = True

                if citations_fetched and new_citations != old_citations:
                    item.citations = new_citations
                    should_update = True

                if should_update:
                    status = "success"
                    updated_count += 1
                    message = "ResearchGate reads/citations fetched and database values updated."
                else:
                    status = "no_change"
                    no_change_count += 1
                    message = "ResearchGate reads/citations fetched successfully, but values did not change."

                create_refresh_log(
                    item=item,
                    status=status,
                    old_reads=old_reads,
                    new_reads=new_reads,
                    old_citations=old_citations,
                    new_citations=new_citations,
                    message=message,
                )

                self.stdout.write(
                    self.style.SUCCESS(
                        f"{message} Reads: {old_reads} → {new_reads or old_reads}, "
                        f"Citations: {old_citations} → {new_citations or old_citations}"
                    )
                )

                if should_update and not dry_run:
                    item.save(update_fields=[
                              "reads", "citations", "updated_at"])

                if should_update and dry_run:
                    self.stdout.write(
                        self.style.WARNING(
                            "Dry run enabled: values not saved.")
                    )

            except requests.HTTPError as error:
                failed_count += 1

                status_code = error.response.status_code if error.response else None

                if status_code == 403:
                    message = (
                        "ResearchGate blocked the automated request with 403 Forbidden. "
                        "Old database values were kept."
                    )
                else:
                    message = f"HTTP error while fetching ResearchGate page: {error}"

                create_refresh_log(
                    item=item,
                    status="failed",
                    old_reads=old_reads,
                    new_reads=None,
                    old_citations=old_citations,
                    new_citations=None,
                    message=message,
                )

                self.stdout.write(self.style.ERROR(message))

            except requests.RequestException as error:
                failed_count += 1

                message = f"Network error while fetching ResearchGate page: {error}"

                create_refresh_log(
                    item=item,
                    status="failed",
                    old_reads=old_reads,
                    new_reads=None,
                    old_citations=old_citations,
                    new_citations=None,
                    message=message,
                )

                self.stdout.write(self.style.ERROR(message))

            except Exception as error:
                failed_count += 1

                message = f"Unexpected error: {error}"

                create_refresh_log(
                    item=item,
                    status="failed",
                    old_reads=old_reads,
                    new_reads=None,
                    old_citations=old_citations,
                    new_citations=None,
                    message=message,
                )

                self.stdout.write(self.style.ERROR(message))

            time.sleep(sleep_seconds)

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Refresh completed. Updated={updated_count}, "
                f"NoChange={no_change_count}, Skipped={skipped_count}, Failed={failed_count}."
            )
        )
