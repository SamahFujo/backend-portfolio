"""How to run it

- Compare two runs:
    python manage.py compare_profile_qa_checks qa_runs/check_01.json qa_runs/check_02.json

- Save the comparison to JSON too:
    python manage.py compare_profile_qa_checks qa_runs/check_01.json qa_runs/check_02.json --output qa_runs/compare_01_02.json

- What it compares
    For each question, it checks changes in:
        retrieval query
        rewrite notes
        verdict
        answer
        provider used
        model used
        fallback used
        citations / used sources
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Compare two profile QA JSON result files and report differences."

    def add_arguments(self, parser):
        parser.add_argument(
            "old_file",
            type=str,
            help="Path to the older/baseline JSON result file.",
        )
        parser.add_argument(
            "new_file",
            type=str,
            help="Path to the newer JSON result file.",
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Optional path to save the comparison JSON report.",
        )

    def handle(self, *args, **options):
        old_path = Path(options["old_file"])
        new_path = Path(options["new_file"])
        output_path = options.get("output")

        old_payload = self._load_json(old_path)
        new_payload = self._load_json(new_path)

        old_results = self._index_results(old_payload.get("results", []))
        new_results = self._index_results(new_payload.get("results", []))

        all_questions = sorted(set(old_results.keys()) |
                               set(new_results.keys()))

        added: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        changed: List[Dict[str, Any]] = []
        unchanged: List[Dict[str, Any]] = []

        for question in all_questions:
            old_item = old_results.get(question)
            new_item = new_results.get(question)

            if old_item is None:
                added.append({
                    "question": question,
                    "new": new_item,
                })
                continue

            if new_item is None:
                removed.append({
                    "question": question,
                    "old": old_item,
                })
                continue

            diff = self._compare_result_items(old_item, new_item)
            if diff["changed"]:
                changed.append({
                    "question": question,
                    "differences": diff,
                    "old": old_item,
                    "new": new_item,
                })
            else:
                unchanged.append({
                    "question": question,
                    "summary": {
                        "verdict": self._safe_get(new_item, "result", "verdict"),
                        "provider_used": self._safe_get(new_item, "result", "meta", "provider_used"),
                        "model_used": self._safe_get(new_item, "result", "meta", "model_used"),
                    }
                })

        report = {
            "old_file": str(old_path),
            "new_file": str(new_path),
            "summary": {
                "old_question_count": len(old_results),
                "new_question_count": len(new_results),
                "added_count": len(added),
                "removed_count": len(removed),
                "changed_count": len(changed),
                "unchanged_count": len(unchanged),
            },
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
        }

        self._print_summary(report)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(
                report, ensure_ascii=False, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(
                f"Saved comparison report to: {out.resolve()}"))

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _index_results(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in results:
            question = str(item.get("question", "")).strip()
            if question:
                indexed[question] = item
        return indexed

    def _compare_result_items(self, old_item: Dict[str, Any], new_item: Dict[str, Any]) -> Dict[str, Any]:
        old_result = old_item.get("result", {})
        new_result = new_item.get("result", {})

        old_meta = old_result.get("meta", {})
        new_meta = new_result.get("meta", {})

        old_citations = self._normalize_citations(old_result.get(
            "used_sources") or old_result.get("citations") or [])
        new_citations = self._normalize_citations(new_result.get(
            "used_sources") or new_result.get("citations") or [])

        differences = {
            "changed": False,
            "fields": {}
        }

        self._record_field_diff(differences, "retrieval_query", old_item.get(
            "retrieval_query"), new_item.get("retrieval_query"))
        self._record_field_diff(differences, "rewrite_notes", old_item.get(
            "rewrite_notes"), new_item.get("rewrite_notes"))
        self._record_field_diff(differences, "verdict", old_result.get(
            "verdict"), new_result.get("verdict"))
        self._record_field_diff(differences, "answer", old_result.get(
            "answer"), new_result.get("answer"))
        self._record_field_diff(differences, "provider_used", old_meta.get(
            "provider_used"), new_meta.get("provider_used"))
        self._record_field_diff(differences, "model_used", old_meta.get(
            "model_used"), new_meta.get("model_used"))
        self._record_field_diff(differences, "fallback_used", old_meta.get(
            "fallback_used"), new_meta.get("fallback_used"))
        self._record_field_diff(differences, "citations",
                                old_citations, new_citations)

        return differences

    def _record_field_diff(self, differences: Dict[str, Any], field_name: str, old_value: Any, new_value: Any) -> None:
        if old_value != new_value:
            differences["changed"] = True
            differences["fields"][field_name] = {
                "old": old_value,
                "new": new_value,
            }

    def _normalize_citations(self, citations: List[Dict[str, Any]]) -> List[Tuple[Any, Any, Any]]:
        normalized = []
        for c in citations:
            normalized.append((
                c.get("document_title") or c.get("doc_title"),
                c.get("chunk_index"),
                c.get("chunk_id"),
            ))
        return normalized

    def _safe_get(self, data: Dict[str, Any], *keys):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _print_summary(self, report: Dict[str, Any]) -> None:
        summary = report["summary"]

        self.stdout.write("")
        self.stdout.write("=" * 80)
        self.stdout.write("PROFILE QA COMPARISON SUMMARY")
        self.stdout.write("=" * 80)
        self.stdout.write(
            f"Old questions     : {summary['old_question_count']}")
        self.stdout.write(
            f"New questions     : {summary['new_question_count']}")
        self.stdout.write(f"Added             : {summary['added_count']}")
        self.stdout.write(f"Removed           : {summary['removed_count']}")
        self.stdout.write(f"Changed           : {summary['changed_count']}")
        self.stdout.write(f"Unchanged         : {summary['unchanged_count']}")
        self.stdout.write("=" * 80)

        if report["changed"]:
            self.stdout.write("\nChanged questions:")
            for item in report["changed"]:
                self.stdout.write(f"- {item['question']}")
                for field_name, values in item["differences"]["fields"].items():
                    self.stdout.write(f"    {field_name}:")
                    self.stdout.write(f"      old = {values['old']}")
                    self.stdout.write(f"      new = {values['new']}")

        if report["added"]:
            self.stdout.write("\nAdded questions:")
            for item in report["added"]:
                self.stdout.write(f"- {item['question']}")

        if report["removed"]:
            self.stdout.write("\nRemoved questions:")
            for item in report["removed"]:
                self.stdout.write(f"- {item['question']}")
