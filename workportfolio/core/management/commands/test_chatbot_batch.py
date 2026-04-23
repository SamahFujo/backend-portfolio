# python manage.py test_chatbot_batch --questions-file core/questions/questions.json --output qa_runs/final_chatbot_eval.json
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from django.conf import settings
from django.core.management.base import BaseCommand
from django.test import Client


DEFAULT_QUESTIONS: List[str] = [
    "hi",
    "who are you",
    "what can you do",
    "what projects did she work on",
    "which projects used django",
    "can this chatbot support arabic",
    "can she build something similar",
    "how many years of experience does she have",
    "how can i contact her",
    "what was my first question",
    "what was your last answer",
    "bye",
]


class Command(BaseCommand):
    help = "Send a batch of questions to the chatbot API and save results as JSON."

    def add_arguments(self, parser):
        parser.add_argument(
            "--endpoint",
            type=str,
            default="/api/chat/ask/",
            help="Chatbot endpoint path. Default: /api/chat/ask/",
        )
        parser.add_argument(
            "--questions-file",
            type=str,
            default="",
            help="Optional JSON or TXT file containing questions.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="runs/chatbot_batch_results.json",
            help="Output JSON file path.",
        )
        parser.add_argument(
            "--pause",
            type=float,
            default=0.0,
            help="Pause in seconds between requests.",
        )
        parser.add_argument(
            "--fresh-session-each-question",
            action="store_true",
            help="If set, each question will use a fresh session.",
        )

    def handle(self, *args, **options):
        endpoint: str = options["endpoint"]
        questions_file: str = options["questions_file"]
        output_path = Path(options["output"])
        pause: float = float(options["pause"])
        fresh_session_each_question: bool = bool(
            options["fresh_session_each_question"])

        questions = self._load_questions(questions_file)

        if not questions:
            self.stdout.write(self.style.ERROR("No questions found to test."))
            return

        client = Client()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        session_id = None
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")

        results: List[Dict[str, Any]] = []

        self.stdout.write(self.style.SUCCESS(f"Testing endpoint: {endpoint}"))
        self.stdout.write(f"Total questions: {len(questions)}")

        for idx, question in enumerate(questions, start=1):
            payload: Dict[str, Any] = {"message": question}

            if not fresh_session_each_question and session_id:
                payload["session_id"] = session_id

            self.stdout.write("")
            self.stdout.write(f"[{idx}/{len(questions)}] USER: {question}")

            try:
                response = client.post(
                    endpoint,
                    data=json.dumps(payload),
                    content_type="application/json",
                )
            except Exception as exc:
                result = {
                    "index": idx,
                    "question": question,
                    "request_payload": payload,
                    "http_status": None,
                    "ok": False,
                    "error": str(exc),
                    "response_json": None,
                }
                results.append(result)
                self.stdout.write(self.style.ERROR(f"Request failed: {exc}"))
                continue

            response_json = self._safe_json(response)

            if not fresh_session_each_question and isinstance(response_json, dict):
                session_id = response_json.get("session_id") or session_id

            answer_preview = ""
            if isinstance(response_json, dict):
                answer_preview = str(response_json.get("answer", ""))[:250]

            result = {
                "index": idx,
                "question": question,
                "request_payload": payload,
                "http_status": response.status_code,
                "ok": 200 <= response.status_code < 300,
                "response_json": response_json,
            }
            results.append(result)

            self.stdout.write(f"HTTP {response.status_code}")
            if answer_preview:
                self.stdout.write(f"ANSWER: {answer_preview}")

            if pause > 0:
                time.sleep(pause)

        summary = self._build_summary(
            endpoint=endpoint,
            started_at=started_at,
            questions=questions,
            results=results,
            final_session_id=session_id,
        )

        output = {
            "meta": summary,
            "results": results,
        }

        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"Saved results to: {output_path}"))
        self.stdout.write(
            self.style.SUCCESS(
                f"Passed: {summary['success_count']} | Failed: {summary['failure_count']}"
            )
        )

    def _load_questions(self, questions_file: str) -> List[str]:
        if not questions_file:
            return DEFAULT_QUESTIONS

        path = Path(questions_file)
        if not path.exists():
            self.stdout.write(
                self.style.WARNING(
                    f"Questions file not found: {questions_file}. Using defaults."
                )
            )
            return DEFAULT_QUESTIONS

        suffix = path.suffix.lower()

        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
            if isinstance(data, dict) and isinstance(data.get("questions"), list):
                return [str(item).strip() for item in data["questions"] if str(item).strip()]
            raise ValueError(
                "JSON questions file must be a list or {'questions': [...]}")

        if suffix == ".txt":
            return [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        raise ValueError("Unsupported questions file type. Use .json or .txt")

    def _safe_json(self, response) -> Any:
        try:
            return json.loads(response.content.decode("utf-8"))
        except Exception:
            return {
                "_raw_text": response.content.decode("utf-8", errors="replace")
            }

    def _build_summary(
        self,
        endpoint: str,
        started_at: str,
        questions: List[str],
        results: List[Dict[str, Any]],
        final_session_id: str | None,
    ) -> Dict[str, Any]:
        success_count = sum(1 for item in results if item.get("ok"))
        failure_count = len(results) - success_count

        route_counts: Dict[str, int] = {}
        answer_source_counts: Dict[str, int] = {}
        http_status_counts: Dict[str, int] = {}

        for item in results:
            http_status = item.get("http_status")
            http_status_counts[str(http_status)] = http_status_counts.get(
                str(http_status), 0) + 1

            response_json = item.get("response_json")
            if isinstance(response_json, dict):
                route = response_json.get(
                    "question_route") or response_json.get("mode")
                if route:
                    route_counts[str(route)] = route_counts.get(
                        str(route), 0) + 1

                answer_source = response_json.get("answer_source")
                if answer_source:
                    answer_source_counts[str(answer_source)] = (
                        answer_source_counts.get(str(answer_source), 0) + 1
                    )

        return {
            "started_at": started_at,
            "django_debug": bool(getattr(settings, "DEBUG", False)),
            "endpoint": endpoint,
            "question_count": len(questions),
            "success_count": success_count,
            "failure_count": failure_count,
            "final_session_id": final_session_id,
            "http_status_counts": http_status_counts,
            "route_counts": route_counts,
            "answer_source_counts": answer_source_counts,
        }
