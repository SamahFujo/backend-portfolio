"""
How to run it

- Run with built-in default questions:

    python manage.py run_profile_qa_checks

- Save to a custom file:

    python manage.py run_profile_qa_checks --output qa_runs/check_01.json

- Use your own questions from a JSON file:

    python manage.py run_profile_qa_checks --questions-file core/questions/questions_smoke.json --output qa_runs/smoke_after_change.json

- After bigger ingestion/retrieval/prompt changes:
    python manage.py run_profile_qa_checks --questions-file core/questions/questions_regression.json --output qa_runs/regression_after_change.json
    
- Then compare:
    python manage.py compare_profile_qa_checks qa_runs/before.json qa_runs/regression_after_change.json --output qa_runs/diff.json  
    
"""


from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import List

from django.conf import settings
from django.core.management.base import BaseCommand

from core.services.chatbot.profile_qa_service import ProfileQAService
from core.services.chatbot.hybrid_query_rewriter import GeminiQueryRewriter


DEFAULT_QUESTIONS = [
    "What tools did she use in the Live Smart Electricity Dashboard?",
    "Which technologies were used in the Spend Analysis Dashboard?",
    "What certificates does she have?",
    "Which certificate relates to AI agents?",
    "What backend framework does Samah prefer most?",
    "Is Samah open to freelance work?",
    "When did she work at Nasser Centre?",
]


class Command(BaseCommand):
    help = "Run a batch of profile QA questions and save results to JSON."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output",
            type=str,
            default="profile_qa_results.json",
            help="Path to output JSON file.",
        )
        parser.add_argument(
            "--questions-file",
            type=str,
            help="Optional path to a JSON or TXT file containing questions.",
        )
        parser.add_argument(
            "--no-rewrite",
            action="store_true",
            help="Skip query rewriting and use the original question directly.",
        )

    def handle(self, *args, **options):
        output_path = Path(options["output"])
        questions_file = options.get("questions_file")
        no_rewrite = options.get("no_rewrite", False)

        questions = self._load_questions(questions_file)

        self.stdout.write(self.style.NOTICE(
            f"Running {len(questions)} question(s)..."))

        results = []
        for idx, question in enumerate(questions, start=1):
            self.stdout.write(f"[{idx}/{len(questions)}] {question}")

            retrieval_query = question
            rewrite_notes = None

            if not no_rewrite:
                try:
                    rewrite = GeminiQueryRewriter.rewrite_cached(
                        user_query=question,
                        history=[],
                    )
                    retrieval_query = rewrite.get(
                        "rewritten_query") or question
                    rewrite_notes = rewrite.get("notes")
                except Exception as exc:
                    retrieval_query = question
                    rewrite_notes = f"rewrite_error:{exc}"

            try:
                qa_result = ProfileQAService.answer_question(
                    question=question,
                    retrieval_query=retrieval_query,
                )

                result_entry = {
                    "question": question,
                    "retrieval_query": retrieval_query,
                    "rewrite_notes": rewrite_notes,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "result": qa_result,
                }
                results.append(result_entry)

                verdict = qa_result.get("verdict")
                provider = (qa_result.get("meta") or {}).get("provider_used")
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  Done -> verdict={verdict}, provider={provider}"
                    )
                )

            except Exception as exc:
                error_entry = {
                    "question": question,
                    "retrieval_query": retrieval_query,
                    "rewrite_notes": rewrite_notes,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "error": str(exc),
                }
                results.append(error_entry)
                self.stderr.write(self.style.ERROR(f"  FAILED -> {exc}"))

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "question_count": len(questions),
            "results": results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"Saved results to: {output_path.resolve()}"))

    def _load_questions(self, questions_file: str | None) -> list[str]:
        if not questions_file:
            return DEFAULT_QUESTIONS

        path = Path(questions_file)
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")

        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))

            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]

            if isinstance(data, dict):
                if isinstance(data.get("questions"), list):
                    return [str(x).strip() for x in data["questions"] if str(x).strip()]

                # Support grouped format like:
                # {"faq": [...], "projects": [...], ...}
                flattened = []
                for _, value in data.items():
                    if isinstance(value, list):
                        flattened.extend(str(x).strip() for x in value if str(x).strip())

                if flattened:
                    return flattened

            raise ValueError(
                "JSON questions file must be a list, {'questions': [...]}, or a grouped dict of question lists."
            )

        questions = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                questions.append(line)

        return questions or DEFAULT_QUESTIONS
