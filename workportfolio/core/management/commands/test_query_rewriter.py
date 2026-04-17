"""
# How to run it
    python manage.py test_query_rewriter --skip-checks --local-only
    
# How to run a specific query
    python manage.py test_query_rewriter --text "does samah know roberta and langchain?" --local-only
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from core.services.chatbot.hybrid_query_rewriter import GeminiQueryRewriter


class Command(BaseCommand):
    help = "Test GeminiQueryRewriter and print query before/after rewriting."

    def add_arguments(self, parser):
        parser.add_argument(
            "--text",
            type=str,
            help="Single query text to test.",
        )
        parser.add_argument(
            "--local-only",
            action="store_true",
            help="Test only the local rewrite layer without calling the full rewrite_cached flow.",
        )

    def _print_result(self, before: str, after: str, notes: str = "", meta=None):
        self.stdout.write("=" * 100)
        self.stdout.write(f"BEFORE : {before}")
        self.stdout.write(f"AFTER  : {after}")
        if notes:
            self.stdout.write(f"NOTES  : {notes}")
        if meta is not None:
            self.stdout.write(f"META   : {meta}")
        self.stdout.write("=" * 100)
        self.stdout.write("")

    def handle(self, *args, **options):
        text = options.get("text")
        local_only = options.get("local_only", False)

        # Optional history for follow-up/context-dependent tests
        history = [
            {
                "role": "user",
                "content": "I want to know about Samah's AI property chatbot project",
            },
            {
                "role": "assistant",
                "content": "Samah built an AI Property Search Chatbot using Django, Next.js, and LLMs.",
            },
        ]

        if text:
            if local_only:
                after = GeminiQueryRewriter._local_rewrite(text)
                self._print_result(
                    before=text,
                    after=after,
                    notes="local_only",
                    meta=None,
                )
            else:
                result = GeminiQueryRewriter.rewrite_cached(
                    text, history=history)
                self._print_result(
                    before=text,
                    after=result.get("rewritten_query", ""),
                    notes=result.get("notes", ""),
                    meta=result.get("meta"),
                )
            return

        # Default batch tests if no --text provided
        test_queries = [
            "does she have cirtifications",
            "what is her experiance in django",
            "does samah know roberta and langchain",
            "هل Samah عندها cirtifications",
            "ما هي experiance Samah في django",
            "do you no her backgroud in ai and llms",
            "what projcts did she do in postgress and react js",
            "with who can i discuss it",
            "هل لديها شهادات؟",
            "ما هي خبرتها في الذكاء الاصطناعي؟",
        ]

        for query in test_queries:
            if local_only:
                after = GeminiQueryRewriter._local_rewrite(query)
                self._print_result(
                    before=query,
                    after=after,
                    notes="local_only",
                    meta=None,
                )
            else:
                use_history = history if query.lower() in {
                    "with who can i discuss it",
                    "discuss it",
                    "with who",
                } else None

                result = GeminiQueryRewriter.rewrite_cached(
                    query, history=use_history)
                self._print_result(
                    before=query,
                    after=result.get("rewritten_query", ""),
                    notes=result.get("notes", ""),
                    meta=result.get("meta"),
                )
