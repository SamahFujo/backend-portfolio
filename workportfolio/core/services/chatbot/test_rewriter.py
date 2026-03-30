from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import django

# ---------------------------------------------------------
# Add project root to Python path BEFORE importing app code
# test_rewriter.py is in: core/services/chatbot/
# parents[3] -> project root: workportfolio
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Django setup
# ---------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "workportfolio.settings")
django.setup()

# ---------------------------------------------------------
# Now imports from your Django project will work
# ---------------------------------------------------------
from core.services.chatbot.gemini_query_rewriter import GeminiQueryRewriter


def load_queries(file_path: str = "test_queries.json") -> list[str]:
    query_file = PROJECT_ROOT / file_path
    with open(query_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_test(queries: list[str], save_output: bool = True) -> None:
    GeminiQueryRewriter.rewrite_cached.cache_clear()

    results = []

    print("=" * 120)
    print("QUERY REWRITER TEST")
    print("=" * 120)

    for i, query in enumerate(queries, start=1):
        print(f"\n[{i}] ORIGINAL QUERY")
        print(repr(query))

        try:
            result = GeminiQueryRewriter.rewrite_cached(query)

            print("\nREWRITER RESULT")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            results.append(
                {
                    "index": i,
                    "original_query": query,
                    "status": "ok",
                    "result": result,
                }
            )

        except Exception as e:
            error_msg = str(e)

            print("\nERROR")
            print(error_msg)

            results.append(
                {
                    "index": i,
                    "original_query": query,
                    "status": "error",
                    "error": error_msg,
                }
            )

        print("-" * 120)

    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROJECT_ROOT / f"rewriter_results_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    queries = load_queries("test_queries.json")
    run_test(queries, save_output=True)