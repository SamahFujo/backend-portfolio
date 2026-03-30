from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import django

# ---------------------------------------------------------
# Add project root to Python path BEFORE importing app code
# test_grounded_answerer.py is in: core/services/chatbot/
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
# Project imports
# ---------------------------------------------------------
from core.models import DocumentChunk
from core.services.chatbot.gemini_grounded_answerer import GeminiGroundedAnswerer


def load_test_cases(file_path: str = "test_grounded_cases.json") -> list[dict]:
    case_file = PROJECT_ROOT / file_path
    with open(case_file, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_chunks_by_ids(chunk_ids: list[int]) -> list[DocumentChunk]:
    """
    Loads DocumentChunk rows by DB ids while preserving input order.
    """
    if not chunk_ids:
        return []

    chunk_map = {
        c.id: c
        for c in DocumentChunk.objects.filter(id__in=chunk_ids).select_related("document")
    }

    ordered_chunks = []
    for chunk_id in chunk_ids:
        chunk = chunk_map.get(chunk_id)
        if chunk is not None:
            ordered_chunks.append(chunk)

    return ordered_chunks


def run_test_cases(cases: list[dict], save_output: bool = True) -> None:
    results = []

    print("=" * 120)
    print("GROUNDED ANSWERER TEST")
    print("=" * 120)

    for i, case in enumerate(cases, start=1):
        case_name = case.get("name", f"case_{i}")
        question = case.get("question", "")
        chunk_ids = case.get("chunk_ids", [])

        print(f"\n[{i}] CASE NAME")
        print(case_name)

        print("\nQUESTION")
        print(question)

        print("\nCHUNK IDS")
        print(chunk_ids)

        try:
            evidence_chunks = fetch_chunks_by_ids(chunk_ids)
            result = GeminiGroundedAnswerer.answer(question, evidence_chunks)

            print("\nANSWERER RESULT")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            results.append(
                {
                    "index": i,
                    "name": case_name,
                    "question": question,
                    "chunk_ids": chunk_ids,
                    "resolved_chunk_count": len(evidence_chunks),
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
                    "name": case_name,
                    "question": question,
                    "chunk_ids": chunk_ids,
                    "status": "error",
                    "error": error_msg,
                }
            )

        print("-" * 120)

    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROJECT_ROOT / f"grounded_answerer_results_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    cases = load_test_cases("test_grounded_cases.json")
    run_test_cases(cases, save_output=True)