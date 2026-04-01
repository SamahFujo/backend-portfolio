import json
import time
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8000/api/chat/ask/"   # change if needed
OUTPUT_FILE = f"test results/chatbot_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
SUMMARY_FILE = f"test results/chatbot_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
TIMEOUT = 120
DELAY_BETWEEN_REQUESTS = 0.5

QUESTIONS = [
    # "What is Samah’s favorite programming language?",
    # "Does Samah prefer Django or FastAPI?",
    # "What payment range is Samah looking for?",
    # "Is Samah open to freelance work?",
    # "Can Samah build an AI chatbot?",
    # "What can Samah help with professionally?",
    # "What does Samah do?",
    # "What are Samah’s strongest technical areas?",
    # "What impact has Samah created through her work?",
    # "What did Samah do before becoming an AI Team Lead?",
    # "How can I contact Samah?",
    # "What is Samah’s favorite movie?"
    # "What can Samah help with professionally?",
    # "What can Samah do?",
    # "Can Samah help with backend API development?"
    # "What can Samah help with professionally?",
    # "What can Samah do?",
    # "What does Samah do?",
    # "What are Samah’s strongest technical areas?",

    "How can I contact Samah?",
    "What is Samah’s email?",
    "What is Samah’s phone number?",
    "Does Samah have LinkedIn?"
]


def call_chatbot(question: str):
    payload = {
        "message": question
    }
    response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def main():
    results = []
    summary = []

    print(f"Starting chatbot test run with {len(QUESTIONS)} questions...\n")

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] Testing: {question}")

        try:
            api_response = call_chatbot(question)

            results.append({
                "question": question,
                "status": "success",
                "response": api_response,
            })

            summary.append({
                "question": question,
                "status": "success",
                "verdict": api_response.get("verdict"),
                "answer": api_response.get("answer"),
                "applied_filters": api_response.get("applied_filters"),
                "used_sources": api_response.get("used_sources"),
            })

        except Exception as e:
            error_msg = str(e)

            results.append({
                "question": question,
                "status": "failed",
                "error": error_msg,
            })

            summary.append({
                "question": question,
                "status": "failed",
                "verdict": None,
                "answer": None,
                "applied_filters": None,
                "used_sources": None,
                "error": error_msg,
            })

        time.sleep(DELAY_BETWEEN_REQUESTS)

    output = {
        "run_at": datetime.now().isoformat(),
        "api_url": API_URL,
        "total_questions": len(QUESTIONS),
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTest run completed.")
    print(f"Detailed results saved to: {OUTPUT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
