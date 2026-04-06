import json
import time
import requests
from datetime import datetime

API_URL = "http://127.0.0.1:8000/api/chat/ask/"
OUTPUT_FILE = f"test results/chatbot_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
SUMMARY_FILE = f"test results/chatbot_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
TIMEOUT = 120
DELAY_BETWEEN_REQUESTS = 0.5

QUESTIONS = [
    "What are Samah’s skills?",
    "What technologies does she use?",
    "What is her tech stack?",
    "What frameworks does Samah use?",
    "What tools does Samah use?",

    "What are Samah’s strongest technical areas?",
    "What is Samah strongest in?",
    "What are her core strengths?",

    "What does Samah do?",
    "What kind of work does Samah do?",
    "What is Samah’s background?",
    "Can Samah help with backend API development?",
    "Can Samah build an AI chatbot?",
    "What can Samah help with professionally?",
    "What can Samah do?",

    "What is Samah’s favorite programming language?",
    "Does Samah prefer Django or FastAPI?",
    "What backend framework does Samah prefer?",
    "What frontend stack does Samah prefer?",
    "Is Samah stronger in backend or frontend?",

    "How can I contact Samah?",
    "What is Samah’s email?",
    "What is Samah’s phone number?",
    "Does Samah have LinkedIn?",
    "What is Samah’s LinkedIn?",

    "Is Samah open to freelance work?",
    "Is Samah open to remote work?",
    "Which locations is Samah open to?",
    "What payment range is Samah looking for?",
    "What is Samah’s hourly rate?",
    "Is Samah open to full-time work?",
    "What is Samah’s availability?",
    "What are Samah’s compensation expectations?",

    "I have a new project need skills of power BI can you develop something like that?",
    "I have a new project using Tableau can she build something like that?",
    "I need someone with Snowflake experience can Samah handle this kind of project?",
    "Can she build this kind of data platform?",
    "Is she a fit for this project?",
    "Can Samah help with this type of dashboard project?",
    "Can Samah build this kind of analytics solution?",
    "Can Samah work on this kind of data-driven project?",

    "What is Samah’s GitHub?",
    "What is Samah’s home address?",
    "What is Samah’s university GPA?",
    "What is Samah’s exact salary?",
    "What is Samah’s favorite movie?"
]


def call_chatbot(question: str, session_id: str | None = None):
    payload = {
        "message": question
    }

    if session_id:
        payload["session_id"] = session_id

    response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def main():
    results = []
    summary = []

    session_id = None  # keep one session across all questions

    print(f"Starting chatbot test run with {len(QUESTIONS)} questions...\n")

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] Testing: {question}")
        print(f"Using session_id: {session_id}")

        try:
            api_response = call_chatbot(question, session_id=session_id)

            # save the first returned session_id, then keep reusing it
            if not session_id:
                session_id = api_response.get("session_id")

            results.append({
                "question": question,
                "status": "success",
                "response": api_response,
            })

            summary.append({
                "question": question,
                "status": "success",
                "session_id": api_response.get("session_id"),
                "debug_history_count": api_response.get("debug_history_count"),
                "retrieval_query": api_response.get("retrieval_query"),
                "verdict": api_response.get("verdict"),
                "answer": api_response.get("answer"),
                "applied_filters": api_response.get("applied_filters"),
                "used_sources": api_response.get("used_sources"),
            })

            print("Returned session_id:", api_response.get("session_id"))
            print("History count:", api_response.get("debug_history_count"))
            print("Retrieval query:", api_response.get("retrieval_query"))
            print("-" * 80)

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
                "session_id": session_id,
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
        "final_session_id": session_id,
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTest run completed.")
    print(f"Detailed results saved to: {OUTPUT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Final reused session_id: {session_id}")


if __name__ == "__main__":
    main()
