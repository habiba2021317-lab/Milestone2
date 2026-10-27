import json
import re
from pathlib import Path

import requests


MODEL = "qwen2.5:3b"
API_URL = "http://localhost:11434/api/generate"

INPUT_FILE = Path("calibration_cases.json")
OUTPUT_FILE = Path("judge_calibration_results.json")


DEFAULT_CASES = [
    {
        "id": 1,
        "question": "What does RAM do?",
        "correct_answer": "RAM temporarily stores data used by running programs.",
        "candidate_explanation": (
            "RAM is temporary working memory for running programs "
            "and normally loses its contents when power is removed."
        ),
        "expected_score": 5
    },
    {
        "id": 2,
        "question": "What does DNS do?",
        "correct_answer": "DNS resolves domain names to IP addresses and other records.",
        "candidate_explanation": (
            "DNS translates a domain name into its corresponding IP address."
        ),
        "expected_score": 4
    },
    {
        "id": 3,
        "question": "What is a stack?",
        "correct_answer": "A stack normally follows Last In, First Out.",
        "candidate_explanation": (
            "A stack stores items and supports adding and removing them."
        ),
        "expected_score": 3
    },
    {
        "id": 4,
        "question": "What does a firewall do?",
        "correct_answer": "A firewall filters network traffic according to security rules.",
        "candidate_explanation": (
            "A firewall mainly increases CPU speed and repairs damaged files."
        ),
        "expected_score": 2
    },
    {
        "id": 5,
        "question": "What is encryption for?",
        "correct_answer": "Encryption helps protect confidentiality.",
        "candidate_explanation": (
            "Encryption makes every file publicly readable without a key."
        ),
        "expected_score": 1
    },
    {
        "id": 6,
        "question": "What does a primary key do?",
        "correct_answer": "A primary key uniquely identifies a database row.",
        "candidate_explanation": (
            "A primary key uniquely identifies each row in its table."
        ),
        "expected_score": 5
    },
    {
        "id": 7,
        "question": "What is overfitting?",
        "correct_answer": "A model fits training data well but generalizes poorly.",
        "candidate_explanation": (
            "Overfitting happens when a model learns training-specific patterns "
            "and performs worse on unseen examples."
        ),
        "expected_score": 4
    },
    {
        "id": 8,
        "question": "What is a loop?",
        "correct_answer": "A loop repeats a block of instructions.",
        "candidate_explanation": (
            "A loop is a programming structure that controls instruction execution."
        ),
        "expected_score": 3
    },
    {
        "id": 9,
        "question": "What does TCP provide?",
        "correct_answer": "TCP provides reliable, ordered byte-stream delivery.",
        "candidate_explanation": (
            "TCP guarantees delivery even when the receiver is offline, "
            "so packets can never be lost."
        ),
        "expected_score": 2
    },
    {
        "id": 10,
        "question": "How should passwords normally be stored?",
        "correct_answer": "Use a suitable password-hashing algorithm with a unique salt.",
        "candidate_explanation": (
            "Passwords should be stored as plain text for easy recovery."
        ),
        "expected_score": 1
    }
]


def load_cases():
    """Create the calibration dataset if it does not exist."""

    if not INPUT_FILE.exists():
        with INPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                DEFAULT_CASES,
                file,
                indent=2,
                ensure_ascii=False
            )
        print("Created calibration_cases.json")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def call_ollama(prompt):
    """Send a prompt to the local model."""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 100
        }
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=300
    )
    response.raise_for_status()

    return response.json()["response"].strip()


def evaluate_case(case):
    """Ask the judge to score one explanation."""

    prompt = f"""
You are evaluating a computer science explanation.

Rubric:
5 = Correct, clear, and complete.
4 = Correct with a minor omission.
3 = Relevant but incomplete or vague.
2 = Contains a significant technical error.
1 = Incorrect or substantially irrelevant.

Do not penalize a short answer when it correctly explains a simple concept.
Judge technical correctness and completeness.

Question:
{case["question"]}

Reference answer:
{case["correct_answer"]}

Candidate explanation:
{case["candidate_explanation"]}

Return only a JSON object in this format:
{{"score": 5, "reason": "Brief explanation"}}

The score must be an integer from 1 to 5.
""".strip()

    output = call_ollama(prompt)

    match = re.search(r"\{.*?\}", output, flags=re.DOTALL)

    if not match:
        raise ValueError("Judge did not return valid JSON.")

    data = json.loads(match.group(0))
    score = data.get("score")

    if type(score) is not int or score not in range(1, 6):
        raise ValueError("Judge returned an invalid score.")

    return score, str(data.get("reason", ""))


def main():
    try:
        cases = load_cases()
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not load calibration cases: {error}")
        return

    results = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] Calibration case {case['id']}")

        result = {
            "id": case["id"],
            "expected_score": case["expected_score"],
            "judge_score": None,
            "reason": "",
            "error": None
        }

        try:
            score, reason = evaluate_case(case)
            result["judge_score"] = score
            result["reason"] = reason

            print(
                f"  Reference: {case['expected_score']}, "
                f"Judge: {score}"
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError
        ) as error:
            result["error"] = str(error)
            print(f"  ERROR: {error}")

        results.append(result)

        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                results,
                file,
                indent=2,
                ensure_ascii=False
            )

    valid = [
        item for item in results
        if item["judge_score"] is not None
    ]

    if not valid:
        print("No valid judge scores were produced.")
        return

    exact = sum(
        item["judge_score"] == item["expected_score"]
        for item in valid
    )

    within_one = sum(
        abs(item["judge_score"] - item["expected_score"]) <= 1
        for item in valid
    )

    mae = sum(
        abs(item["judge_score"] - item["expected_score"])
        for item in valid
    ) / len(valid)

    print("\n========== Judge Calibration Summary ==========")
    print(f"Total cases: {len(cases)}")
    print(f"Valid scores: {len(valid)}")
    print(f"Exact agreement: {exact}/{len(valid)}")
    print(f"Agreement within one point: {within_one}/{len(valid)}")
    print(f"Mean absolute error: {mae:.2f}")
    print(f"Results saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()