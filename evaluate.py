import argparse
import json
import re
import time
from pathlib import Path

import requests

from validators import validate_answer


MODEL = "qwen2.5:3b"
API_URL = "http://localhost:11434/api/generate"

TEST_FILE = Path("golden_tests.json")
RESULT_FILE = Path("evaluation_results.json")


def call_ollama(prompt, num_predict=120):
    """Send a prompt to the local Ollama model."""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": num_predict
        }
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()
    return data["response"].strip()


def build_question_prompt(test_case):
    """Prepare the question and instructions for the model."""

    return f"""
You are answering a basic computer science multiple-choice test.

Question:
{test_case["question"]}

Instructions:
1. Choose exactly one option: A, B, C, or D.
2. Your first line must be exactly: Answer: X
3. Your second line must begin with: Explanation:
4. Give a short explanation in one sentence.
5. Do not write anything before the first line.

Required format:
Answer: B
Explanation: A short reason for the choice.
""".strip()


def extract_explanation(raw_output):
    """Extract the explanation after the first line."""

    lines = raw_output.splitlines()

    if len(lines) < 2:
        return ""

    explanation = "\n".join(lines[1:]).strip()

    if explanation.lower().startswith("explanation:"):
        explanation = explanation[len("explanation:"):].strip()

    return explanation


def save_results(results):
    """Save results after each test."""

    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )


def judge_explanation(test_case, raw_output):
    """Ask the local model to evaluate the explanation."""

    explanation = extract_explanation(raw_output)

    if not explanation:
        return {
            "score": None,
            "reason": "No explanation was provided."
        }

    # The judge's prompt is defined before it is used.
    prompt = f"""
You are a fair evaluator of basic computer science explanations.

Score the candidate explanation from 1 to 5:

5 = Correct, clear, and sufficiently explains the concept.
4 = Correct with a minor omission.
3 = Partly correct but missing an important detail.
2 = Contains a significant technical error.
1 = Incorrect, irrelevant, or fails to explain the concept.

Rules:
- Do not penalize short explanations when the question is simple.
- If the key concept is correctly explained, normally score 5.
- Judge technical correctness, not writing style.
- Compare the candidate with the reference explanation.

Question:
{test_case["question"]}

Correct answer:
{test_case["expected_answer"]}

Reference explanation:
{test_case["reference_explanation"]}

Candidate explanation:
{explanation}

Return only one JSON object.
The score must be an integer from 1 to 5.

Required format:
{{"score": 5, "reason": "The key concept is explained correctly."}}
""".strip()

    try:
        judge_output = call_ollama(
            prompt,
            num_predict=100
        )

        match = re.search(
            r"\{.*?\}",
            judge_output,
            flags=re.DOTALL
        )

        if not match:
            return {
                "score": None,
                "reason": "The judge did not return valid JSON.",
                "raw_judge_output": judge_output
            }

        judge_data = json.loads(match.group(0))
        score = judge_data.get("score")
        reason = judge_data.get("reason", "")

        if type(score) is not int or score not in range(1, 6):
            return {
                "score": None,
                "reason": "The judge returned an invalid score.",
                "raw_judge_output": judge_output
            }

        return {
            "score": score,
            "reason": str(reason)
        }

    except (requests.RequestException, ValueError, KeyError) as error:
        return {
            "score": None,
            "reason": f"Judge error: {error}"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a local Ollama model."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to test."
    )

    parser.add_argument(
        "--judge-sample",
        type=int,
        default=5,
        help="Number of explanations for the judge to score."
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        print("--limit must be at least 1.")
        return

    if args.judge_sample < 0:
        print("--judge-sample cannot be negative.")
        return

    try:
        with TEST_FILE.open("r", encoding="utf-8") as file:
            test_cases = json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read {TEST_FILE}: {error}")
        return

    if args.limit is not None:
        test_cases = test_cases[:args.limit]

    if not test_cases:
        print("No test cases were found.")
        return

    print(f"Model: {MODEL}")
    print(f"Tests to run: {len(test_cases)}")
    print("Starting evaluation...\n")

    results = []

    for index, test_case in enumerate(test_cases, start=1):
        print(
            f"[{index}/{len(test_cases)}] "
            f"Test ID: {test_case['id']}"
        )

        result = {
            "id": test_case["id"],
            "topic": test_case["topic"],
            "expected_answer": test_case["expected_answer"],
            "raw_output": "",
            "parsed_answer": None,
            "format_valid": False,
            "is_correct": False,
            "status": "error",
            "elapsed_seconds": None,
            "explanation": "",
            "judge_score": None,
            "judge_reason": ""
        }

        start_time = time.perf_counter()

        try:
            prompt = build_question_prompt(test_case)
            raw_output = call_ollama(prompt)
            elapsed = time.perf_counter() - start_time

            validation = validate_answer(
                raw_output,
                test_case["expected_answer"]
            )

            result.update({
                "raw_output": raw_output,
                "parsed_answer": validation["parsed_answer"],
                "format_valid": validation["format_valid"],
                "is_correct": validation["is_correct"],
                "status": (
                    "pass"
                    if validation["is_correct"]
                    else (
                        "incorrect"
                        if validation["format_valid"]
                        else "invalid_format"
                    )
                ),
                "elapsed_seconds": round(elapsed, 2),
                "explanation": extract_explanation(raw_output)
            })

            print(f"  Expected: {test_case['expected_answer']}")
            print(f"  Received: {validation['parsed_answer']}")
            print(f"  Status: {result['status']}")

        except (
            requests.RequestException,
            ValueError,
            KeyError
        ) as error:
            elapsed = time.perf_counter() - start_time
            result["elapsed_seconds"] = round(elapsed, 2)
            result["error"] = str(error)

            print(f"  ERROR: {error}")

        results.append(result)
        save_results(results)

    # Evaluate a small sample of explanations.
    eligible = [
        item for item in results
        if item["raw_output"] and item["status"] != "error"
    ]

    eligible = eligible[:args.judge_sample]
    cases_by_id = {case["id"]: case for case in test_cases}

    if eligible:
        print("\nScoring a sample of explanations with the LLM judge...")

    for item in eligible:
        test_case = cases_by_id[item["id"]]

        judgment = judge_explanation(
            test_case,
            item["raw_output"]
        )

        item["judge_score"] = judgment["score"]
        item["judge_reason"] = judgment["reason"]

        save_results(results)

        print(
            f"  Test {item['id']}: "
            f"Judge score = {item['judge_score']}"
        )

    total = len(results)
    correct_count = sum(
        item["is_correct"] for item in results
    )
    valid_count = sum(
        item["format_valid"] for item in results
    )
    error_count = sum(
        item["status"] == "error" for item in results
    )

    accuracy = 100 * correct_count / total
    format_rate = 100 * valid_count / total

    print("\n========== Evaluation Summary ==========")
    print(f"Total tests: {total}")
    print(f"Correct answers: {correct_count}")
    print(f"Errors: {error_count}")
    print(f"Answer accuracy: {accuracy:.2f}%")
    print(f"Valid answer format rate: {format_rate:.2f}%")
    print(f"Results saved to: {RESULT_FILE.resolve()}")


if __name__ == "__main__":
    main()