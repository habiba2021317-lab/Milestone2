import re


def parse_answer(raw_text):
    """
    Read the first non-empty line and extract A, B, C, or D.
    Accepted examples: 'B' and 'Answer: B'.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not lines:
        return None

    first_line = lines[0]
    match = re.fullmatch(
        r"(?:answer\s*:\s*)?([ABCD])[\.\)]?",
        first_line,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return None


def validate_answer(raw_text, expected_answer):
    """
    Check both the answer format and whether the choice is correct.
    """
    parsed_answer = parse_answer(raw_text)

    format_valid = parsed_answer is not None
    is_correct = (
        format_valid
        and parsed_answer == expected_answer.upper()
    )

    return {
        "parsed_answer": parsed_answer,
        "format_valid": format_valid,
        "is_correct": is_correct
    }