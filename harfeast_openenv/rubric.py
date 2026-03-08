"""Rubric scoring for HarFeast OpenEnv."""

import re
from typing import Sequence


def _extract_expected_value(criterion: str) -> str | None:
    """
    Extract the expected value from a rubric criterion.
    Pattern: "States that ... is VALUE" or "States that ... VALUE"
    """
    # Match " is X" or " is $X" at the end
    m = re.search(r"\s+is\s+(.+)$", criterion)
    if m:
        return m.group(1).strip().strip('"')
    return None


def _normalize_for_match(value: str) -> list[str]:
    """
    Return variants of the value to check against the answer.
    Handles numbers with commas, percentages, etc.
    """
    value = value.strip()
    variants = [value]
    # Remove commas from numbers
    no_commas = value.replace(",", "")
    if no_commas != value:
        variants.append(no_commas)
    # For percentages: "14%" -> also accept "14" and "14 percent"
    if value.endswith("%"):
        num_part = value[:-1].strip()
        variants.extend([num_part, f"{num_part}%", f"{num_part} percent"])
        # Remove trailing .0 for whole numbers
        if "." in num_part and num_part.endswith("0"):
            variants.append(num_part.rstrip("0").rstrip("."))
    # For dollar amounts: "$21,953,848,911" -> also without $
    if value.startswith("$"):
        variants.append(value[1:].strip())
        variants.append(value[1:].replace(",", ""))
    # For decimals like 87.00% - accept 87
    if "%" in value and "." in value:
        num_part = value.replace("%", "").strip()
        try:
            f = float(num_part)
            if f == int(f):
                variants.append(str(int(f)))
        except ValueError:
            pass
    return list(dict.fromkeys(variants))  # dedupe preserving order


def _answer_contains_value(answer: str, expected: str) -> bool:
    """Check if answer contains the expected value (or a normalized variant)."""
    answer_lower = answer.lower()
    variants = _normalize_for_match(expected)
    for v in variants:
        if not v:
            continue
        # Case-insensitive for text; exact substring for numbers
        if v.lower() in answer_lower:
            return True
        # For numbers, also check without leading zeros
        if v.isdigit() and str(int(v)) in answer:
            return True
    return False


def score_answer(answer: str, rubric: Sequence[str]) -> tuple[float, list[tuple[str, bool]]]:
    """
    Score an answer against rubric criteria.
    Returns (score_0_to_100, list of (criterion, passed)).
    """
    if not rubric:
        return 100.0, []
    results = []
    for criterion in rubric:
        expected = _extract_expected_value(criterion)
        if expected is None:
            # No " is X" pattern - fall back to substring of criterion
            # e.g. "States that X" - check if key phrase appears
            key = criterion.replace("States that ", "").strip()
            passed = key.lower() in answer.lower()
        else:
            passed = _answer_contains_value(answer, expected)
        results.append((criterion, passed))
    passed_count = sum(1 for _, p in results if p)
    score = (passed_count / len(rubric)) * 100.0
    return round(score, 1), results
