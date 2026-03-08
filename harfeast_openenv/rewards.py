"""
GDPO-style decomposed reward functions for HarFeast GRPO training.

Three independent reward signals, each normalized independently by TRL's
GRPOTrainer when passed as a list to reward_funcs. This is equivalent to
NVIDIA's GDPO (Jan 2026) multi-signal normalization.

Signature: reward_func(completions: list[list[dict]], **kwargs) -> list[float]
  - completions[i] = [{"role": "assistant", "content": "..."}]
  - kwargs include dataset columns: "rubric" (JSON-serialized list of criteria)
"""

import json
import re
from .rubric import score_answer


def _extract_text(completions):
    """Extract plain text from TRL chat-format completions."""
    texts = []
    for comp in completions:
        if isinstance(comp, list) and comp:
            texts.append(comp[-1].get("content", ""))
        elif isinstance(comp, str):
            texts.append(comp)
        else:
            texts.append("")
    return texts


def _extract_answer(text):
    """Pull the answer portion after 'Answer:' if present."""
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()


def reward_correctness(completions, **kwargs):
    """
    Signal 1: Rubric correctness (0.0 - 1.0).
    Scores each completion against task rubric criteria using deterministic
    substring matching. This is the primary learning signal.
    """
    texts = _extract_text(completions)
    rubric_strs = kwargs.get("rubric", [])
    rewards = []
    for i, text in enumerate(texts):
        answer = _extract_answer(text)
        try:
            rubric = json.loads(rubric_strs[i]) if i < len(rubric_strs) else []
        except (json.JSONDecodeError, TypeError):
            rubric = []
        if not rubric:
            rewards.append(0.0)
            continue
        score, _ = score_answer(answer, rubric)
        rewards.append(score / 100.0)
    return rewards


def reward_format(completions, **kwargs):
    """
    Signal 2: Format compliance (0.0 or 1.0).
    Checks that the completion follows the expected output structure:
    contains 'Answer:', includes at least one number, reasonable length.
    """
    texts = _extract_text(completions)
    rewards = []
    for text in texts:
        score = 0.0
        has_answer_prefix = "Answer:" in text or "answer:" in text.lower()
        has_number = bool(re.search(r"\d+\.?\d*", text))
        reasonable_length = 50 <= len(text) <= 3000
        if has_answer_prefix and has_number and reasonable_length:
            score = 1.0
        elif has_number and reasonable_length:
            score = 0.5
        rewards.append(score)
    return rewards


def reward_completeness(completions, **kwargs):
    """
    Signal 3: Numeric completeness (0.0 - 1.0).
    Measures how many distinct numeric values appear in the answer relative
    to the number of rubric criteria. Rewards specificity: an answer with
    concrete numbers for every criterion scores higher.
    """
    texts = _extract_text(completions)
    rubric_strs = kwargs.get("rubric", [])
    rewards = []
    for i, text in enumerate(texts):
        answer = _extract_answer(text)
        try:
            rubric = json.loads(rubric_strs[i]) if i < len(rubric_strs) else []
        except (json.JSONDecodeError, TypeError):
            rubric = []
        n_criteria = max(len(rubric), 1)
        numbers = set(re.findall(r"\b\d[\d,.]*\d\b|\b\d+\b", answer))
        ratio = min(len(numbers) / n_criteria, 1.0)
        rewards.append(round(ratio, 3))
    return rewards
