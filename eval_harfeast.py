#!/usr/bin/env python3
"""
HarFeast evaluation script — score a model on all 14 tasks.

Usage:
  python eval_harfeast.py --model unsloth/Qwen3-4B --world ./harfeast_world
  python eval_harfeast.py --model ./checkpoints --world ./harfeast_world
  python eval_harfeast.py --model unsloth/Qwen3-4B --model-after ./checkpoints --world ./harfeast_world
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def evaluate_model(model_path, tasks, max_tasks=None):
    """
    Generate answers for each unique task and score against rubric.
    Returns (overall_pct, list of per-task results).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from harfeast_openenv.rubric import score_answer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    seen_ids = set()
    unique_tasks = []
    for t in tasks:
        if t["task_id"] not in seen_ids:
            seen_ids.add(t["task_id"])
            unique_tasks.append(t)
    if max_tasks:
        unique_tasks = unique_tasks[:max_tasks]

    system = (
        "You are a management consultant analyzing HarFeast data. "
        "Given a task, provide your final answer with specific numbers. "
        "Format: Answer: <your analysis with numbers>"
    )

    total_passed, total_criteria = 0, 0
    results = []

    for t in unique_tasks:
        prompt = f"{system}\n\nTask:\n{t['prompt']}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=512, temperature=0.7,
                do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answer = completion.split("Answer:")[-1].strip() if "Answer:" in completion else completion.strip()

        rubric = t.get("rubric", [])
        score, criteria_results = score_answer(answer, rubric)
        passed = sum(1 for _, p in criteria_results if p)
        total = len(criteria_results)
        total_passed += passed
        total_criteria += total

        results.append({
            "task_id": t["task_id"],
            "task_name": t["task_name"],
            "passed": passed,
            "total": total,
            "score": score,
            "answer_preview": answer[:200],
        })

    overall = (total_passed / total_criteria * 100) if total_criteria > 0 else 0

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return round(overall, 1), results, total_passed, total_criteria


def print_results(model_path, overall, results, total_passed, total_criteria):
    """Pretty-print evaluation results."""
    print(f"\n{'='*72}")
    print(f"  Model: {model_path}")
    print(f"{'='*72}")
    print(f"{'Task':<10} {'Name':<38} {'Passed':>10} {'Score':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['task_id']:<10} {r['task_name'][:36]:<38} {r['passed']}/{r['total']:>3}      {r['score']:>5.1f}%")
    print("-" * 72)
    print(f"{'OVERALL':<10} {'':<38} {total_passed}/{total_criteria:>3}      {overall:>5.1f}%")
    print(f"{'='*72}")
    return overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on HarFeast tasks")
    parser.add_argument("--model", required=True, help="Model name or checkpoint path")
    parser.add_argument("--model-after", default=None,
                        help="Second model (trained) for before/after comparison")
    parser.add_argument("--world", default="./harfeast_world", help="World directory")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    args = parser.parse_args()

    tasks_path = os.path.join(args.world, "tasks.json")
    if not os.path.isfile(tasks_path):
        print(f"tasks.json not found in {args.world}")
        sys.exit(1)
    with open(tasks_path) as f:
        tasks = json.load(f)

    overall_before, results_before, tp_before, tc_before = evaluate_model(
        args.model, tasks, args.max_tasks
    )
    print_results(args.model, overall_before, results_before, tp_before, tc_before)

    if args.model_after:
        overall_after, results_after, tp_after, tc_after = evaluate_model(
            args.model_after, tasks, args.max_tasks
        )
        print_results(args.model_after, overall_after, results_after, tp_after, tc_after)

        print(f"\n{'='*72}")
        print(f"  COMPARISON")
        print(f"{'='*72}")
        print(f"  Before ({args.model}):      {overall_before:.1f}%")
        print(f"  After  ({args.model_after}): {overall_after:.1f}%")
        delta = overall_after - overall_before
        print(f"  Delta:                       {'+' if delta >= 0 else ''}{delta:.1f}%")
        print(f"{'='*72}")

        # Per-task comparison
        print(f"\n{'Task':<10} {'Before':>10} {'After':>10} {'Delta':>10}")
        print("-" * 44)
        for rb, ra in zip(results_before, results_after):
            d = ra["score"] - rb["score"]
            sign = "+" if d >= 0 else ""
            print(f"{rb['task_id']:<10} {rb['score']:>8.1f}%  {ra['score']:>8.1f}%  {sign}{d:>7.1f}%")


if __name__ == "__main__":
    main()
