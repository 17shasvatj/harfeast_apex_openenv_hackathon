#!/usr/bin/env python3
"""
HarFeast GRPO training with GDPO-style multi-signal rewards.

Uses 3 independent reward functions (correctness, format, completeness)
scored in-process against deterministic rubrics. No env server needed.

Usage:
  python train_harfeast.py --model unsloth/Qwen3-4B
  python train_harfeast.py --model unsloth/Qwen3-4B --worlds-base ./harfeast_worlds --samples 128
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_CSV_ROWS_IN_PROMPT = 30
MAX_DOC_CHARS_IN_PROMPT = 1500


def load_world_data_summary(world_path):
    """Read CSVs and documents from a world directory into a compact text summary."""
    sections = []

    data_dir = os.path.join(world_path, "data")
    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                if not rows:
                    continue
                header = rows[0]
                data_rows = rows[1:]
                lines = [f"## {fname} ({len(data_rows)} rows)"]
                lines.append(" | ".join(header))
                for r in data_rows[:MAX_CSV_ROWS_IN_PROMPT]:
                    lines.append(" | ".join(r))
                if len(data_rows) > MAX_CSV_ROWS_IN_PROMPT:
                    lines.append(f"[...{len(data_rows) - MAX_CSV_ROWS_IN_PROMPT} more rows]")
                sections.append("\n".join(lines))
            except Exception:
                continue

    doc_dir = os.path.join(world_path, "documents")
    if os.path.isdir(doc_dir):
        for fname in sorted(os.listdir(doc_dir)):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(doc_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > MAX_DOC_CHARS_IN_PROMPT:
                    content = content[:MAX_DOC_CHARS_IN_PROMPT] + "..."
                sections.append(f"## {fname}\n{content}")
            except Exception:
                continue

    return "\n\n".join(sections)


def load_tasks(worlds_base=None, single_world=None):
    """Load task list from augmented dataset or single world. Attaches world_path to each task."""
    if worlds_base:
        path = os.path.join(worlds_base, "all_tasks.json")
        if not os.path.isfile(path):
            print(f"all_tasks.json not found in {worlds_base}.")
            print(f"Run: python harfeast_synthetic_world_generator.py --batch 40 --output-dir {worlds_base}")
            sys.exit(1)
        with open(path) as f:
            all_tasks = json.load(f)
        tasks = []
        for entry in all_tasks:
            wp = entry["world_path"]
            if not os.path.isabs(wp):
                wp = os.path.join(worlds_base, os.path.basename(wp.rstrip("/")))
            wp = os.path.abspath(wp)
            tasks_path = os.path.join(wp, "tasks.json")
            with open(tasks_path) as f:
                world_tasks = json.load(f)
            task = next(t for t in world_tasks if t["task_id"] == entry["task_id"])
            task["world_path"] = wp
            tasks.append(task)
        return tasks

    world = single_world or os.path.join(os.path.dirname(__file__), "harfeast_world")
    world = os.path.abspath(world)
    tasks_path = os.path.join(world, "tasks.json")
    if not os.path.isfile(tasks_path):
        print("Run harfeast_synthetic_world_generator.py first.")
        sys.exit(1)
    with open(tasks_path) as f:
        tasks = json.load(f)
    for t in tasks:
        t["world_path"] = world
    return tasks


_data_cache = {}

def _get_data_summary(world_path):
    """Cached data loading per world."""
    if world_path not in _data_cache:
        _data_cache[world_path] = load_world_data_summary(world_path)
    return _data_cache[world_path]


def build_dataset(tasks, n_samples, seed=42):
    """Build HF Dataset with prompt (including data) + rubric columns for GRPO training."""
    from datasets import Dataset
    import random

    rng = random.Random(seed)
    system = (
        "You are a management consultant analyzing HarFeast food manufacturing data.\n"
        "Below is the company data, followed by a task. Analyze the data and provide "
        "your final answer with specific numbers.\n"
        "Format: Answer: <your analysis with specific numbers>"
    )

    indices = [rng.randint(0, len(tasks) - 1) for _ in range(n_samples)]
    prompts, rubrics, ground_truths = [], [], []
    for i in indices:
        t = tasks[i]
        data_summary = _get_data_summary(t["world_path"])
        prompt = f"{system}\n\n# Company Data\n{data_summary}\n\n# Task\n{t['prompt']}\n\nAnswer:"
        prompts.append(prompt)
        rubrics.append(json.dumps(t.get("rubric", [])))
        gt_values = []
        for criterion in t.get("rubric", []):
            import re
            m = re.search(r"\s+is\s+(.+)$", criterion)
            if m:
                gt_values.append(m.group(1).strip().strip('"'))
        ground_truths.append(json.dumps(gt_values))

    return Dataset.from_dict({
        "prompt": prompts,
        "rubric": rubrics,
        "ground_truth": ground_truths,
    })


def quick_eval(model_name_or_path, tasks, tokenizer=None, label="eval"):
    """Run a quick eval: generate for each unique task, score with rubric."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"\n{'='*60}")
    print(f"  {label}: {model_name_or_path}")
    print(f"{'='*60}")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    from harfeast_openenv.rubric import score_answer

    seen_ids = set()
    unique_tasks = []
    for t in tasks:
        if t["task_id"] not in seen_ids:
            seen_ids.add(t["task_id"])
            unique_tasks.append(t)

    system = (
        "You are a management consultant analyzing HarFeast food manufacturing data.\n"
        "Below is the company data, followed by a task. Analyze the data and provide "
        "your final answer with specific numbers.\n"
        "Format: Answer: <your analysis with specific numbers>"
    )

    total_passed, total_criteria = 0, 0
    results = []

    for t in unique_tasks:
        data_summary = _get_data_summary(t["world_path"])
        prompt = f"{system}\n\n# Company Data\n{data_summary}\n\n# Task\n{t['prompt']}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
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
        results.append((t["task_id"], t["task_name"], passed, total, score))

    print(f"\n{'Task':<10} {'Name':<40} {'Passed':>8} {'Score':>8}")
    print("-" * 70)
    for tid, name, passed, total, score in results:
        print(f"{tid:<10} {name[:38]:<40} {passed}/{total:>3}    {score:>5.1f}%")

    overall = (total_passed / total_criteria * 100) if total_criteria > 0 else 0
    print("-" * 70)
    print(f"{'OVERALL':<10} {'':<40} {total_passed}/{total_criteria:>3}    {overall:>5.1f}%")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return overall


def main():
    parser = argparse.ArgumentParser(description="HarFeast GRPO Training")
    parser.add_argument("--model", default="unsloth/Qwen3-4B")
    parser.add_argument("--worlds-base", default=None,
                        help="Augmented dataset dir (harfeast_worlds)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--eval-before", action=argparse.BooleanOptionalAction, default=True,
                        help="Run eval on base model before training (--no-eval-before to skip)")
    parser.add_argument("--use-vllm", action="store_true", default=False,
                        help="Use vLLM for generation (requires vllm installed)")
    parser.add_argument("--output-dir", default="./checkpoints")
    args = parser.parse_args()

    tasks = load_tasks(worlds_base=args.worlds_base)
    print(f"Loaded {len(tasks)} tasks")

    dataset = build_dataset(tasks, args.samples)
    print(f"Training dataset: {len(dataset)} samples")
    print(f"Prompt length (sample): {len(dataset[0]['prompt'])} chars")

    # Before-training eval
    before_score = None
    if args.eval_before:
        before_score = quick_eval(args.model, tasks, label="BEFORE training")

    # Import reward functions and trainer
    from harfeast_openenv.rewards import reward_correctness, reward_format, reward_completeness
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    class LoggingCallback(TrainerCallback):
        """Print live training progress."""
        def __init__(self):
            self._start = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            elapsed = time.time() - self._start
            step = state.global_step
            total = state.max_steps
            pct = (step / total * 100) if total > 0 else 0
            reward_str = ""
            for k in sorted(logs):
                if "reward" in k.lower() or "loss" in k.lower():
                    reward_str += f"  {k}={logs[k]:.4f}"
            print(f"[Step {step}/{total} ({pct:.0f}%) | {elapsed:.0f}s]{reward_str}", flush=True)

    vllm_kwargs = {}
    if args.use_vllm:
        vllm_kwargs = {"use_vllm": True, "vllm_mode": "colocate"}

    config = GRPOConfig(
        output_dir=args.output_dir,
        **vllm_kwargs,
        num_train_epochs=args.epochs,
        num_generations=4,
        max_completion_length=512,
        max_prompt_length=4096,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_steps=50,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_correctness, reward_format, reward_completeness],
        train_dataset=dataset,
        args=config,
        callbacks=[LoggingCallback()],
    )

    print("\nStarting GRPO training with 3 reward signals (correctness, format, completeness)...")
    print(f"  use_vllm={args.use_vllm}  epochs={args.epochs}  samples={args.samples}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    # After-training eval
    after_score = quick_eval(args.output_dir, tasks, label="AFTER training")

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    if before_score is not None:
        print(f"  Before: {before_score:.1f}%")
    print(f"  After:  {after_score:.1f}%")
    if before_score is not None:
        delta = after_score - before_score
        print(f"  Delta:  {'+' if delta >= 0 else ''}{delta:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
