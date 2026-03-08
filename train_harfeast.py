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
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_tasks(worlds_base=None, single_world=None):
    """Load task list from augmented dataset or single world."""
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
            tasks_path = os.path.join(os.path.abspath(wp), "tasks.json")
            with open(tasks_path) as f:
                world_tasks = json.load(f)
            task = next(t for t in world_tasks if t["task_id"] == entry["task_id"])
            tasks.append(task)
        return tasks

    world = single_world or os.path.join(os.path.dirname(__file__), "harfeast_world")
    tasks_path = os.path.join(world, "tasks.json")
    if not os.path.isfile(tasks_path):
        print("Run harfeast_synthetic_world_generator.py first.")
        sys.exit(1)
    with open(tasks_path) as f:
        return json.load(f)


def build_dataset(tasks, n_samples, seed=42):
    """Build HF Dataset with prompt + rubric columns for GRPO training."""
    from datasets import Dataset
    import random

    rng = random.Random(seed)
    system = (
        "You are a management consultant analyzing HarFeast data. "
        "Given a task, provide your final answer with specific numbers. "
        "Format: Answer: <your analysis with numbers>"
    )

    indices = [rng.randint(0, len(tasks) - 1) for _ in range(n_samples)]
    prompts, rubrics = [], []
    for i in indices:
        t = tasks[i]
        prompts.append(f"{system}\n\nTask:\n{t['prompt']}\n\nAnswer:")
        rubrics.append(json.dumps(t.get("rubric", [])))

    return Dataset.from_dict({"prompt": prompts, "rubric": rubrics})


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
    parser.add_argument("--eval-before", action="store_true", default=True,
                        help="Run eval on base model before training")
    parser.add_argument("--output-dir", default="./checkpoints")
    args = parser.parse_args()

    tasks = load_tasks(worlds_base=args.worlds_base)
    print(f"Loaded {len(tasks)} tasks")

    dataset = build_dataset(tasks, args.samples)
    print(f"Training dataset: {len(dataset)} samples")

    # Before-training eval
    before_score = None
    if args.eval_before:
        before_score = quick_eval(args.model, tasks, label="BEFORE training")

    # Import reward functions and trainer
    from harfeast_openenv.rewards import reward_correctness, reward_format, reward_completeness
    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode="colocate",
        num_train_epochs=args.epochs,
        num_generations=4,
        max_completion_length=512,
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
    )

    print("\nStarting GRPO training with 3 reward signals (correctness, format, completeness)...")
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
