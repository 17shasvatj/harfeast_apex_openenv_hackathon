#!/usr/bin/env python3
"""
HarFeast GRPO training with GDPO-style multi-signal rewards.

Uses 3 independent reward functions (correctness, format, completeness)
scored in-process against deterministic rubrics. No env server needed.

Usage (H200, proper training):
  python train_harfeast.py --model unsloth/Qwen3-4B --worlds-base ./harfeast_worlds \
      --samples 256 --epochs 3 --output-dir ./checkpoints

Usage (quick test):
  python train_harfeast.py --model unsloth/Qwen3-4B --samples 32 --epochs 1
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*vLLM.*")
warnings.filterwarnings("ignore", message=".*TRL currently supports.*")

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
    """Build HF Dataset with prompt (including data) + rubric + ground_truth columns."""
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
            m = re.search(r"\s+is\s+(.+)$", criterion)
            if m:
                gt_values.append(m.group(1).strip().strip('"'))
        ground_truths.append(json.dumps(gt_values))

    return Dataset.from_dict({
        "prompt": prompts,
        "rubric": rubrics,
        "ground_truth": ground_truths,
    })


def run_eval(model_name_or_path, tasks, label="eval"):
    """Full eval: generate for each unique task, score with rubric. Returns (score, results_list)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"\n{'='*70}")
    print(f"  {label}: {model_name_or_path}")
    print(f"{'='*70}")

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

    for i, t in enumerate(unique_tasks):
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
        results.append({
            "task_id": t["task_id"],
            "task_name": t["task_name"],
            "passed": passed,
            "total": total,
            "score": score,
        })
        print(f"  [{i+1}/{len(unique_tasks)}] {t['task_id']}  {passed}/{total}  {score:.1f}%", flush=True)

    overall = (total_passed / total_criteria * 100) if total_criteria > 0 else 0

    print(f"\n{'Task':<10} {'Name':<40} {'Passed':>8} {'Score':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['task_id']:<10} {r['task_name'][:38]:<40} {r['passed']}/{r['total']:>3}    {r['score']:>5.1f}%")
    print("-" * 70)
    print(f"{'OVERALL':<10} {'':<40} {total_passed}/{total_criteria:>3}    {overall:>5.1f}%")
    print(f"{'='*70}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return round(overall, 1), results


def main():
    parser = argparse.ArgumentParser(description="HarFeast GRPO Training")
    parser.add_argument("--model", default="unsloth/Qwen3-4B")
    parser.add_argument("--worlds-base", default=None,
                        help="Augmented dataset dir (harfeast_worlds)")
    parser.add_argument("--single-world", default=None,
                        help="Single world dir for eval (harfeast_world)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Completions per prompt per GRPO step")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max-completion-length", type=int, default=512,
                        help="Max tokens to generate per completion")
    parser.add_argument("--eval-before", action="store_true", default=True,
                        help="Run eval on base model before training")
    parser.add_argument("--no-eval-before", dest="eval_before", action="store_false",
                        help="Skip before-training eval")
    parser.add_argument("--use-vllm", action="store_true", default=False,
                        help="Use vLLM for generation (requires compatible vllm)")
    parser.add_argument("--output-dir", default="./checkpoints")
    args = parser.parse_args()

    start_time = time.time()

    # Load tasks
    tasks = load_tasks(worlds_base=args.worlds_base, single_world=args.single_world)
    print(f"Loaded {len(tasks)} tasks")

    # For eval, use single world if available, otherwise first world from augmented set
    eval_tasks = tasks
    if args.single_world:
        eval_tasks = load_tasks(single_world=args.single_world)

    # Build dataset
    dataset = build_dataset(tasks, args.samples)
    print(f"Training dataset: {len(dataset)} samples")
    print(f"Prompt length (sample): {len(dataset[0]['prompt'])} chars")

    # Training math
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(len(dataset) // effective_batch, 1)
    total_steps = steps_per_epoch * args.epochs
    print(f"\nTraining plan:")
    print(f"  Model:          {args.model}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Samples:        {args.samples}")
    print(f"  Batch:          {args.batch_size} x {args.grad_accum} = {effective_batch} effective")
    print(f"  Steps/epoch:    ~{steps_per_epoch}")
    print(f"  Total steps:    ~{total_steps}")
    print(f"  Generations:    {args.num_generations} per prompt")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Rewards:        correctness + format + completeness (GDPO-style)")

    # ── Before-training eval ──
    before_score, before_results = None, None
    if args.eval_before:
        before_score, before_results = run_eval(args.model, eval_tasks, label="BEFORE training")

    # ── GRPO Training ──
    from harfeast_openenv.rewards import reward_correctness, reward_format, reward_completeness
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    class LoggingCallback(TrainerCallback):
        def __init__(self):
            self._start = time.time()
            self._rewards = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            elapsed = time.time() - self._start
            step = state.global_step
            total = state.max_steps
            pct = (step / total * 100) if total > 0 else 0

            parts = []
            for k in sorted(logs):
                if isinstance(logs[k], (int, float)):
                    parts.append(f"{k}={logs[k]:.4f}")
            log_str = "  ".join(parts)
            print(f"[Step {step}/{total} ({pct:.0f}%) | {elapsed:.0f}s]  {log_str}", flush=True)

            if "reward" in str(logs):
                self._rewards.append({"step": step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}})

        def on_train_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self._start
            print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
            if self._rewards:
                first = self._rewards[0]
                last = self._rewards[-1]
                print(f"  First step rewards: {first}")
                print(f"  Last step rewards:  {last}")

    grpo_kwargs = {
        "output_dir": args.output_dir,
        "use_vllm": args.use_vllm,
        "num_train_epochs": args.epochs,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "logging_steps": 1,
        "save_steps": 50,
        "save_total_limit": 2,
        "bf16": True,
        "warmup_steps": 2,
        "report_to": "none",
    }
    try:
        config = GRPOConfig(max_prompt_length=4096, **grpo_kwargs)
    except TypeError:
        config = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_correctness, reward_format, reward_completeness],
        train_dataset=dataset,
        args=config,
        callbacks=[LoggingCallback()],
    )

    print("\n" + "="*70)
    print("  STARTING GRPO TRAINING")
    print("="*70)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    # ── After-training eval ──
    after_score, after_results = run_eval(args.output_dir, eval_tasks, label="AFTER training")

    # ── Summary ──
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    if before_score is not None:
        print(f"  Before: {before_score:.1f}%")
    print(f"  After:  {after_score:.1f}%")
    if before_score is not None:
        delta = after_score - before_score
        print(f"  Delta:  {'+' if delta >= 0 else ''}{delta:.1f}%")
    print(f"  Time:   {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")

    if before_results and after_results:
        print(f"\n{'Task':<10} {'Before':>10} {'After':>10} {'Delta':>10}")
        print("-" * 44)
        for rb, ra in zip(before_results, after_results):
            d = ra["score"] - rb["score"]
            sign = "+" if d >= 0 else ""
            print(f"{rb['task_id']:<10} {rb['score']:>8.1f}%  {ra['score']:>8.1f}%  {sign}{d:>7.1f}%")

    # Save results JSON for later analysis
    results_path = os.path.join(args.output_dir, "training_results.json")
    results_data = {
        "model": args.model,
        "epochs": args.epochs,
        "samples": args.samples,
        "num_generations": args.num_generations,
        "learning_rate": args.lr,
        "before_score": before_score,
        "after_score": after_score,
        "before_results": before_results,
        "after_results": after_results,
        "total_time_s": round(total_time, 1),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
