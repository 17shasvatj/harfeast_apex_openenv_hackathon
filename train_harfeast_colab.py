"""
HarFeast GRPO Training — Colab-ready script.
GDPO-style multi-signal rewards (correctness, format, completeness).

Copy cells into Colab or run as: python train_harfeast_colab.py

Setup (Colab cell 1):
  !pip install -q trl datasets accelerate transformers
  !git clone https://github.com/17shasvatj/harfeast_apex_openenv_hackathon.git
  %cd harfeast_apex_openenv_hackathon
"""

# %% [markdown]
# # HarFeast OpenEnv — GRPO Training with Multi-Signal Rewards
#
# Train `unsloth/Qwen3-4B` on management consulting tasks using 3 independent
# reward signals (GDPO-style): **correctness**, **format**, **completeness**.
# Each signal is normalized independently by TRL's GRPOTrainer.

# %%  Cell 1: Setup & Imports
import csv
import json
import os
import re
import sys
import time

if os.path.isdir("harfeast_apex_openenv_hackathon"):
    os.chdir("harfeast_apex_openenv_hackathon")
sys.path.insert(0, os.getcwd())

from datasets import Dataset

# %% Cell 2: Data loading helpers

MAX_CSV_ROWS = 30
MAX_DOC_CHARS = 1500

def load_world_data_summary(world_path):
    """Read CSVs and documents into a compact text summary for prompts."""
    sections = []
    data_dir = os.path.join(world_path, "data")
    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                if not rows:
                    continue
                header, data_rows = rows[0], rows[1:]
                lines = [f"## {fname} ({len(data_rows)} rows)", " | ".join(header)]
                for r in data_rows[:MAX_CSV_ROWS]:
                    lines.append(" | ".join(r))
                if len(data_rows) > MAX_CSV_ROWS:
                    lines.append(f"[...{len(data_rows) - MAX_CSV_ROWS} more rows]")
                sections.append("\n".join(lines))
            except Exception:
                continue
    doc_dir = os.path.join(world_path, "documents")
    if os.path.isdir(doc_dir):
        for fname in sorted(os.listdir(doc_dir)):
            if not fname.endswith(".txt"):
                continue
            try:
                with open(os.path.join(doc_dir, fname), "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > MAX_DOC_CHARS:
                    content = content[:MAX_DOC_CHARS] + "..."
                sections.append(f"## {fname}\n{content}")
            except Exception:
                continue
    return "\n\n".join(sections)


# %% Cell 3: Load tasks and build dataset

def load_tasks():
    for p in ["harfeast_world/tasks.json",
              "harfeast_apex_openenv_hackathon/harfeast_world/tasks.json"]:
        if os.path.isfile(p):
            world_path = os.path.dirname(os.path.abspath(p))
            with open(p) as f:
                tasks = json.load(f)
            for t in tasks:
                t["world_path"] = world_path
            return tasks
    raise FileNotFoundError("tasks.json not found. Run the world generator first.")

tasks = load_tasks()
print(f"Loaded {len(tasks)} tasks")

_data_cache = {}
def get_data_summary(world_path):
    if world_path not in _data_cache:
        _data_cache[world_path] = load_world_data_summary(world_path)
    return _data_cache[world_path]

SYSTEM = (
    "You are a management consultant analyzing HarFeast food manufacturing data.\n"
    "Below is the company data, followed by a task. Analyze the data and provide "
    "your final answer with specific numbers.\n"
    "Format: Answer: <your analysis with specific numbers>"
)

import random
rng = random.Random(42)
N_SAMPLES = 64
indices = [rng.randint(0, len(tasks) - 1) for _ in range(N_SAMPLES)]

prompts, rubrics, ground_truths = [], [], []
for i in indices:
    t = tasks[i]
    data_summary = get_data_summary(t["world_path"])
    prompts.append(f"{SYSTEM}\n\n# Company Data\n{data_summary}\n\n# Task\n{t['prompt']}\n\nAnswer:")
    rubrics.append(json.dumps(t.get("rubric", [])))
    gt = []
    for c in t.get("rubric", []):
        m = re.search(r"\s+is\s+(.+)$", c)
        if m:
            gt.append(m.group(1).strip().strip('"'))
    ground_truths.append(json.dumps(gt))

dataset = Dataset.from_dict({
    "prompt": prompts,
    "rubric": rubrics,
    "ground_truth": ground_truths,
})
print(f"Dataset: {len(dataset)} samples, prompt length sample: {len(dataset[0]['prompt'])} chars")

# %% Cell 4: Rubric scoring helpers (inlined for Colab portability)

def _extract_expected_value(criterion):
    m = re.search(r"\s+is\s+(.+)$", criterion)
    return m.group(1).strip().strip('"') if m else None

def _normalize_for_match(value):
    value = value.strip()
    variants = [value]
    no_commas = value.replace(",", "")
    if no_commas != value:
        variants.append(no_commas)
    if value.endswith("%"):
        num_part = value[:-1].strip()
        variants.extend([num_part, f"{num_part}%", f"{num_part} percent"])
        if "." in num_part and num_part.endswith("0"):
            variants.append(num_part.rstrip("0").rstrip("."))
    if value.startswith("$"):
        variants.append(value[1:].strip())
        variants.append(value[1:].replace(",", ""))
    if "%" in value and "." in value:
        num_part = value.replace("%", "").strip()
        try:
            f = float(num_part)
            if f == int(f):
                variants.append(str(int(f)))
        except ValueError:
            pass
    return list(dict.fromkeys(variants))

def _answer_contains_value(answer, expected):
    answer_lower = answer.lower()
    for v in _normalize_for_match(expected):
        if v and v.lower() in answer_lower:
            return True
    return False

def score_answer(answer, rubric):
    if not rubric:
        return 100.0, []
    results = []
    for criterion in rubric:
        expected = _extract_expected_value(criterion)
        if expected is None:
            key = criterion.replace("States that ", "").strip()
            passed = key.lower() in answer.lower()
        else:
            passed = _answer_contains_value(answer, expected)
        results.append((criterion, passed))
    passed_count = sum(1 for _, p in results if p)
    return round((passed_count / len(rubric)) * 100.0, 1), results

# %% Cell 5: Define 3 GDPO reward functions

def _get_text(completions):
    texts = []
    for c in completions:
        if isinstance(c, list) and c:
            texts.append(c[-1].get("content", ""))
        elif isinstance(c, str):
            texts.append(c)
        else:
            texts.append("")
    return texts

def _get_answer(text):
    return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()

def reward_correctness(completions, **kwargs):
    """Signal 1: Rubric score (0.0-1.0)."""
    texts = _get_text(completions)
    rubric_strs = kwargs.get("rubric", [])
    rewards = []
    for i, text in enumerate(texts):
        answer = _get_answer(text)
        try:
            rubric = json.loads(rubric_strs[i]) if i < len(rubric_strs) else []
        except (json.JSONDecodeError, TypeError):
            rubric = []
        if not rubric:
            rewards.append(0.0)
            continue
        sc, _ = score_answer(answer, rubric)
        rewards.append(sc / 100.0)
    return rewards

def reward_format(completions, **kwargs):
    """Signal 2: Format compliance (0.0, 0.5, or 1.0)."""
    texts = _get_text(completions)
    rewards = []
    for text in texts:
        has_prefix = "answer:" in text.lower()
        has_number = bool(re.search(r"\d+\.?\d*", text))
        good_len = 50 <= len(text) <= 3000
        if has_prefix and has_number and good_len:
            rewards.append(1.0)
        elif has_number and good_len:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def reward_completeness(completions, **kwargs):
    """Signal 3: Ground-truth value coverage (0.0-1.0)."""
    texts = _get_text(completions)
    gt_strs = kwargs.get("ground_truth", [])
    rubric_strs = kwargs.get("rubric", [])
    rewards = []
    for i, text in enumerate(texts):
        answer = _get_answer(text).lower()
        gt_values = []
        try:
            gt_values = json.loads(gt_strs[i]) if i < len(gt_strs) else []
        except (json.JSONDecodeError, TypeError):
            pass
        if not gt_values:
            try:
                rubric = json.loads(rubric_strs[i]) if i < len(rubric_strs) else []
            except (json.JSONDecodeError, TypeError):
                rubric = []
            for c in rubric:
                m = re.search(r"\s+is\s+(.+)$", c)
                if m:
                    gt_values.append(m.group(1).strip().strip('"'))
        if not gt_values:
            rewards.append(0.0)
            continue
        hits = sum(1 for v in gt_values
                   if v.lower() in answer or
                   v.lower().replace(",", "") in answer.replace(",", ""))
        rewards.append(round(hits / len(gt_values), 3))
    return rewards

print("Reward functions defined: correctness, format, completeness")

# %% Cell 6: Before-training eval

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "unsloth/Qwen3-4B"

def run_eval(model_path, label="Eval"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    seen, unique = set(), []
    for t in tasks:
        if t["task_id"] not in seen:
            seen.add(t["task_id"])
            unique.append(t)

    total_passed, total_criteria = 0, 0
    print(f"\n--- {label}: {model_path} ---")
    for t in unique:
        data_summary = get_data_summary(t["world_path"])
        prompt = f"{SYSTEM}\n\n# Company Data\n{data_summary}\n\n# Task\n{t['prompt']}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, temperature=0.7,
                do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answer = _get_answer(text)
        rubric = t.get("rubric", [])
        sc, res = score_answer(answer, rubric)
        p = sum(1 for _, x in res if x)
        total_passed += p
        total_criteria += len(res)
        print(f"  {t['task_id']}  {p}/{len(res)}  {sc:.1f}%")

    overall = (total_passed / total_criteria * 100) if total_criteria > 0 else 0
    print(f"  OVERALL: {total_passed}/{total_criteria}  {overall:.1f}%")
    del model
    torch.cuda.empty_cache()
    return overall

before_score = run_eval(MODEL_NAME, "BEFORE training")

# %% Cell 7: Train with GRPO + 3 reward signals

from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    def __init__(self):
        self._start = time.time()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        elapsed = time.time() - self._start
        step, total = state.global_step, state.max_steps
        rw = "".join(f"  {k}={logs[k]:.4f}" for k in sorted(logs) if "reward" in k.lower() or "loss" in k.lower())
        print(f"[Step {step}/{total} ({step/total*100:.0f}%) | {elapsed:.0f}s]{rw}", flush=True)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    reward_funcs=[reward_correctness, reward_format, reward_completeness],
    train_dataset=dataset,
    args=GRPOConfig(
        output_dir="./harfeast_checkpoints",
        num_train_epochs=1,
        num_generations=4,
        max_completion_length=512,
        max_prompt_length=4096,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_steps=50,
        bf16=True,
    ),
    callbacks=[LoggingCallback()],
)

print("\nTraining with 3 GDPO reward signals...")
trainer.train()
trainer.save_model("./harfeast_checkpoints")

# %% Cell 8: After-training eval + comparison

after_score = run_eval("./harfeast_checkpoints", "AFTER training")

print(f"\n{'='*50}")
print(f"  RESULTS")
print(f"{'='*50}")
print(f"  Before training: {before_score:.1f}%")
print(f"  After training:  {after_score:.1f}%")
delta = after_score - before_score
print(f"  Improvement:     {'+' if delta >= 0 else ''}{delta:.1f}%")
print(f"{'='*50}")
