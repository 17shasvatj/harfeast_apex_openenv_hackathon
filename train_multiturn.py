#!/usr/bin/env python3
"""
HarFeast Multi-Turn GRPO Training (Batched)

The model interacts with the HarFeast environment over multiple turns,
issuing tool calls and receiving real observations. Generation is batched
across K trajectories per task for GPU efficiency.

Usage:
  python train_multiturn.py --model unsloth/Qwen3-4B --world ./harfeast_world \
      --epochs 3 --num-generations 8 --output-dir ./checkpoints_mt
"""

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
import copy

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*vLLM.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPT = """\
You are a management consultant analyzing data for HarFeast, a food manufacturing company.

You have these tools (output ONLY one JSON object per turn, no other text):

{"action": "files.list", "path": "."}
{"action": "files.read", "path": "documents/report.txt"}
{"action": "spreadsheet.read_range", "file": "data.csv", "range": "1:10"}
{"action": "data.filter", "dataset": "data.csv", "column": "col", "operator": "eq", "value": "x"}
{"action": "data.group_by", "dataset": "data.csv", "column": "col", "aggregation": "sum", "target_column": "val"}
{"action": "data.add_columns", "dataset": "data.csv", "new_column": "total", "expression": "a + b"}
{"action": "data.compute", "expression": "15000 * 0.12"}
{"action": "submit", "answer": "your final answer with specific numbers"}

Operators for data.filter: eq, neq, gt, lt, gte, lte, contains
Aggregations for data.group_by: sum, mean, median, count, min, max
Range for spreadsheet.read_range: "columns" (headers), "1:10" (rows), "all"

Strategy: explore files -> read relevant CSVs -> filter/compute -> submit with numbers.
Output ONLY a JSON object. No explanation text."""


def extract_json_action(text):
    """Parse a JSON action dict from model output, handling Qwen3 thinking tags."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    for pattern in [r'\{[^{}]*"action"\s*:\s*"[^"]+?"[^{}]*\}', r"\{[^{}]+\}"]:
        match = re.search(pattern, text)
        if match:
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict) and "action" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

    return None


THINK_SKIP = "<think>\n</think>\n"


def batched_rollout(model, tokenizer, world_path, task_id, K=8,
                    max_turns=10, temperature=0.8, max_new_tokens=512,
                    max_length=6144, force_submit=True):
    """
    Run K trajectories in PARALLEL using batched model.generate().
    Each trajectory gets its own environment instance.
    Thinking mode is disabled for Qwen3 by injecting closing think tags.
    Returns: (list_of_messages, list_of_rewards, list_of_turns)
    """
    import torch
    from harfeast_openenv.environment import HarFeastOpenEnv

    envs = [HarFeastOpenEnv(world_path=world_path) for _ in range(K)]
    all_messages = []
    all_rewards = [0.0] * K
    all_done = [False] * K
    all_turns = [0] * K

    for i in range(K):
        obs = envs[i].reset(task_id=task_id)
        all_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs.observation},
        ])

    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for turn in range(max_turns):
        active = [i for i in range(K) if not all_done[i]]
        if not active:
            break

        prompts = []
        for i in active:
            p = tokenizer.apply_chat_template(
                all_messages[i], tokenize=False, add_generation_prompt=True
            )
            p += THINK_SKIP
            prompts.append(p)

        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]

        for batch_idx, i in enumerate(active):
            new_tokens = outputs[batch_idx][input_len:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            action_dict = extract_json_action(new_text)
            if action_dict is None:
                all_messages[i].append({"role": "assistant", "content": new_text})
                all_messages[i].append({
                    "role": "user",
                    "content": "Invalid action. Output a single JSON object with an 'action' key.",
                })
                all_turns[i] = turn + 1
                continue

            step_result = envs[i].step(action_dict)
            all_messages[i].append({"role": "assistant", "content": json.dumps(action_dict)})
            all_messages[i].append({"role": "user", "content": step_result.observation})
            all_turns[i] = turn + 1

            if step_result.done:
                all_rewards[i] = step_result.reward / 100.0
                all_done[i] = True

    if force_submit:
        for i in range(K):
            if not all_done[i]:
                last_content = ""
                for msg in reversed(all_messages[i]):
                    if msg["role"] == "user" and msg["content"] != "Invalid action. Output a single JSON object with an 'action' key.":
                        last_content = msg["content"][:500]
                        break
                submit_action = {"action": "submit", "answer": last_content}
                step_result = envs[i].step(submit_action)
                all_messages[i].append({"role": "assistant", "content": json.dumps(submit_action)})
                all_messages[i].append({"role": "user", "content": step_result.observation})
                all_rewards[i] = step_result.reward / 100.0
                all_done[i] = True

    tokenizer.padding_side = orig_pad_side
    return all_messages, all_rewards, all_turns


def compute_turn_loss(model, tokenizer, messages, turn_index, max_length=6144):
    """Cross-entropy for one assistant turn."""
    import torch

    prefix_msgs = messages[:turn_index]
    full_msgs = messages[:turn_index + 1]

    prompt_text = tokenizer.apply_chat_template(
        prefix_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False)

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    full_ids = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids.to(model.device)

    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return None

    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    outputs = model(full_ids, labels=labels)
    if outputs.loss is None or torch.isnan(outputs.loss):
        return None
    return outputs.loss


def compute_trajectory_loss(model, tokenizer, messages, advantage, max_length=6144):
    """GRPO loss = advantage * mean(CE over assistant turns)."""
    import torch

    total_loss = torch.tensor(0.0, device=model.device)
    n = 0

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        loss = compute_turn_loss(model, tokenizer, messages, i, max_length=max_length)
        if loss is not None:
            total_loss = total_loss + loss
            n += 1

    if n == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    return advantage * (total_loss / n)


def run_eval(model, tokenizer, world_path, tasks, label="Eval", max_length=6144, max_new_tokens=512):
    """Evaluate with multi-turn interaction on all unique tasks."""
    import torch
    from harfeast_openenv.rubric import score_answer
    from harfeast_openenv.environment import HarFeastOpenEnv

    model.eval()
    seen, unique = set(), []
    for t in tasks:
        if t["task_id"] not in seen:
            seen.add(t["task_id"])
            unique.append(t)

    total_passed, total_criteria = 0, 0
    results = []

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for i, t in enumerate(unique):
        msgs_list, rewards, turns_list = batched_rollout(
            model, tokenizer, world_path, t["task_id"],
            K=1, max_turns=12, temperature=0.3,
            max_length=max_length, max_new_tokens=max_new_tokens,
        )

        env_check = HarFeastOpenEnv(world_path=world_path)
        msgs = msgs_list[0]
        reward = rewards[0]
        turns = turns_list[0]

        rubric = t.get("rubric", [])
        passed, total = 0, len(rubric)
        score = 0.0

        for msg in reversed(msgs):
            if msg["role"] == "assistant":
                action = extract_json_action(msg["content"])
                if action and action.get("action") == "submit":
                    answer = action.get("answer", "")
                    score, res = score_answer(answer, rubric)
                    passed = sum(1 for _, p in res if p)
                    break

        total_passed += passed
        total_criteria += total
        submitted = score > 0 or any(
            extract_json_action(m["content"]) and
            extract_json_action(m["content"]).get("action") == "submit"
            for m in msgs if m["role"] == "assistant"
        )
        sub_icon = "S" if submitted else "X"
        results.append({
            "task_id": t["task_id"],
            "task_name": t["task_name"],
            "passed": passed,
            "total": total,
            "score": score,
            "turns": turns,
            "submitted": submitted,
        })
        print(
            f"  [{i+1}/{len(unique)}] {t['task_id']}  "
            f"{passed}/{total}  {score:.1f}%  "
            f"({turns} turns, {sub_icon})"
        )

    overall = (total_passed / total_criteria * 100) if total_criteria > 0 else 0
    n_submitted = sum(1 for r in results if r["submitted"])
    print(f"\n  OVERALL: {total_passed}/{total_criteria}  {overall:.1f}%")
    print(f"  Submitted: {n_submitted}/{len(unique)}")
    print(f"{'='*60}")
    return round(overall, 1), results


def main():
    import torch

    parser = argparse.ArgumentParser(description="HarFeast Multi-Turn GRPO (Batched)")
    parser.add_argument("--model", default="unsloth/Qwen3-4B")
    parser.add_argument("--world", default="./harfeast_world")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-generations", type=int, default=2,
                        help="Trajectories per task (batched)")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=6144,
                        help="Max sequence length (context window)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max new tokens per generation")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--eval-before", action="store_true", default=True)
    parser.add_argument("--no-eval-before", dest="eval_before", action="store_false")
    parser.add_argument("--output-dir", default="./checkpoints_multiturn")
    args = parser.parse_args()

    start_time = time.time()
    world_path = os.path.abspath(args.world)

    # ── W&B ──
    use_wandb = True
    try:
        import wandb
        if os.environ.get("WANDB_API_KEY"):
            wandb.init(
                project="harfeast-grpo",
                name=f"mt-K{args.num_generations}-e{args.epochs}-lr{args.lr}",
                config=vars(args),
            )
            use_wandb = True
            print("W&B logging enabled")
        else:
            print("W&B: no API key found, logging disabled")
    except ImportError:
        print("W&B: not installed, logging disabled")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── Tasks ──
    tasks_path = os.path.join(args.world, "tasks.json")
    with open(tasks_path) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks from {args.world}")
    props = torch.cuda.get_device_properties(0)
    total_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB / {total_gb:.1f}GB ({props.name})")

    # ── Before-training eval ──
    before_score, before_results = None, None
    if args.eval_before:
        before_score, before_results = run_eval(
            model, tokenizer, world_path, tasks, "BEFORE training (multi-turn)",
            max_length=args.max_length, max_new_tokens=args.max_new_tokens,
        )
        if use_wandb:
            wandb.log({"eval/before_score": before_score})

    # ── Training ──
    K = args.num_generations
    print(f"\n{'='*60}")
    print(f"  MULTI-TURN GRPO TRAINING (BATCHED)")
    print(f"{'='*60}")
    print(f"  Model:        {args.model}")
    print(f"  Tasks:        {len(tasks)}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Generations:  {K} per task (batched generation)")
    print(f"  Max turns:    {args.max_turns}")
    print(f"  Max length:   {args.max_length}")
    print(f"  Max new tok:  {args.max_new_tokens}")
    print(f"  LR:           {args.lr}")
    print(f"  Temperature:  {args.temperature}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    global_step = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        random.shuffle(tasks)
        epoch_rewards = []
        steps_with_signal = 0

        for t_idx, task in enumerate(tasks):
            task_start = time.time()

            # ── Batched rollout (all K trajectories at once) ──
            model.eval()
            trajectories, rewards, turns_list = batched_rollout(
                model, tokenizer, world_path, task["task_id"],
                K=K, max_turns=args.max_turns, temperature=args.temperature,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            epoch_rewards.extend(rewards)

            # ── GRPO advantages ──
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = var_r**0.5 + 1e-8
            advantages = [(r - mean_r) / std_r for r in rewards]

            if var_r < 1e-10:
                task_time = time.time() - task_start
                print(
                    f"  [{t_idx+1}/{len(tasks)}] {task['task_id']}  "
                    f"r={mean_r:.3f} turns={[t for t in turns_list]}  "
                    f"no variance → skip  ({task_time:.0f}s)"
                )
                continue

            steps_with_signal += 1

            # ── Train on trajectories (gradient accumulation to avoid OOM) ──
            model.train()
            optimizer.zero_grad()

            n_valid = 0
            loss_sum = 0.0
            for traj, adv in zip(trajectories, advantages):
                loss = compute_trajectory_loss(model, tokenizer, traj, adv, max_length=args.max_length)
                if loss.requires_grad:
                    (loss / len(trajectories)).backward()
                    n_valid += 1
                    loss_sum += loss.item()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if n_valid > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1

            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            task_time = time.time() - task_start
            loss_val = loss_sum / n_valid if n_valid > 0 else 0.0
            print(
                f"  [{t_idx+1}/{len(tasks)}] {task['task_id']}  "
                f"r=[{', '.join(f'{r:.2f}' for r in rewards)}]  "
                f"adv_range=[{min(advantages):+.2f},{max(advantages):+.2f}]  "
                f"loss={loss_val:.4f}  step={global_step}  ({task_time:.0f}s)"
            )

            if use_wandb:
                wandb.log({
                    "train/loss": loss_val,
                    "train/mean_reward": mean_r,
                    "train/max_reward": max(rewards),
                    "train/reward_variance": var_r,
                    "train/task_time_s": task_time,
                    "train/step": global_step,
                    f"task_reward/{task['task_id']}": mean_r,
                })

        mean_r = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        nonzero_rewards = sum(1 for r in epoch_rewards if r > 0)
        print(
            f"\n  Epoch {epoch+1}: mean_reward={mean_r:.3f}, "
            f"signal_steps={steps_with_signal}/{len(tasks)}, "
            f"nonzero_rewards={nonzero_rewards}/{len(epoch_rewards)}, "
            f"global_step={global_step}"
        )

        if use_wandb:
            wandb.log({
                "epoch/mean_reward": mean_r,
                "epoch/signal_tasks": steps_with_signal,
                "epoch/nonzero_rewards_pct": nonzero_rewards / max(len(epoch_rewards), 1),
                "epoch/epoch": epoch + 1,
            })

        # Save checkpoint after each epoch
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        with open(os.path.join(ckpt_path, "train_state.json"), "w") as f:
            json.dump({"epoch": epoch+1, "global_step": global_step,
                        "mean_reward": mean_r, "signal_steps": steps_with_signal}, f, indent=2)
        print(f"  Checkpoint saved: {ckpt_path}\n")
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    # ── Save final ──
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nFinal model saved to {args.output_dir}")

    # ── After-training eval ──
    after_score, after_results = run_eval(
        model, tokenizer, world_path, tasks, "AFTER training (multi-turn)",
        max_length=args.max_length, max_new_tokens=args.max_new_tokens,
    )

    if use_wandb:
        wandb.log({"eval/after_score": after_score})
        if before_score is not None:
            wandb.log({"eval/delta": after_score - before_score})

    # ── Summary ──
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    if before_score is not None:
        print(f"  Before: {before_score:.1f}%")
    print(f"  After:  {after_score:.1f}%")
    if before_score is not None:
        delta = after_score - before_score
        print(f"  Delta:  {'+' if delta >= 0 else ''}{delta:.1f}%")
    print(f"  Time:   {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"{'='*60}")

    if before_results and after_results:
        print(f"\n{'Task':<10} {'Before':>10} {'After':>10} {'Delta':>10}")
        print("-" * 44)
        for rb, ra in zip(before_results, after_results):
            d = ra["score"] - rb["score"]
            sign = "+" if d >= 0 else ""
            print(
                f"{rb['task_id']:<10} {rb['score']:>8.1f}%  "
                f"{ra['score']:>8.1f}%  {sign}{d:>7.1f}%"
            )

    results_data = {
        "model": args.model,
        "training_type": "multi-turn GRPO (batched)",
        "epochs": args.epochs,
        "num_generations": args.num_generations,
        "max_turns": args.max_turns,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "before_score": before_score,
        "after_score": after_score,
        "before_results": before_results,
        "after_results": after_results,
        "total_time_s": round(total_time, 1),
    }
    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
