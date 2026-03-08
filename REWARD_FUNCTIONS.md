# Reward Functions

## Rank 1: GDPO with Decomposed Rubric Rewards (Most Ambitious, Best Fit)

This is the strongest option given H200 compute. Instead of collapsing your rubric into a single score, treat each rubric criterion as its own binary reward signal, plus a format reward and an efficiency reward. Use GDPO (NVIDIA's drop-in GRPO replacement from January 2026) to normalize each reward independently before aggregation.Here are five variations, ranked from strongest to weakest for your specific setup.

---

## Rank 1: GDPO with Decomposed Rubric Rewards

**Why it's the best fit:** Your rubric is already a multi-criteria checklist. GDPO was built exactly for this — it normalizes each reward signal independently so they don't collapse into identical advantage values. With H200 compute you can easily run the GDPO TRL implementation. This is also the most impressive thing to show judges: "We used NVIDIA's GDPO paper from January 2026 to handle our multi-signal rewards."

You decompose into 3–4 independent reward signals:

```python
def compute_rewards_gdpo(episode) -> list[float]:
    """Returns one reward per signal — GDPO normalizes each independently."""
    
    # Signal 1: Correctness (rubric score, 0-1)
    rubric_score = (episode["rubric_score"] or 0) / 100.0
    
    # Signal 2: Format compliance (did every action parse as valid JSON?)
    valid_actions = sum(1 for h in episode["history"] if h["success"])
    total_actions = max(len(episode["history"]), 1)
    format_reward = valid_actions / total_actions
    
    # Signal 3: Efficiency (inverse step count, rewarding shorter trajectories)
    max_steps = 20
    steps_used = len(episode["history"])
    efficiency_reward = max(0, 1.0 - (steps_used / max_steps))
    
    # Signal 4: Evidence gathering (did the agent use data actions?)
    data_actions = {"spreadsheet.read_range", "data.filter", "data.group_by"}
    used_data = any(h["action"].get("action") in data_actions for h in episode["history"])
    evidence_reward = 1.0 if used_data else 0.0
    
    return [rubric_score, format_reward, efficiency_reward, evidence_reward]
```

GDPO normalizes each of these four signals within the rollout group independently, then aggregates. This means the model gets a clear signal for "your answer was wrong but your format was good and you were efficient" versus "your answer was right but you wasted 18 steps." Standard GRPO would collapse these distinctions.

The TRL integration requires modifying about 20 lines in `grpo_trainer.py` — NVIDIA provides the exact diff. With H200 compute, you can run 16+ rollouts per prompt which gives GDPO even more distinct advantage groups to work with.

---

## Rank 2: Per-Criterion Verifiable Rewards

**Why it's strong:** Inspired by the ICLR 2026 "Rubric-as-Reward" paper. Instead of scoring the rubric as a single percentage, each rubric criterion becomes its own binary verifiable reward. A task with 5 rubric criteria gives you 5 separate binary rewards plus efficiency.

```python
def compute_per_criterion_rewards(episode, task) -> list[float]:
    """Each rubric criterion is a separate verifiable reward."""
    rubric = task.get("rubric", [])
    answer = episode.get("submitted_answer", "")
    
    rewards = []
    for criterion in rubric:
        expected = extract_expected_value(criterion)
        if expected and answer_contains_value(answer, expected):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    # Add efficiency reward
    steps = len(episode["history"])
    rewards.append(max(0, 1.0 - steps / 20))
    
    # Add format reward
    valid = sum(1 for h in episode["history"] if h["success"])
    rewards.append(valid / max(len(episode["history"]), 1))
    
    return rewards
```

This is pure RLVR — every reward is deterministic and verifiable. No LLM judge, no learned reward model. The advantage is that it gives extremely fine-grained signal. The model learns that getting criterion 3 right is independent from criterion 1. The downside is that the number of reward signals varies per task (some have 5 criteria, some have 3), which makes batching trickier with GDPO.

You could handle the variable count by padding to the max number of criteria and masking, or by grouping tasks with the same criterion count.

---

## Rank 3: LLM-as-Judge with Rubric Grounding

**Why it's worth considering with H200:** Your regex rubric scorer is brittle. An LLM judge can evaluate whether the answer actually addresses the criterion semantically, not just whether a number substring appears. With H200 compute, you can afford to run a judge model.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a frozen judge (separate from the policy model)
judge_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
judge_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def llm_judge_reward(answer: str, rubric: list[str], task_prompt: str) -> float:
    """Use a frozen LLM to score the answer against rubric criteria."""
    judge_prompt = f"""You are evaluating an analyst's answer to a data analysis task.

Task: {task_prompt[:500]}

Answer: {answer[:1000]}

Score each criterion as PASS or FAIL:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(rubric))}

Respond with only a JSON list of booleans, e.g. [true, false, true].
"""
    inputs = judge_tokenizer(judge_prompt, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=100, temperature=0.1)
    
    response = judge_tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        results = json.loads(response.strip())
        return sum(results) / len(results)
    except:
        # Fallback to regex scorer
        score, _ = score_answer(answer, rubric)
        return score / 100.0
```

The judge is frozen (never trained), so it provides a stable reward signal. The policy model (1.5B) learns to produce answers that satisfy the judge (7B). This is more robust than regex matching but costs ~5x more compute per reward evaluation. With H200, that's fine.

The risk is reward hacking — the policy model could learn to produce outputs that trick the judge without actually being correct. Grounding the judge with the actual rubric criteria and ground truth values mitigates this.

---

## Rank 4: Hybrid Verifiable + Partial Credit

**Why it's practical:** This is what I originally proposed, but refined. Pure verifiable rewards for numeric criteria, plus heuristic shaping for exploration. No LLM judge, no GDPO modification. Works with stock TRL GRPO.

```python
def compute_shaped_reward(episode, task) -> float:
    rubric_score = (episode["rubric_score"] or 0)
    history = episode["history"]
    gt = task.get("ground_truth", {})
    submitted = episode["submitted_answer"] is not None
    
    reward = 0.0
    
    # Primary: rubric score (0-100)
    reward += rubric_score if submitted else -20.0
    
    # Efficiency: -1 per step
    reward -= len(history) * 1.0
    
    # Exploration: +10 if used data actions
    data_actions = {"spreadsheet.read_range", "data.filter", "data.group_by"}
    if any(h["action"].get("action") in data_actions for h in history):
        reward += 10.0
    
    # Partial credit: ground truth values found in observations
    if gt and (not submitted or rubric_score < 100):
        all_obs = " ".join(h["observation"] for h in history)
        gt_values = extract_gt_values(gt)
        found = sum(1 for v in gt_values if str(v) in all_obs)
        partial = (found / max(len(gt_values), 1)) * 50.0
        existing = rubric_score or 0
        if partial > existing:
            reward += (partial - existing) * 0.5
    
    # Unique actions bonus
    action_strs = [json.dumps(h["action"], sort_keys=True) for h in history]
    if action_strs:
        reward += (len(set(action_strs)) / len(action_strs)) * 5.0
    
    return reward
```

This works, it's simple, and it requires no infrastructure changes. The downside is everything gets summed into one number, losing the multi-signal granularity that GDPO preserves. But it's the fastest to implement and most debuggable.

---

## Rank 5: Binary Outcome Reward (Simplest Baseline)

**Why it still matters:** Pure RLVR. Did the agent get the right answer or not? No shaping, no partial credit, no LLM judge. This is what DeepSeek-R1 used — and it worked.

```python
def compute_binary_reward(episode, task) -> float:
    rubric_score = episode["rubric_score"] or 0
    if rubric_score >= 80:
        return 1.0
    elif rubric_score >= 40:
        return 0.0
    else:
        return -1.0
```

Three-tier: good, neutral, bad. GRPO can differentiate between these across rollouts. The advantage is zero reward hacking risk — the signal is clean and verifiable. The disadvantage is it's extremely sparse and needs many rollouts to get variance. With H200 compute and 16+ rollouts per prompt, this might actually work surprisingly well because you'll get enough variance per group for GRPO to learn from.

This is also a great baseline to run alongside your main approach. If the shaped reward function only marginally beats the binary one, that tells you the shaping isn't adding much.