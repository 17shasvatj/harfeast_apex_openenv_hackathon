# HarFeast Training Results

## Run 1: H200 — 6 Epochs (Baseline)

**Date:** March 9, 2026
**GPU:** NVIDIA H200 (144 GB)
**Model:** unsloth/Qwen3-4B
**Wall time:** 122 min (7,337 s)
**Gradient steps:** 12

### Configuration

| Parameter | Value |
|---|---|
| Epochs | 6 |
| Generations (K) | 16 |
| Max turns | 10 |
| Max length | 6,144 |
| Max new tokens | 512 |
| Learning rate | 5e-6 |
| Temperature | 0.8 |

### Overall Results

| | Score | Criteria passed |
|---|---|---|
| **Before** | **4.3%** | 4 / 92 |
| **After** | **13.0%** | 12 / 92 |
| **Delta** | **+8.7pp (3x)** | +8 |

### Per-Task Breakdown

| Task | Name | Before | After | Delta |
|---|---|---|---|---|
| task_01 | High-Priority Digital Training Employees | 0.0% | 0.0% | — |
| task_02 | Adjusted Cost of Instability | 0.0% | 0.0% | — |
| task_03 | Predictive Maintenance Scrap Impact | 0.0% | 0.0% | — |
| task_04 | Digital Lever Agreement and OEE Projections | 9.1% | 9.1% | — |
| task_05 | Labor Cost Analysis | 0.0% | 0.0% | — |
| task_06 | Operational Efficiency Analysis | 0.0% | **30.0%** | +30.0 |
| task_07 | Productivity Loss Quantification | 0.0% | 0.0% | — |
| task_08 | High-Priority Equipment Quality Losses | 0.0% | 0.0% | — |
| task_09 | Labor Variance Analysis | 0.0% | 0.0% | — |
| task_10 | Updated Productivity Loss with New Wages | 0.0% | 0.0% | — |
| task_11 | Technology Investment Impact | 0.0% | 0.0% | — |
| task_12 | Digital Adoption Willingness Analysis | 0.0% | **50.0%** | +50.0 |
| task_13 | Frito-Lay Downtime Reduction Application | 0.0% | **20.0%** | +20.0 |
| task_14 | Training Quality Assessment | 60.0% | **100.0%** | +40.0 |

### Epoch-by-Epoch Progress

| Epoch | Mean Reward | Signal Tasks | Nonzero / Total | Gradient Steps |
|---|---|---|---|---|
| 1 | 0.067 | 2/14 | 40/224 | 2 |
| 2 | 0.083 | 2/14 | 46/224 | 2 |
| 3 | 0.081 | 2/14 | 44/224 | 2 |
| 4 | 0.103 | 2/14 | 49/224 | 2 |
| 5 | 0.109 | 2/14 | 52/224 | 2 |
| 6 | 0.109 | 2/14 | 56/224 | 2 |

### Key Observations

1. **Clear upward trend** in mean reward: 0.067 → 0.109 (+63% relative improvement)
2. **Nonzero rewards increasing**: 40 → 56 out of 224 (more trajectories scoring above zero each epoch)
3. **New tasks unlocked**: task_06 (Operational Efficiency) emerged in Epoch 4 — the model learned a brand new skill mid-training
4. **task_14 mastered**: Went from 60% to 100% (5/5 criteria), with individual trajectory rewards going from avg 0.56 to frequently hitting 1.0
5. **Sparse gradient signal**: Only 2/14 tasks per epoch provide variance for GRPO. 12/14 tasks still produce uniform 0 reward across all K trajectories — more epochs needed for these harder tasks to crack
6. **task_04 converged**: All 16 trajectories consistently get exactly 1/11 criteria (9.1%) — deterministic behavior, no variance for learning

### A100-40GB Attempts

Both A100 runs (K=8 and K=4) hit CUDA OOM during backward pass. The 40GB model variant does not have enough memory for multi-turn GRPO with this model. H200 (144GB) used only ~8GB for model weights with ample headroom for K=16 trajectories.

---

## Run 2: H200 — Continued from Run 1 (FAILED — Catastrophic Forgetting)

**Date:** March 9-10, 2026
**GPU:** NVIDIA H200 (144 GB)
**Model:** Run 1 checkpoint (ckpt_h200)
**Config changes:** LR 8e-6 (was 5e-6), Temperature 0.9 (was 0.8), 20 epochs
**Outcome:** Model collapsed. Cancelled at epoch 11.

### What Happened

| Epoch | Mean Reward | Notes |
|---|---|---|
| 1 | 0.111 | Started at Run 1 level, task_14 still producing 0.8-1.0 |
| 2 | 0.020 | **Collapse.** task_14 dropped to 0.0, task_12 dropped to 0.0 |
| 3 | 0.007 | Worst point. Only task_13 (0.20) still producing signal |
| 4-10 | 0.019-0.024 | Flat. Never recovered. |

### Root Cause

1. **Learning rate too high (8e-6).** Run 1 was stable at 5e-6. The 60% increase caused the model to overshoot and destroy learned behaviors.
2. **Temperature too high (0.9).** More randomness → more bad trajectories → noisier gradients.
3. **Qwen3 thinking mode.** The model generates `<think>...</think>` tags consuming 100s of tokens before producing any tool call. After 10 turns of thinking, `force_submit` auto-submits garbage → 0 reward. Our `THINK_SKIP` hack is fragile and gets ignored.

### Lesson: The Qwen3 Thinking Problem

12/14 tasks consistently get 0.0 reward across ALL 16 trajectories. The model uses all its tokens on thinking instead of tool calls. GRPO needs reward variance to learn — with all-zero rewards, there is zero gradient signal.

---

## Run 3 (v4): H200 — Qwen2.5-7B-Instruct (COMPLETED)

**Date:** March 10, 2026
**GPU:** NVIDIA H200 (144 GB)
**Model:** Qwen/Qwen2.5-7B-Instruct (fresh start, thinking mode disabled)
**Job:** 4966541
**Wall time:** 321 min (19,248 s)
**Gradient steps:** 38

### Why Qwen2.5-7B

- **No thinking mode** — outputs JSON directly, no wasted tokens on `<think>` tags
- **Native JSON/tool-calling** — trained for structured output generation
- **7B params** — more capacity than 4B, H200 has 144GB headroom
- **Same architecture family** — no code changes needed

### Configuration

| Parameter | Value | vs Run 1 |
|---|---|---|
| Epochs | 10 | +4 |
| Generations (K) | 16 | same |
| Max turns | 15 | +5 (more exploration room) |
| Max length | 8,192 | +2,048 (bigger context) |
| Max new tokens | 768 | +256 (longer responses) |
| Learning rate | 5e-6 | same (stable) |
| Temperature | 0.7 | -0.1 (more focused) |

### Overall Results

| | Score | Criteria Passed |
|---|---|---|
| **Before** | **9.8%** | 9 / 92 |
| **After** | **20.7%** | 19 / 92 |
| **Delta** | **+10.9pp (2.1x)** | +10 |

### Per-Task Breakdown

| Task | Name | Before | After | Delta |
|---|---|---|---|---|
| task_01 | High-Priority Digital Training Employees | 0.0% | **25.0%** | +25.0 |
| task_02 | Adjusted Cost of Instability | 0.0% | 0.0% | — |
| task_03 | Predictive Maintenance Scrap Impact | 16.7% | 16.7% | — |
| task_04 | Digital Lever Agreement and OEE Projections | 0.0% | **9.1%** | +9.1 |
| task_05 | Labor Cost Analysis | 0.0% | 0.0% | — |
| task_06 | Operational Efficiency Analysis | 0.0% | **40.0%** | +40.0 |
| task_07 | Productivity Loss Quantification | 0.0% | 0.0% | — |
| task_08 | High-Priority Equipment Quality Losses | 0.0% | 0.0% | — |
| task_09 | Labor Variance Analysis | 0.0% | 0.0% | — |
| task_10 | Updated Productivity Loss with New Wages | 50.0% | 50.0% | — |
| task_11 | Technology Investment Impact | 0.0% | 0.0% | — |
| task_12 | Digital Adoption Willingness Analysis | 50.0% | 50.0% | — |
| task_13 | Frito-Lay Downtime Reduction Application | 100.0% | 100.0% | — |
| task_14 | Training Quality Assessment | 0.0% | **60.0%** | +60.0 |

### Epoch-by-Epoch Progress

| Epoch | Mean Reward | Signal Tasks | Nonzero / Total | Gradient Steps |
|---|---|---|---|---|
| 1 | 0.058 | 7/14 | 38/224 | 7 |
| 2 | 0.156 | 5/14 | 82/224 | 5 |
| 3 | 0.181 | 7/14 | 94/224 | 7 |
| 4 | 0.187 | 5/14 | 95/224 | 5 |
| 5 | 0.227 | 5/14 | 114/224 | 5 |
| 6 | 0.210 | 2/14 | 107/224 | 2 |
| 7 | 0.213 | 1/14 | 109/224 | 1 |
| 8 | 0.211 | 2/14 | 108/224 | 2 |
| 9 | 0.218 | 3/14 | 111/224 | 3 |
| 10 | 0.233 | 1/14 | 118/224 | 1 |

### Key Observations

1. **Strong upward trend in mean reward**: 0.058 → 0.233 (+4x relative improvement over 10 epochs). Peak at epoch 5 (0.227) then plateau with slight oscillation — typical of GRPO convergence.

2. **4 tasks unlocked from zero**:
   - task_01 (Workforce Readiness): 0% → 25% — agent learned to filter employees by training status
   - task_04 (OEE Projections): 0% → 9.1% — agent began reading agreement documents
   - task_06 (Operational Efficiency): 0% → 40% — one of the largest single-task improvements; agent learned multi-step filter → group_by → compute workflow
   - task_14 (Training Quality): 0% → 60% — agent mastered reading CSVs and matching criteria

3. **task_13 reached perfect score**: Frito-Lay Downtime task went from 100% baseline to consistent 1.0 reward across all 16 trajectories by epoch 6, with fewer turns needed (12 → 8). The model learned the optimal tool-call sequence and locked it in.

4. **Convergence behavior**: Signal tasks (tasks with reward variance) decreased from 7/14 in epoch 1 to 1/14 in epoch 10. This means 13/14 tasks converged to deterministic behavior — the model found a stable policy for each. Of these, 8 converged to a nonzero reward.

5. **Nonzero rewards doubled**: 38/224 (17%) in epoch 1 → 118/224 (53%) in epoch 10. More than half of all trajectories now produce positive reward.

6. **Loss dynamics**: GRPO loss showed healthy negative trend early (reward-increasing updates), with a notable -1.58 spike at step 31 (task_01 consolidation). Later epochs had smaller updates as the policy stabilized.

7. **Stubborn zero tasks**: task_02, task_05, task_07, task_09, task_11 remained at 0.0 across all epochs. These complex analytical tasks (labor cost analysis, productivity loss, technology investment) likely require more turns, longer context, or chained multi-step reasoning the model hasn't yet acquired.

8. **Qwen2.5-7B vs Qwen3-4B**: Switching from Qwen3-4B to Qwen2.5-7B-Instruct was transformative. Run 1 (Qwen3-4B) got only 2 signal tasks per epoch and 12/14 tasks stuck at zero. Run 3 (Qwen2.5-7B) got 7 signal tasks in epoch 1 alone. The elimination of the thinking mode was the single biggest unlock.

### Plots

All plots saved in `plots_v4/`:

- **reward_curve.png** — Per-step mean reward with 14-step moving average. Clear upward trend from ~0.05 to ~0.25, with individual steps occasionally hitting 0.6-1.0 (task_13, task_14).
- **loss_curve.png** — GRPO loss over 38 gradient steps. Mostly negative (reward-improving), with a large negative spike at step 31.
- **epoch_summary.png** — Left: mean reward climbing 0.058 → 0.233. Right: fraction of tasks with reward variance decreasing (convergence).
- **per_task_heatmap.png** — Heat map showing task_13 (dark red = 1.0) and task_14 (0.60) as the top performers. task_06 (0.40) and task_10 (0.50) also consistent.
- **reward_distribution.png** — Histogram: 976/2240 trajectories had nonzero reward. Bimodal distribution with peaks at 0.0 and 0.4-0.6.
- **before_after.png** — Side-by-side bar chart: overall 9.8% → 20.7%.
