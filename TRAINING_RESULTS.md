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

## Run 2: H200 — Continued Training (pending)

Resuming from Run 1 checkpoint with 20 additional epochs, higher temperature (0.9) for more exploration.
