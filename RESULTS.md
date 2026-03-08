# HarFeast Training Results

## Setup

| Parameter | Value |
|-----------|-------|
| **Model** | `unsloth/Qwen3-4B` |
| **Environment** | HarFeast OpenEnv (14 management consulting tasks) |
| **Training method** | GRPO with GDPO-style multi-signal rewards |
| **Reward signals** | correctness (rubric), format, completeness |
| **Training samples** | 128 (from 560 augmented tasks across 40 worlds) |
| **Epochs** | 1 |
| **Hardware** | Northeastern Explorer HPC — V100-SXM2 32GB |
| **Framework** | TRL GRPOTrainer + vLLM |

---

## Baseline (Before Training) — `unsloth/Qwen3-4B`

**Overall: 4/90 criteria passed (4.4%)**

| Task | Name | Passed | Score |
|------|------|--------|-------|
| task_01 | High-Priority Digital Training Employee | 2/8 | 25.0% |
| task_02 | Adjusted Cost of Instability | 0/5 | 0.0% |
| task_03 | Predictive Maintenance Scrap Impact | 0/6 | 0.0% |
| task_04 | Digital Lever Agreement and OEE Project | 0/11 | 0.0% |
| task_05 | Labor Cost Analysis | 0/9 | 0.0% |
| task_06 | Operational Efficiency Analysis | 0/8 | 0.0% |
| task_07 | Productivity Loss Quantification | 0/9 | 0.0% |
| task_08 | High-Priority Equipment Quality Losses | 0/2 | 0.0% |
| task_09 | Labor Variance Analysis | 1/6 | 16.7% |
| task_10 | Updated Productivity Loss with New Wages | 0/2 | 0.0% |
| task_11 | Technology Investment Impact | 0/10 | 0.0% |
| task_12 | Digital Adoption Willingness Analysis | 0/4 | 0.0% |
| task_13 | Frito-Lay Downtime Reduction Application | 1/5 | 20.0% |
| task_14 | Training Quality Assessment | 0/5 | 0.0% |

> Note: Two baseline runs were conducted. Run 1 scored 2.2% (2/90), Run 2 scored 4.4% (4/90).
> Variation is expected due to stochastic sampling (temperature=0.7, top_p=0.9).

---

## After Training

*Training in progress...*

---

## Comparison

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Overall score | 4.4% | — | — |
| Criteria passed | 4/90 | — | — |

---

## Run Log

### Run 1 — Baseline eval only (first model download)
- **Date**: 2026-03-08 ~07:05 ET
- **Result**: 2/90 = 2.2%
- **Notes**: Cold model download (~30s). Hit Keras/TF import error before GRPO training started.

### Run 2 — Baseline eval only (cached model)
- **Date**: 2026-03-08 ~07:15 ET
- **Result**: 4/90 = 4.4%
- **Notes**: Model cached. Same Keras/TF error blocked GRPO. Fix: `export USE_TF=0`.
