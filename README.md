# HarFeast

<p align="center">
  <img src="Mercor-logo.png" alt="Mercor" height="80">
</p>

**OpenEnv Hackathon 2026 — Mercor APEX-Agents + Scaler AI Labs Enterprise Workflows**

A synthetic RL environment where LLM agents learn to do management consulting work: explore enterprise data, use analytical tools, and produce verifiable deliverables.

**Live environment:** [openenv-community/harfeast-env](https://huggingface.co/spaces/openenv-community/harfeast-env)

---

## The Problem

Mercor's [APEX-Agents benchmark](https://arxiv.org/abs/2601.14242) ([dataset](https://huggingface.co/datasets/mercor/apex-agents)) gave us something very important — proof that AI agents are not capable of performing professional tasks that require multi-step reasoning and tool calling over long horizons. The best models struggle with tasks that a junior analyst handles in an afternoon: navigating fragmented data, chaining analytical steps where each depends on the last, and producing answers with exact numbers that hold up to verification.

The problem isn't knowledge — it's workflow. Current RL environments for tool use are too simple: single API calls, obvious databases, one-step answers. Real enterprise work requires sustained reasoning across messy, interconnected data systems. We need environments that teach agents this workflow, and we need to measure whether they actually learn it.

HarFeast is that environment.

---

## The Synthetic World

At the heart of HarFeast is a **procedural world generator** that creates realistic enterprise data ecosystems. Each world simulates a food manufacturing company with 5 plants across the Midwest.

### What Gets Generated

| Data Source | What It Contains | Scale |
|---|---|---|
| **Employee Survey** | Digital readiness, training status, comfort with technology, willingness to adopt ERP, role type, department | 2,400+ employees |
| **Equipment Records** | Machine types, failure rates, OEE metrics, maintenance schedules, scrap rates per plant | 250+ machines |
| **Quality & Scrap Losses** | Defect categories, scrap costs, predictive maintenance flags | 250 entries |
| **Plant Labor Data** | Hourly wages, overtime rates, productivity metrics by plant and shift | 64 records |
| **Interview Transcripts** | Qualitative insights from plant managers about operational challenges | 5 documents |
| **Industry Benchmarks** | OEE targets, digital transformation best practices | Reference docs |

### How Generation Works

The generator is fully parameterized. A single seed controls:

- **Employee distributions** — willingness scores, training completion rates, digital comfort levels all follow configurable distributions per plant
- **Equipment characteristics** — failure rates, OEE baselines, and scrap percentages vary by plant and equipment type
- **Wage structures** — hourly rates, overtime multipliers, and shift differentials across plants
- **Document content** — interview transcripts reference the actual generated data, so findings are internally consistent

Every seed produces different numbers but the same task structure. An agent trained on seed 42 can't memorize answers — when evaluated on seed 99, the data distributions shift and everything has to be recomputed. This forces the agent to learn the **analytical procedure**, not the output.

```bash
# Single world
python harfeast_synthetic_world_generator.py --seed 42 --output-dir ./harfeast_world

# Batch: 200 unique worlds for training diversity
python harfeast_synthetic_world_generator.py --batch 200 --output-dir ./harfeast_worlds
```

### Ground Truth

Every world comes with deterministic ground truth computed directly from the generated data. When a task asks "how many high-priority employees qualify for the ERP pilot?", the answer is computed by applying the exact criteria (willingness > threshold AND training_received == Yes AND digital_comfort >= level AND role in target set) to the generated employee data. There is one correct answer per seed, and the rubric checks it.

---

## The Environment

Built on [OpenEnv 0.2.1](https://github.com/OpenEnv-dev/openenv). The agent starts each episode seeing only a task prompt — it has no idea what data exists or where to find it.

### 8 Tools

| Tool | What It Does |
|---|---|
| `files.list` | Browse directories to discover available data |
| `files.read` | Read documents and interview transcripts |
| `spreadsheet.read_range` | Query CSV data with row/column ranges |
| `data.filter` | Filter datasets on conditions |
| `data.group_by` | Aggregate by categorical columns |
| `data.add_columns` | Compute derived metrics |
| `data.compute` | Run calculations over filtered data |
| `submit` | Deliver final analysis for rubric scoring |

The agent interacts turn by turn: it issues a tool call as JSON, gets back an observation, and decides what to do next. A typical successful trajectory looks like:

```
Turn 1:  files.list(".")           → discovers data/, documents/
Turn 2:  files.list("data/")       → finds employee_survey.csv, equipment.csv, ...
Turn 3:  spreadsheet.read_range("data/employee_survey.csv", rows="1:5")  → sees schema
Turn 4:  data.filter(dataset="employee_survey", condition="willing_to_adopt == 'Yes'")
Turn 5:  data.filter(dataset="filtered_0", condition="training_received == 'Yes'")
Turn 6:  data.group_by(dataset="filtered_1", by="plant", agg="count")
Turn 7:  submit(answer="52 employees across 5 plants qualify...")
```

### 14 Consulting Tasks

Each world has 14 tasks modeled on APEX-Agents professional scenarios:

- Workforce readiness analysis (cross-referencing 4+ employee criteria)
- Predictive maintenance scrap impact
- OEE projections with digital lever adjustments
- Labor cost and variance analysis across plants
- Technology investment ROI
- Training program quality assessment

Each task has a multi-criteria rubric — not just "right or wrong" but "did you identify the correct count AND the correct percentage AND the correct plant breakdown?" Partial credit for getting some criteria right.

---

## Training

We train with **multi-turn GRPO** (Group Relative Policy Optimization). For each task, we roll out K=16 parallel trajectories (up to 15 turns each). GRPO computes advantages across the group — trajectories that led to higher rubric scores get reinforced.

The reward uses decomposed signals inspired by [NVIDIA's GDPO](https://arxiv.org/abs/2501.12948):
- **Rubric correctness** — did the final answer match ground truth criteria?
- **Evidence gathering** — did the agent surface relevant data during exploration?
- **Format compliance** — were tool calls valid JSON?
- **Efficiency** — shorter successful trajectories score higher

```bash
python train_multiturn.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --world ./harfeast_world \
    --epochs 10 \
    --num-generations 16 \
    --max-turns 15 \
    --max-length 8192 \
    --max-new-tokens 768 \
    --no-think \
    --eval-before \
    --output-dir ./checkpoints
```

### Results

Trained on NVIDIA H200 (144 GB). Qwen2.5-7B-Instruct, 10 epochs, 38 gradient steps, ~5.3 hours.

**Overall: 9.8% → 20.7% (+10.9 percentage points, 2.1x improvement)**

| Task | Before | After | Change |
|---|---|---|---|
| High-Priority Digital Training | 0.0% | **25.0%** | +25.0 |
| Adjusted Cost of Instability | 0.0% | 0.0% | — |
| Predictive Maintenance Scrap | 16.7% | 16.7% | — |
| Digital Lever / OEE Projections | 0.0% | **9.1%** | +9.1 |
| Labor Cost Analysis | 0.0% | 0.0% | — |
| Operational Efficiency | 0.0% | **40.0%** | +40.0 |
| Productivity Loss | 0.0% | 0.0% | — |
| Equipment Quality Losses | 0.0% | 0.0% | — |
| Labor Variance Analysis | 0.0% | 0.0% | — |
| Productivity Loss (New Wages) | 50.0% | 50.0% | — |
| Technology Investment Impact | 0.0% | 0.0% | — |
| Digital Adoption Willingness | 50.0% | 50.0% | — |
| Frito-Lay Downtime Reduction | 100.0% | 100.0% | — |
| Training Quality Assessment | 0.0% | **60.0%** | +60.0 |

Four tasks went from zero to meaningful scores. The agent learned multi-step analytical workflows it couldn't perform at all before training — filtering, grouping, computing, and submitting with correct numbers.

![Reward Curve](plots_v4/reward_curve.png)

![Before vs After](plots_v4/before_after.png)

![Per-Task Heatmap](plots_v4/per_task_heatmap.png)

---

## Partner Sub-Themes

### Mercor — APEX-Agents

HarFeast directly addresses the Mercor sub-theme: *"Build an OpenEnv environment that captures a complex, real-world multi-step task and train an agent within it to improve performance on APEX-Agents."*

The environment models the same kind of professional consulting work that APEX-Agents evaluates — partially observable enterprise data, multi-step tool use, and rubric-scored deliverables with verifiable ground truth. We show that multi-turn GRPO training produces measurable improvement on these tasks: a 7B model that starts unable to navigate enterprise data learns to chain tool calls and produce correct analytical findings.

### Scaler AI Labs — Enterprise Workflows

HarFeast is a long-horizon, non-code business environment. The agent reasons across HR data, operations, and finance — exactly the kind of enterprise workflow that Scaler AI Labs' track targets. Tasks span sales-adjacent analysis (labor cost optimization), project management (digital transformation readiness), and HR (workforce training assessment). The procedural world generator creates unlimited training scenarios with different data distributions, enabling scalable evaluation of enterprise workflow agents.

### Statement 2 — Long-Horizon Planning

Tasks require 8-15 sequential tool calls with stateful dependencies. The agent must plan its exploration, track intermediate results across turns, and adapt its strategy based on what it discovers in the data.

### Statement 3.1 — World Modeling (Professional Tasks)

The environment is partially observable. The agent must build an internal model of the data landscape through exploration — discovering what files exist, reading schemas, understanding relationships between datasets. This is genuine causal reasoning, not pattern matching.

---

## Project Structure

```
├── harfeast_openenv/                # Core environment
│   ├── environment.py               # Multi-turn env with tool dispatch
│   ├── actions.py                   # 8 tool handlers
│   ├── rubric.py                    # Ground-truth rubric scoring
│   └── schemas.py                   # Action/Observation models
├── harfeast_env/server/app.py       # OpenEnv FastAPI server
├── harfeast_synthetic_world_generator.py  # Procedural data generation
├── train_multiturn.py               # Multi-turn GRPO training
├── train_harfeast_colab.py          # Colab-ready training script
├── harfeast_training.ipynb          # Training notebook
├── visualize_training.py            # Plot generation from training logs
├── eval_harfeast.py                 # Evaluation script
├── TRAINING_RESULTS.md              # Detailed run-by-run results
└── training.sh                      # Slurm batch script (H200)
```

---

Built during OpenEnv Hackathon, March 2026.
