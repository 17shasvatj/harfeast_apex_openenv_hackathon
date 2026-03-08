# HarFeast OpenEnv Environment – Implementation Plan

This plan outlines how to build an OpenEnv environment that combines the HarFeast synthetic world generator with the 8-action agent API for management consulting tasks.

---

## 1. Overview

**Input:** HarFeast world generator output (`./harfeast_world/`):
- `data/` – CSV files (employee_survey.csv, equipment_data.csv, etc.)
- `documents/` – Text files (scrap report, interviews, Frito-Lay case, Aptean report)
- `tasks.json` – Task prompts, ground truth, rubrics
- `ground_truth.json` – Computed answers (for rubric scoring)

**Action Space:** 8 actions the agent can call to explore data, compute, and submit.

**Episode Flow:** `reset(task_id)` → agent calls actions → `submit(answer)` → rubric scores → episode ends.

---

## 2. Action Space → Implementation Mapping

| Action | Purpose | Implementation |
|--------|---------|----------------|
| **files.list(path)** | Discover available data | List files in `data/` and `documents/`. Path can be `"."`, `"data"`, `"documents"`, or specific subpath. Return JSON with file names and types. |
| **files.read(path)** | Read text documents | Load `documents/*.txt` (interviews, scrap report, Frito-Lay case, Aptean). Return full content. Reject CSV paths (agent should use `spreadsheet.read_range`). |
| **spreadsheet.read_range(file, range)** | Read CSV rows | Load CSV from `data/`. Range: `"1:10"` (rows 1–10), `"all"` (everything), `"columns"` (headers only). Return formatted table (e.g., markdown or JSON). |
| **data.filter(dataset, column, operator, value)** | Filter rows | Operators: `eq`, `neq`, `gt`, `lt`, `gte`, `lte`, `contains`. Load dataset (CSV or `filtered_N`), apply filter, store as `filtered_0`, `filtered_1`, … in session state. Return row count. |
| **data.group_by(dataset, column, aggregation, target_column)** | Group & aggregate | Aggregations: `sum`, `mean`, `median`, `count`, `min`, `max`. Load dataset, group by column, aggregate target_column. Return result table (no storage for chaining unless you add it). |
| **data.add_columns(dataset, new_column, expression)** | Derived column | Expression references columns with basic arithmetic (e.g., `hours_manual_entry + hours_searching_data + hours_fixing_errors`). Create new dataset `filtered_N` or `derived_N` with extra column. |
| **data.compute(expression)** | Simple calculator | Parse math expression (numbers, +, -, *, /, (), etc.). Evaluate and return result. No access to data—agent must have numbers in context. |
| **submit(answer)** | Terminal action | Store answer text, compute rubric score, set `done=True`, end episode. |

---

## 3. Environment Architecture

### 3.1 Core Components

```
HarFeastOpenEnv
├── reset(task_id=None, seed=None, **kwargs)
│   └── Load world from path, pick task (or task_id), set prompt as initial observation
├── step(action: HarFeastAction)
│   └── Parse action, dispatch to handler, return observation + reward (0 until submit)
├── state
│   └── Current task, prompt, filtered datasets, step count, submitted answer, rubric score
```

### 3.2 Action Schema

Actions can be represented as structured calls, e.g.:

```python
# Option A: Dict/JSON
{"action": "files.list", "path": "."}
{"action": "files.read", "path": "documents/interview_sarah_jenkins.txt"}
{"action": "spreadsheet.read_range", "file": "data/employee_survey.csv", "range": "columns"}
{"action": "data.filter", "dataset": "employee_survey.csv", "column": "training_received", "operator": "eq", "value": "Yes"}
{"action": "data.group_by", "dataset": "employee_survey.csv", "column": "plant", "aggregation": "mean", "target_column": "hours_manual_entry"}
{"action": "data.add_columns", "dataset": "employee_survey.csv", "new_column": "inefficient_hours", "expression": "hours_manual_entry + hours_searching_data + hours_fixing_errors"}
{"action": "data.compute", "expression": "(37.5 - 8.5) / 8.5 * 100"}
{"action": "submit", "answer": "The high-priority count is 127..."}
```

### 3.3 Observation Structure

After each `step`, the agent receives:

```python
{
    "observation": "...",      # Action output (file list, table, number, confirmation)
    "prompt": "...",           # Task prompt (included in initial obs, repeated if useful)
    "step_count": int,
    "done": bool,              # True after submit
    "reward": float,           # 0 until submit; then rubric score
    "info": {
        "action_taken": str,
        "datasets_available": ["filtered_0", "filtered_1", ...],  # For chaining
        "last_error": str | None  # If action failed
    }
}
```

---

## 4. Rubric Scoring Integration

From `tasks.json`, each task has a `rubric` list, e.g.:

```json
[
  "States that the number of high-priority employees is 127",
  "States that the percentage of all employees the high-priority employees represent is 4.2%",
  ...
]
```

**Scoring logic:**
- On `submit(answer)`, compare `answer` text against each rubric criterion.
- Use flexible matching: regex, substring, or LLM-assisted extraction for numeric values.
- Score = (criteria_met / total_criteria) * 100 or similar.
- Store in `state` and return as `reward` in the final step.

---

## 5. File Structure

```
harfeast_apex_openenv_hackathon/
├── harfeast_synthetic_world_generator.py   # Existing
├── harfeast_openenv/
│   ├── __init__.py
│   ├── environment.py       # HarFeastOpenEnv class
│   ├── actions.py           # Action handlers (files, spreadsheet, data)
│   ├── rubric.py            # Rubric scoring logic
│   └── schemas.py           # Action/Observation dataclasses
├── harfeast_world/          # Generated by world generator (gitignore or committed)
│   ├── data/
│   ├── documents/
│   ├── tasks.json
│   └── ground_truth.json
├── run_environment.py       # CLI to test: reset, step with sample actions
├── requirements.txt
└── OPENENV_PLAN.md          # This file
```

---

## 6. Implementation Phases

### Phase 1: Core Environment Shell ✅
- [x] Create `harfeast_openenv/` package.
- [x] Implement `reset()` and `step()` with stub handlers.
- [x] Support `files.list` and `files.read` only.
- [x] Validate against a single task (e.g., Task 14 – training quality).
- [x] CLI: `python run_environment.py --task task_14`.

### Phase 2: Spreadsheet & Data Actions
- [ ] Implement `spreadsheet.read_range` with CSV loading.
- [ ] Implement `data.filter` with operator support and session storage.
- [ ] Implement `data.group_by` with aggregation functions.
- [ ] Implement `data.add_columns` with expression parsing (safe eval or simple parser).
- [ ] Implement `data.compute` (restricted eval for math only).
- [ ] Validate with tasks that require filtering/aggregation (e.g., Task 1, 6, 7).

### Phase 3: Submit & Rubric
- [ ] Implement `submit(answer)`.
- [ ] Rubric scorer: parse answer vs criteria (regex/LLM).
- [ ] Return reward and `done=True` on submit.
- [ ] Validate end-to-end: agent-style trajectory → submit → score.

### Phase 4: OpenEnv Compatibility
- [ ] Align with OpenEnv `Environment` base class if provided by hackathon.
- [ ] Expose HTTP/Docker API if required.
- [ ] Add `state()` property, `observation_space`, `action_space` descriptors.
- [ ] Documentation and example agent loop.

---

## 7. Action-Specific Notes

### files.list
- Paths: `"."`, `"data"`, `"documents"`, or `"data/employee_survey.csv"` (parent dir listing).
- Return: `{"files": ["employee_survey.csv", "equipment_data.csv", ...], "path": "data"}`.

### files.read
- Only allow `documents/*.txt`. Reject `data/*.csv` with a clear message to use `spreadsheet.read_range`.

### spreadsheet.read_range
- `file` can be `"employee_survey.csv"` or `"data/employee_survey.csv"` (normalize to `data/`).
- `range`: `"columns"` → headers; `"1:10"` → rows 1–10; `"all"` → all rows (consider row limit for large CSVs).

### data.filter
- `dataset` can be a CSV filename or `filtered_N` from a previous filter.
- Store result as next available `filtered_N`. Return: `{"rows": N, "stored_as": "filtered_2"}`.

### data.group_by
- If chaining is needed, optionally store as `grouped_N` for use in subsequent filters.

### data.add_columns
- Parse expression safely (whitelist: column names, +, -, *, /, (, )). No arbitrary Python.

### submit
- Validate that at least one non-submit action was taken (optional).
- Truncate very long answers if needed for rubric parsing.

---

## 8. Dependencies

```
# requirements.txt (example)
pandas>=2.0
openenv  # If hackathon provides a package; otherwise implement from spec
```

---

## 9. Example Agent Trajectory (Task 1)

1. `files.list(".")` → see `data/`, `documents/`, `tasks.json`
2. `files.list("data")` → see CSVs
3. `spreadsheet.read_range("employee_survey.csv", "columns")` → understand schema
4. `data.filter("employee_survey.csv", "willing_to_pilot", "eq", "Yes")` → filtered_0
5. (Chain more filters for readiness, training days, comfort, etc.)
6. `data.group_by("filtered_0", "role_type", "count", "employee_id")` → get counts by role type
7. `data.compute("127 / 3000 * 100")` → 4.23%
8. `submit("The high-priority count is 127 (4.2% of employees). Total inefficient hours...")` → rubric score, done.

---

## 10. Open Questions

1. **OpenEnv base class:** Does the hackathon provide a Python base class or only an HTTP spec?
2. **Action format:** JSON strings from the agent, or structured objects?
3. **Observation format:** Plain text vs structured JSON for the agent’s observation?
4. **Rubric matching:** Strict string match, regex, or LLM-based extraction?
5. **Step limit:** Max steps before forced termination (e.g., 100)?

---

## Next Steps

1. Confirm OpenEnv hackathon spec (API, base class, deployment).
2. Implement Phase 1 (files.list, files.read, basic reset/step).
3. Add spreadsheet and data actions (Phase 2).
4. Integrate rubric scoring (Phase 3).
5. Run and tune on all 14 HarFeast tasks.
