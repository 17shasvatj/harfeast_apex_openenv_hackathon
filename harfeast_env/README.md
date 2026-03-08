# HarFeast Environment

Management consulting RL environment for OpenEnv. Agents explore CSV data, text documents, run filters/aggregations, and submit answers scored by rubric.

## Actions (8)

- **files.list(path)** - List files in data/ or documents/
- **files.read(path)** - Read text documents
- **spreadsheet.read_range(file, range)** - Read CSV (columns, 1:10, all)
- **data.filter(dataset, column, operator, value)** - Filter rows
- **data.group_by(dataset, column, aggregation, target_column)** - Aggregate
- **data.add_columns(dataset, new_column, expression)** - Derived columns
- **data.compute(expression)** - Math calculator
- **submit(answer)** - Submit final answer; episode ends; rubric scores 0-100

## Action format (JSON)

```json
{"action": "files.list", "path": "."}
{"action": "data.filter", "dataset": "employee_survey.csv", "column": "training_received", "operator": "eq", "value": "Yes"}
{"action": "submit", "answer": "The count is 1202. Excellent: 14%, Good: 41%..."}
```

## Usage

```python
from harfeast_env import HarFeastEnv, HarFeastAction
import json

# Connect to HF Space
client = HarFeastEnv(base_url="https://YOUR-USERNAME-harfeast-env.hf.space")

# Reset (load task)
result = client.reset()
print(result.observation.observation)  # Task prompt

# Step - send action as JSON string
action = HarFeastAction(action_json=json.dumps({"action": "files.list", "path": "."}))
result = client.step(action)
print(result.observation.observation)
print(result.reward, result.done)

client.close()
```

## Local run

```bash
cd /path/to/harfeast_apex_openenv_hackathon
python -m uvicorn harfeast_env.server.app:app --host 0.0.0.0 --port 8000
```

Then: `HarFeastEnv(base_url="http://localhost:8000")`
