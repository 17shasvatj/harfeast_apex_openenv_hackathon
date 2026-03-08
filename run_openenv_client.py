#!/usr/bin/env python3
"""
Test HarFeast OpenEnv client (local server).
Run server first: python -m uvicorn harfeast_env.server.app:app --host 0.0.0.0 --port 8000
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use local server
from harfeast_openenv import HarFeastOpenEnv


def test_local():
    """Test with local HarFeastOpenEnv (no HTTP)."""
    env = HarFeastOpenEnv()
    r = env.reset(task_id="task_14")
    print("Reset:", r.observation[:200], "...")
    r = env.step({"action": "files.list", "path": "."})
    print("Step 1:", r.observation[:150])
    r = env.step({"action": "data.filter", "dataset": "employee_survey.csv", "column": "training_received", "operator": "eq", "value": "Yes"})
    print("Step 2:", r.observation)
    gt = env._task["ground_truth"]
    answer = f"Training received: {gt['trained_count']}. " + " ".join(f"{k}: {v}%" for k,v in gt["quality_pcts"].items())
    r = env.step({"action": "submit", "answer": answer})
    print("Submit reward:", r.reward, "done:", r.done)
    print("OK - local env works")


if __name__ == "__main__":
    test_local()
