#!/usr/bin/env python3
"""
CLI to run and test the HarFeast OpenEnv environment.
Phase 1-3: files, spreadsheet, data actions, submit.

Usage:
    python run_environment.py --task task_14
    python run_environment.py --task task_14 --interactive
    python run_environment.py  # random task
"""

import argparse
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harfeast_openenv import HarFeastOpenEnv


def main():
    parser = argparse.ArgumentParser(description="Run HarFeast OpenEnv")
    parser.add_argument(
        "--task",
        default=None,
        help="Task ID (e.g. task_14). If omitted, picks random task.",
    )
    parser.add_argument(
        "--world",
        default=None,
        help="Path to harfeast_world directory. Default: ./harfeast_world",
    )
    parser.add_argument(
        "--worlds-base",
        default=None,
        help="Path to augmented dataset (harfeast_worlds with all_tasks.json). Samples across 200+ tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: prompt for actions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task selection.",
    )
    args = parser.parse_args()

    world_path = args.world or os.path.join(os.path.dirname(__file__), "harfeast_world")
    env = HarFeastOpenEnv(world_path=world_path, worlds_base=args.worlds_base)

    result = env.reset(task_id=args.task, seed=args.seed)
    print("=== RESET ===")
    print(f"Task: {env.state['task_id']} - {env.state['task_name']}")
    print(f"\n{result.observation}\n")

    if args.interactive:
        while not result.done:
            raw = input("Action (JSON or 'q' to quit): ").strip()
            if raw.lower() == "q":
                break
            try:
                action = json.loads(raw) if raw.startswith("{") else {"action": raw, "path": "."}
                result = env.step(action)
                print(f"\n--- Step {result.step_count} ---")
                print(result.observation)
                if result.info.get("last_error"):
                    print(f"[Error: {result.info['last_error']}]")
                print()
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
        return

    # Demo: Phase 1 + Phase 2 + Phase 3 (submit) - full flow for task_14
    task_id = env.state["task_id"]
    gt = env._task.get("ground_truth", {})
    if task_id == "task_14":
        submit_answer = f"The number of respondents who received training is {gt.get('trained_count', 'N/A')}. "
        qp = gt.get("quality_pcts", {})
        for quality, pct in qp.items():
            submit_answer += f'"{quality}": {pct}%. '
        demo_actions = [
            {"action": "files.list", "path": "."},
            {"action": "spreadsheet.read_range", "file": "employee_survey.csv", "range": "columns"},
            {"action": "data.filter", "dataset": "employee_survey.csv", "column": "training_received", "operator": "eq", "value": "Yes"},
            {"action": "data.group_by", "dataset": "filtered_0", "column": "training_quality", "aggregation": "count", "target_column": "employee_id"},
            {"action": "submit", "answer": submit_answer},
        ]
    else:
        demo_actions = [
            {"action": "files.list", "path": "."},
            {"action": "spreadsheet.read_range", "file": "employee_survey.csv", "range": "columns"},
            {"action": "submit", "answer": f"Demo for {task_id} - use --task task_14 for full submit demo."},
        ]

    print("=== DEMO: Phase 1 + 2 + 3 ===\n")
    for action in demo_actions:
        print(f"> {json.dumps(action)}")
        result = env.step(action)
        print(f"  Observation (step {result.step_count}):")
        obs = result.observation
        if len(obs) > 400:
            print(f"  {obs[:400]}...")
        else:
            print(f"  {obs}")
        if result.info.get("last_error"):
            print(f"  [Error: {result.info['last_error']}]")
        print()
        if result.done:
            break

    print("=== DONE ===")
    print(f"Final state: {json.dumps(env.state, indent=2)}")


if __name__ == "__main__":
    main()
