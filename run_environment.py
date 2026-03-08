#!/usr/bin/env python3
"""
CLI to run and test the HarFeast OpenEnv environment.
Phase 1: files.list, files.read.

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
    env = HarFeastOpenEnv(world_path=world_path)

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

    # Demo: run a fixed sequence of Phase 1 actions
    demo_actions = [
        {"action": "files.list", "path": "."},
        {"action": "files.list", "path": "data"},
        {"action": "files.list", "path": "documents"},
        {"action": "files.read", "path": "documents/scrap_rate_report.txt"},
        {"action": "files.read", "path": "data/employee_survey.csv"},  # Should reject
        {"action": "files.read", "path": "interview_sarah_jenkins.txt"},
    ]

    print("=== DEMO: Phase 1 actions ===\n")
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
