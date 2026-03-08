#!/usr/bin/env python3
"""
Minimal HarFeast + TRL training script for OpenEnv hackathon.
Trains a model to produce answers that score well on HarFeast rubric.

Usage (after deploying HarFeast to HF Space):
  python train_harfeast.py --env-url https://YOUR-USERNAME-harfeast-env.hf.space

Or with local server:
  python -m uvicorn harfeast_env.server.app:app --host 0.0.0.0 --port 8001 &
  python train_harfeast.py --env-url http://localhost:8001
"""

import argparse
import json
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_harfeast_prompts(tasks_path: str, n: int = 32) -> list[dict]:
    """Load task prompts for training."""
    with open(tasks_path) as f:
        tasks = json.load(f)
    # Cycle through tasks
    samples = []
    for i in range(n):
        t = tasks[i % len(tasks)]
        samples.append({
            "prompt": t["prompt"],
            "task_id": t["task_id"],
        })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:8000",
                        help="HarFeast env URL (HF Space or local)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()

    # Lazy imports for faster --help
    from datasets import Dataset
    from harfeast_env import HarFeastEnv, HarFeastAction

    # Load prompts
    tasks_path = os.path.join(os.path.dirname(__file__), "harfeast_world", "tasks.json")
    if not os.path.isfile(tasks_path):
        print("Run harfeast_synthetic_world_generator.py first.")
        sys.exit(1)
    samples = get_harfeast_prompts(tasks_path, args.samples)

    # Build dataset - prompt instructs model to analyze and submit answer
    system = (
        "You are a management consultant analyzing HarFeast data. "
        "Given a task, provide your final answer with specific numbers. "
        "Format: Answer: <your analysis with numbers>"
    )
    prompts = [
        f"{system}\n\nTask:\n{s['prompt']}\n\nAnswer:"
        for s in samples
    ]
    dataset = Dataset.from_dict({"prompt": prompts})

    # Connect to env
    client = HarFeastEnv(base_url=args.env_url)

    def rollout_func(prompts_list, trainer):
        try:
            from trl.experimental.openenv import generate_rollout_completions
        except ImportError:
            raise ImportError("Install trl: pip install trl")

        outputs = generate_rollout_completions(trainer, prompts_list)
        tokenizer = trainer.processing_class
        completions = [
            tokenizer.decode(o["completion_ids"], skip_special_tokens=True).strip()
            for o in outputs
        ]

        env_rewards = []
        for i, comp in enumerate(completions):
            # Extract answer (after "Answer:" or use full completion)
            answer = comp
            if "Answer:" in comp:
                answer = comp.split("Answer:")[-1].strip()
            if not answer:
                answer = comp[:500]

            try:
                client.reset(seed=i)
                action = HarFeastAction(action_json=json.dumps({"action": "submit", "answer": answer}))
                result = client.step(action)
                env_rewards.append(float(result.reward or 0))
            except Exception as e:
                print(f"Env error: {e}")
                env_rewards.append(0.0)

        return {
            "prompt_ids": [o["prompt_ids"] for o in outputs],
            "completion_ids": [o["completion_ids"] for o in outputs],
            "logprobs": [o["logprobs"] for o in outputs],
            "env_reward": env_rewards,
        }

    def reward_func(completions, **kwargs):
        r = kwargs.get("env_reward", [])
        return [float(x) for x in r] if r else [0.0] * len(completions)

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("Install trl: pip install trl")
        sys.exit(1)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        rollout_func=rollout_func,
        args=GRPOConfig(
            use_vllm=True,
            vllm_mode="colocate",
            num_train_epochs=args.epochs,
            num_generations=4,
            max_completion_length=512,
            per_device_train_batch_size=4,
        ),
    )
    trainer.train()
    client.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
