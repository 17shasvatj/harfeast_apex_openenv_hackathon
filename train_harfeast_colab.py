"""
HarFeast + TRL Training - Colab-ready script.
Copy cells into Colab or run as: python train_harfeast_colab.py

Setup (Colab cell 1):
  !pip install -q openenv-core trl datasets accelerate
  # If harfeast not in path, clone or upload
"""

# %% [markdown]
# # HarFeast OpenEnv Training
# Train a model to produce management consulting answers that score well on rubric.

# %%
import json
import os
import sys

# Add project to path (adjust if needed)
if "harfeast_apex_openenv_hackathon" not in os.getcwd():
    sys.path.insert(0, "/content/harfeast_apex_openenv_hackathon")  # Colab mount
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

# %%
from datasets import Dataset
# Install from project: pip install -e . or add to path
from harfeast_env import HarFeastEnv, HarFeastAction

# Env URL: HF Space or local
ENV_URL = os.environ.get("HARFEAST_ENV_URL", "http://localhost:8000")
print(f"Using env: {ENV_URL}")

# %%
def get_prompts(n=16):
    path = "harfeast_world/tasks.json"
    if not os.path.isfile(path):
        path = "harfeast_apex_openenv_hackathon/harfeast_world/tasks.json"
    with open(path) as f:
        tasks = json.load(f)
    return [tasks[i % len(tasks)]["prompt"] for i in range(n)]

# %%
def rollout_func(prompts, trainer):
    from trl.experimental.openenv import generate_rollout_completions
    outputs = generate_rollout_completions(trainer, prompts)
    tokenizer = trainer.processing_class
    completions = [tokenizer.decode(o["completion_ids"], skip_special_tokens=True).strip() for o in outputs]
    client = HarFeastEnv(base_url=ENV_URL)
    rewards = []
    for i, comp in enumerate(completions):
        answer = comp.split("Answer:")[-1].strip() if "Answer:" in comp else comp[:500]
        try:
            client.reset(seed=i)
            r = client.step(HarFeastAction(action_json=json.dumps({"action": "submit", "answer": answer})))
            rewards.append(float(r.reward or 0))
        except Exception:
            rewards.append(0.0)
    client.close()
    return {
        "prompt_ids": [o["prompt_ids"] for o in outputs],
        "completion_ids": [o["completion_ids"] for o in outputs],
        "logprobs": [o["logprobs"] for o in outputs],
        "env_reward": rewards,
    }

def reward_func(completions, **kw):
    r = kw.get("env_reward", [])
    return [float(x) for x in r] if r else [0.0] * len(completions)

# %%
system = "You are a management consultant. Given the task, provide your final answer with specific numbers. Format: Answer: <your answer>"
prompts = get_prompts(16)
dataset = Dataset.from_dict({"prompt": [f"{system}\n\nTask:\n{p}\n\nAnswer:" for p in prompts]})

# %%
from trl import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_func,
    train_dataset=dataset,
    rollout_func=rollout_func,
    args=GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        num_train_epochs=1,
        num_generations=4,
        max_completion_length=512,
        per_device_train_batch_size=4,
    ),
)
trainer.train()
