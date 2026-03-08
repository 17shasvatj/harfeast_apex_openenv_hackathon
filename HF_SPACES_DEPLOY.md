# Deploying HarFeast to Hugging Face Spaces

## Prerequisites

- Hugging Face account
- OpenEnv 0.2.1 compatible setup

## Option 1: Docker Space

1. Create a new Space at https://huggingface.co/new-space
2. Select **Docker** as the SDK
3. Clone your repo or upload files
4. Ensure these files exist at repo root:
   - `Dockerfile`
   - `harfeast_env/`
   - `harfeast_openenv/`
   - `harfeast_world/` (or it will be generated on build)
   - `harfeast_synthetic_world_generator.py`

5. The Space will build from the Dockerfile and expose port 8000.

## Option 2: openenv push (if available)

```bash
pip install "openenv-core[cli]>=0.2.1"
cd harfeast_apex_openenv_hackathon
openenv push --repo-id YOUR_USERNAME/harfeast-env
```

## Verify Deployment

Once the Space is running:

```python
from harfeast_env import HarFeastEnv, HarFeastAction
import json

client = HarFeastEnv(base_url="https://YOUR-USERNAME-harfeast-env.hf.space")
result = client.reset()
print(result.observation.observation)

action = HarFeastAction(action_json=json.dumps({"action": "files.list", "path": "."}))
result = client.step(action)
print(result.observation.observation)
client.close()
```

## Training with TRL

After deployment, run the training script pointing to your Space URL:

```bash
python train_harfeast.py --env-url https://YOUR-USERNAME-harfeast-env.hf.space
```

For Colab: copy `train_harfeast.py` and the `harfeast_env` package, install `trl`, `harfeast-env`, and run.
