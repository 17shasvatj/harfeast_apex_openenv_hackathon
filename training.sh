#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=harfeast_grpo
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/harfeast_%j.out
#SBATCH --error=logs/harfeast_%j.err

set -e

# ── Env setup ────────────────────────────────────────────────────
module load anaconda3/2024.06
module load cuda/12.8.0
module load cuDNN/9.10.2

source /scratch/patel.pranav2/OpenEnv/venv/bin/activate

PROJECT_DIR=/scratch/patel.pranav2/OpenEnv/harfeast_apex_openenv_hackathon
cd "$PROJECT_DIR"
mkdir -p logs checkpoints
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Install deps (cached after first run) ────────────────────────
pip install -q trl datasets accelerate transformers vllm openenv-core fastapi uvicorn 2>/dev/null

# ── Generate data on HPC (pure Python, ~30s) ────────────────────
if [ ! -f harfeast_world/tasks.json ]; then
    echo "Generating default world (seed=42)..."
    python harfeast_synthetic_world_generator.py --seed 42 --output-dir ./harfeast_world
fi

if [ ! -f harfeast_worlds/all_tasks.json ]; then
    echo "Generating augmented dataset (40 worlds, 560 tasks)..."
    python harfeast_synthetic_world_generator.py --batch 40 --output-dir ./harfeast_worlds
fi
echo "Data ready: $(wc -l < harfeast_worlds/all_tasks.json) lines in all_tasks.json"

# ── Train (no server needed — rewards scored in-process) ────────
echo ""
echo "=== Starting GRPO training ==="
echo "Model:    unsloth/Qwen3-4B"
echo "Rewards:  correctness + format + completeness (GDPO-style)"
echo "Samples:  128"
echo ""

python train_harfeast.py \
    --model unsloth/Qwen3-4B \
    --worlds-base ./harfeast_worlds \
    --samples 128 \
    --epochs 1 \
    --eval-before \
    --output-dir ./checkpoints

echo ""
echo "=== Standalone eval on checkpoint ==="
python eval_harfeast.py \
    --model ./checkpoints \
    --world ./harfeast_world

echo "Done."
