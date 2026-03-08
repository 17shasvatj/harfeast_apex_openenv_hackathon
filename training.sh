#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --job-name=harfeast_mt
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

PROJECT_DIR=/scratch/patel.pranav2/OpenEnv
cd "$PROJECT_DIR"
mkdir -p logs checkpoints_multiturn
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_API_KEY="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"

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Install deps ─────────────────────────────────────────────────
pip install -q trl datasets accelerate transformers openenv-core fastapi uvicorn wandb 2>/dev/null

# ── Generate data ────────────────────────────────────────────────
if [ ! -f harfeast_world/tasks.json ]; then
    echo "Generating default world (seed=42)..."
    python harfeast_synthetic_world_generator.py --seed 42 --output-dir ./harfeast_world
fi
echo "Data ready"

# ── Multi-Turn GRPO Training ─────────────────────────────────────
echo ""
echo "=== Starting Multi-Turn GRPO training ==="
echo "Model:      unsloth/Qwen3-4B"
echo "Training:   Multi-turn agent (batched, K=16)"
echo "Epochs:     10"
echo "Max turns:  10"
echo ""

python train_multiturn.py \
    --model unsloth/Qwen3-4B \
    --world ./harfeast_world \
    --epochs 10 \
    --num-generations 16 \
    --max-turns 10 \
    --lr 5e-6 \
    --temperature 0.8 \
    --eval-before \
    --output-dir ./checkpoints_multiturn

echo "Done."
