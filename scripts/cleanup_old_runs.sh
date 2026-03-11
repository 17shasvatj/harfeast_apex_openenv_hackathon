#!/bin/bash
# Run from repo root on HPC before starting v4 (Qwen2.5-7B).
# v4 uses --model Qwen/Qwen2.5-7B-Instruct (fresh), so no checkpoint dependency.
# Usage: cd /scratch/.../harfeast_apex_openenv_hackathon && bash scripts/cleanup_old_runs.sh

set -e
# If run from scripts/, go to repo root; else assume already in repo root
case "$(dirname "$0")" in
  .)   ;;
  *)   cd "$(dirname "$0")/.." ;;
esac
echo "=== HarFeast cleanup (old checkpoints & outputs) ==="
echo "Working directory: $(pwd)"
echo ""

# Old checkpoints (v4 does NOT use these)
for dir in ckpt_h200 ckpt_h200_v2 ckpt_h200_v3; do
  if [ -d "$dir" ]; then
    echo "Removing $dir ..."
    rm -rf "$dir"
  fi
done

# Old plot dirs (can regenerate from train_log.jsonl if needed)
for dir in plots_h200 plots_h200_v2 plots_h200_v3 plots_a1 plotsa1; do
  if [ -d "$dir" ]; then
    echo "Removing $dir ..."
    rm -rf "$dir"
  fi
done

# Old log files (keep logs/ folder and any v4_* logs when they appear)
for f in logs/harfeast_*.out logs/harfeast_*.err logs/a100_*.out logs/a100_*.err \
         logs/h200_*.out logs/h200_*.err logs/h200v2_*.out logs/h200v2_*.err \
         logs/h200v3_*.out logs/h200v3_*.err; do
  if [ -f "$f" ]; then
    echo "Removing $f ..."
    rm -f "$f"
  fi
done

# Optional: wandb run cache (can be large). Uncomment if you need space.
# if [ -d "wandb" ]; then echo "Removing wandb/ ..."; rm -rf wandb; fi

echo ""
echo "Done. Kept: harfeast_world/, venv/, code, logs/ (v4_* will appear when job runs)."
