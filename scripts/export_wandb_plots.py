#!/usr/bin/env python3
"""
Export W&B run data and generate plots for HarFeast GRPO training.

Usage:
  python scripts/export_wandb_plots.py              # use W&B API (set WANDB_API_KEY or wandb login)
  python scripts/export_wandb_plots.py --local     # use local train_log.jsonl (no W&B needed)

Requires: pip install wandb pandas matplotlib
"""

import argparse
import json
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Project from user
ENTITY_PROJECT = "pranavpatel-northeastern-university/harfeast-grpo"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "wandb_export")
EXCLUDE_TASK_PLOTS = {"task_08", "task_13", "task_14"}  # do not download per-task for these

# Default local log path (v4 run)
DEFAULT_LOCAL_LOG = os.path.join(REPO_ROOT, "ckpt_v4_q25", "train_log.jsonl")


def build_df_from_local_log(log_path):
    """Build a DataFrame matching W&B history from train_log.jsonl."""
    with open(log_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    if not entries:
        return pd.DataFrame(), pd.DataFrame()

    # Step-level rows (each log entry = one task, has epoch + step)
    step_rows = []
    for d in entries:
        row = {
            "train/step": d.get("step"),
            "train/mean_reward": d.get("mean_reward"),
            "train/loss": d.get("loss"),
            "train/reward_variance": d.get("variance"),
            "epoch/epoch": d.get("epoch"),
        }
        task = d.get("task")
        if task:
            row[f"task_reward/{task}"] = d.get("mean_reward")
        step_rows.append(row)
    df = pd.DataFrame(step_rows)
    task_cols = [c for c in df.columns if c.startswith("task_reward/")]
    for c in task_cols:
        df[c] = df[c].ffill()

    # Epoch-level: one row per epoch
    by_epoch = {}
    for e in entries:
        ep = e.get("epoch")
        if ep not in by_epoch:
            by_epoch[ep] = []
        by_epoch[ep].append(e)
    epoch_rows = []
    for ep in sorted(by_epoch.keys()):
        L = by_epoch[ep]
        mean_r = sum(x.get("mean_reward", 0) for x in L) / len(L)
        nonzero = sum(1 for x in L if x.get("mean_reward", 0) > 0) / len(L)
        signal = sum(1 for x in L if x.get("signal"))
        epoch_rows.append({
            "epoch/epoch": ep,
            "epoch/mean_reward": mean_r,
            "epoch/signal_tasks": signal,
            "epoch/nonzero_rewards_pct": nonzero,
        })
    df_epoch = pd.DataFrame(epoch_rows)
    # Merge epoch stats into main df so epoch plots have data
    for col in ["epoch/mean_reward", "epoch/signal_tasks", "epoch/nonzero_rewards_pct"]:
        if col not in df.columns:
            df[col] = None
    for _, er in df_epoch.iterrows():
        ep = er["epoch/epoch"]
        mask = df["epoch/epoch"] == ep
        df.loc[mask, "epoch/mean_reward"] = er["epoch/mean_reward"]
        df.loc[mask, "epoch/signal_tasks"] = er["epoch/signal_tasks"]
        df.loc[mask, "epoch/nonzero_rewards_pct"] = er["epoch/nonzero_rewards_pct"]
    return df, df_epoch


def fetch_runs_and_csv():
    """Fetch runs and save project.csv (summary, config, name)."""
    import wandb
    api = wandb.Api()
    runs = api.runs(ENTITY_PROJECT)
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        name_list.append(run.name)
    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs_df.to_csv(os.path.join(OUTPUT_DIR, "project.csv"))
    print(f"Saved {OUTPUT_DIR}/project.csv ({len(runs_df)} runs)")
    return runs_df, api


def get_run_history(api, run_id=None):
    """Get full history for one run. If run_id is None, use latest run."""
    import wandb
    if run_id:
        run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    else:
        runs = api.runs(ENTITY_PROJECT)
        run = next(iter(runs))
    # run.history() returns DataFrame with all logged metrics
    df = run.history()
    print(f"Run: {run.name} ({run.id}), history rows: {len(df)}")
    return df, run


def plot_reward_curve(df, outpath):
    """train/mean_reward vs train/step with smoothing."""
    if "train/mean_reward" not in df.columns or "train/step" not in df.columns:
        print("  Skip reward_curve: missing train/mean_reward or train/step")
        return
    df = df.dropna(subset=["train/mean_reward", "train/step"]).sort_values("train/step")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["train/step"], df["train/mean_reward"], alpha=0.5, s=20, color="steelblue")
    if len(df) >= 14:
        smooth = df["train/mean_reward"].rolling(14, min_periods=1).mean()
        ax.plot(df["train/step"], smooth, color="darkblue", lw=2, label="14-step MA")
    ax.axhline(df["train/mean_reward"].iloc[0], color="red", ls="--", alpha=0.7, label="Initial")
    ax.set_xlabel("Training Step (task)")
    ax.set_ylabel("Mean Rubric Score")
    ax.set_title("Training Reward Curve (W&B)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_loss_curve(df, outpath):
    """train/loss vs train/step."""
    if "train/loss" not in df.columns or "train/step" not in df.columns:
        print("  Skip loss_curve: missing train/loss or train/step")
        return
    df = df.dropna(subset=["train/loss", "train/step"]).sort_values("train/step")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["train/step"], df["train/loss"], color="coral", alpha=0.7, label="Loss")
    if len(df) >= 7:
        smooth = df["train/loss"].rolling(7, min_periods=1).mean()
        ax.plot(df["train/step"], smooth, color="darkred", lw=2, label="7-step MA")
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Loss")
    ax.set_title("GRPO Loss Over Training (W&B)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_epoch_mean_reward(df, outpath):
    """epoch/mean_reward vs epoch/epoch."""
    if "epoch/mean_reward" not in df.columns or "epoch/epoch" not in df.columns:
        print("  Skip epoch_mean_reward: missing epoch/*")
        return
    df = df.dropna(subset=["epoch/mean_reward", "epoch/epoch"])
    df = df.drop_duplicates(subset=["epoch/epoch"], keep="last").sort_values("epoch/epoch")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["epoch/epoch"], df["epoch/mean_reward"], color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward Per Epoch (W&B)")
    for i, (e, v) in enumerate(zip(df["epoch/epoch"], df["epoch/mean_reward"])):
        ax.text(e, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(df["epoch/mean_reward"].max() * 1.15, 0.1))
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_epoch_nonzero_pct(df, outpath):
    """epoch/nonzero_rewards_pct and epoch/signal_tasks vs epoch."""
    need = ["epoch/epoch", "epoch/nonzero_rewards_pct", "epoch/signal_tasks"]
    if not all(c in df.columns for c in need):
        print("  Skip epoch_nonzero: missing epoch columns")
        return
    df = df.dropna(subset=need)
    df = df.drop_duplicates(subset=["epoch/epoch"], keep="last").sort_values("epoch/epoch")
    if df.empty:
        return
    n_tasks = 14
    df = df.copy()
    df["signal_frac"] = df["epoch/signal_tasks"] / n_tasks
    fig, ax = plt.subplots(figsize=(7, 4))
    x = df["epoch/epoch"]
    w = 0.35
    ax.bar(x - w/2, df["epoch/nonzero_rewards_pct"], width=w, label="Nonzero reward %", color="purple", alpha=0.7)
    ax.bar(x + w/2, df["signal_frac"], width=w, label="Has variance (trained)", color="green", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of tasks")
    ax.set_title("Learning Signal Per Epoch (W&B)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_reward_variance(df, outpath):
    """train/reward_variance vs train/step."""
    if "train/reward_variance" not in df.columns or "train/step" not in df.columns:
        print("  Skip reward_variance: missing columns")
        return
    df = df.dropna(subset=["train/reward_variance", "train/step"]).sort_values("train/step")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["train/step"], df["train/reward_variance"], color="teal", alpha=0.8)
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Reward Variance")
    ax.set_title("Reward Variance Over Training (W&B)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_per_task(df, outpath_base, exclude):
    """One plot per task_reward/task_* (excluding task_08, task_13, task_14)."""
    prefix = "task_reward/"
    task_cols = [c for c in df.columns if c.startswith(prefix)]
    task_cols = [c for c in task_cols if c.replace(prefix, "") not in exclude]
    if not task_cols:
        print("  Skip per_task: no task_reward columns (or all excluded)")
        return
    # Single figure: one line per task
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in sorted(task_cols):
        task_id = col.replace(prefix, "")
        sub = df[["train/step", col]].dropna().sort_values("train/step")
        if sub.empty:
            continue
        ax.plot(sub["train/step"], sub[col], label=task_id, alpha=0.8)
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Per-Task Reward Over Training (W&B, excl. task_08/13/14)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath_base + ".png", dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath_base}.png")


def main():
    parser = argparse.ArgumentParser(description="Export W&B data and generate plots")
    parser.add_argument("--local", action="store_true", help="Use local train_log.jsonl instead of W&B API")
    parser.add_argument("--log", default=DEFAULT_LOCAL_LOG, help="Path to train_log.jsonl (default: ckpt_v4_q25/train_log.jsonl)")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        if not args.local:
            print("Install: pip install wandb pandas matplotlib")
            sys.exit(1)
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install: pip install matplotlib")
        sys.exit(1)

    df = None
    if args.local:
        if not os.path.isfile(args.log):
            print(f"Local log not found: {args.log}")
            sys.exit(1)
        df, _ = build_df_from_local_log(args.log)
        print(f"Loaded {len(df)} rows from {args.log}")
    else:
        runs_df, api = fetch_runs_and_csv()
        df, run = get_run_history(api, run_id=None)
        if df is None or df.empty:
            print("No history for run. Exiting.")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating plots...")
    plot_reward_curve(df, os.path.join(OUTPUT_DIR, "wandb_reward_curve.png"))
    plot_loss_curve(df, os.path.join(OUTPUT_DIR, "wandb_loss_curve.png"))
    plot_epoch_mean_reward(df, os.path.join(OUTPUT_DIR, "wandb_epoch_mean_reward.png"))
    plot_epoch_nonzero_pct(df, os.path.join(OUTPUT_DIR, "wandb_epoch_signal.png"))
    plot_reward_variance(df, os.path.join(OUTPUT_DIR, "wandb_reward_variance.png"))
    plot_per_task(df, os.path.join(OUTPUT_DIR, "wandb_per_task_rewards"), EXCLUDE_TASK_PLOTS)
    if df is not None and not df.empty:
        df.to_csv(os.path.join(OUTPUT_DIR, "run_history.csv"), index=False)
        print(f"Saved run_history.csv to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
