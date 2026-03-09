"""
Visualize HarFeast training from saved logs.

Reads:
  - checkpoints_multiturn/train_log.jsonl   (per-step log)
  - checkpoints_multiturn/training_results.json  (before/after eval)

Usage:
  python visualize_training.py [--log-dir ./checkpoints_multiturn] [--output-dir ./plots]
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
})

BLUE = "#2563EB"
GRAY = "#94A3B8"
RED = "#DC2626"
GREEN = "#059669"
PURPLE = "#7C3AED"
ORANGE = "#EA580C"


def load_log(log_path):
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)


def plot_reward_curve(entries, output_dir):
    """Mean reward per step with moving average."""
    rewards = [e["mean_reward"] for e in entries]
    steps = list(range(1, len(rewards) + 1))
    epochs = [e["epoch"] for e in entries]

    window = min(14, len(rewards) // 3) or 1
    smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(steps, rewards, s=10, alpha=0.3, color=BLUE, zorder=2)
    ax.plot(range(window, len(rewards)+1), smooth, linewidth=2, color=BLUE,
            label=f"Moving avg ({window}-step)", zorder=3)

    baseline = rewards[0] if rewards else 0
    ax.axhline(y=baseline, color=RED, linestyle="--", linewidth=1, alpha=0.5,
               label=f"Initial ({baseline:.3f})")

    epoch_boundaries = []
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i-1]:
            epoch_boundaries.append(i + 1)
    for eb in epoch_boundaries:
        ax.axvline(x=eb, color=GRAY, linestyle=":", alpha=0.4)

    ax.set_xlabel("Training Step (task)")
    ax.set_ylabel("Mean Rubric Score")
    ax.set_title("Training Reward Curve", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  reward_curve.png")


def plot_loss_curve(entries, output_dir):
    """Loss over training steps (signal steps only)."""
    signal = [e for e in entries if e.get("signal")]
    if not signal:
        print("  (no signal steps, skipping loss plot)")
        return

    losses = [e["loss"] for e in signal]
    steps = list(range(1, len(losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=1.5, color=RED, alpha=0.7)
    if len(losses) > 7:
        w = min(7, len(losses) // 3) or 1
        smooth = np.convolve(losses, np.ones(w)/w, mode="valid")
        ax.plot(range(w, len(losses)+1), smooth, linewidth=2, color=RED, label=f"Smoothed ({w}-step)")
        ax.legend()
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Loss")
    ax.set_title("GRPO Loss Over Training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  loss_curve.png")


def plot_epoch_summary(entries, output_dir):
    """Bar chart: mean reward and signal fraction per epoch."""
    epoch_data = {}
    for e in entries:
        ep = e["epoch"]
        if ep not in epoch_data:
            epoch_data[ep] = {"rewards": [], "signals": 0, "total": 0}
        epoch_data[ep]["rewards"].append(e["mean_reward"])
        epoch_data[ep]["total"] += 1
        if e.get("signal"):
            epoch_data[ep]["signals"] += 1

    epochs = sorted(epoch_data.keys())
    mean_rewards = [np.mean(epoch_data[ep]["rewards"]) for ep in epochs]
    signal_frac = [epoch_data[ep]["signals"] / epoch_data[ep]["total"] for ep in epochs]
    nonzero_frac = [np.mean([1 if r > 0 else 0 for r in epoch_data[ep]["rewards"]]) for ep in epochs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.bar(epochs, mean_rewards, color=BLUE, alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Mean Reward Per Epoch", fontweight="bold")
    for ep, mr in zip(epochs, mean_rewards):
        ax1.text(ep, mr + 0.002, f"{mr:.3f}", ha="center", va="bottom", fontsize=9)

    x = np.array(epochs)
    w = 0.35
    ax2.bar(x - w/2, signal_frac, w, label="Has variance (trained)", color=GREEN, alpha=0.7)
    ax2.bar(x + w/2, nonzero_frac, w, label="Nonzero reward", color=PURPLE, alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Fraction of tasks")
    ax2.set_title("Learning Signal Per Epoch", fontweight="bold")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "epoch_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  epoch_summary.png")


def plot_per_task(entries, output_dir):
    """Per-task reward heatmap across epochs."""
    task_epoch = {}
    for e in entries:
        key = (e["task"], e["epoch"])
        task_epoch[key] = e["mean_reward"]

    tasks = sorted(set(e["task"] for e in entries))
    epochs = sorted(set(e["epoch"] for e in entries))

    matrix = np.zeros((len(tasks), len(epochs)))
    for ti, task in enumerate(tasks):
        for ei, ep in enumerate(epochs):
            matrix[ti, ei] = task_epoch.get((task, ep), 0)

    fig, ax = plt.subplots(figsize=(max(8, len(epochs) * 1.2), max(5, len(tasks) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest", vmin=0)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([f"E{ep}" for ep in epochs])
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_title("Per-Task Reward Across Epochs", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Reward")

    for ti in range(len(tasks)):
        for ei in range(len(epochs)):
            val = matrix[ti, ei]
            if val > 0:
                color = "white" if val > 0.3 else "black"
                ax.text(ei, ti, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_task_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  per_task_heatmap.png")


def plot_before_after(results, output_dir):
    """Before vs After bar chart from training_results.json."""
    before = results.get("before_results")
    after = results.get("after_results")
    if not before or not after:
        print("  (no before/after results, skipping)")
        return

    tasks = [r["task_id"] for r in before]
    names = [r.get("task_name", r["task_id"])[:20] for r in before]
    before_scores = [r["score"] for r in before]
    after_scores = [r["score"] for r in after]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(tasks))
    w = 0.35
    ax.bar(x - w/2, before_scores, w, label="Before", color=GRAY, alpha=0.6)
    ax.bar(x + w/2, after_scores, w, label="After GRPO", color=BLUE)

    for i, (b, a) in enumerate(zip(before_scores, after_scores)):
        if a > b + 1:
            ax.annotate(f"+{a-b:.0f}", xy=(x[i] + w/2, a), ha="center", va="bottom",
                        fontsize=8, color=GREEN, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Rubric Score (%)")
    ax.set_title("Before vs After Training", fontweight="bold")
    ax.legend()

    overall_b = results.get("before_score", 0)
    overall_a = results.get("after_score", 0)
    ax.text(0.98, 0.95, f"Overall: {overall_b:.1f}% -> {overall_a:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN, alpha=0.15, edgecolor=GREEN))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "before_after.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  before_after.png")


def plot_reward_distribution(entries, output_dir):
    """Histogram of all individual rewards across training."""
    all_rewards = []
    for e in entries:
        all_rewards.extend(e.get("rewards", [e["mean_reward"]]))

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(max(all_rewards), 0.5), 30)
    ax.hist(all_rewards, bins=bins, color=BLUE, alpha=0.7, edgecolor="white")
    nonzero = sum(1 for r in all_rewards if r > 0)
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title(f"Reward Distribution ({nonzero}/{len(all_rewards)} nonzero)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  reward_distribution.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./checkpoints_multiturn")
    parser.add_argument("--output-dir", default="./plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, "train_log.jsonl")
    results_path = os.path.join(args.log_dir, "training_results.json")

    print(f"Reading from: {args.log_dir}")
    print(f"Output to:    {args.output_dir}\n")

    entries = []
    if os.path.exists(log_path):
        entries = load_log(log_path)
        print(f"Loaded {len(entries)} log entries\n")
    else:
        print(f"No train_log.jsonl found at {log_path}")

    results = {}
    if os.path.exists(results_path):
        results = load_results(results_path)
        print(f"Loaded training_results.json\n")

    print("Generating plots:")
    if entries:
        plot_reward_curve(entries, args.output_dir)
        plot_loss_curve(entries, args.output_dir)
        plot_epoch_summary(entries, args.output_dir)
        plot_per_task(entries, args.output_dir)
        plot_reward_distribution(entries, args.output_dir)

    if results:
        plot_before_after(results, args.output_dir)

    if not entries and not results:
        print("  No data found. Run training first.")
        return

    print(f"\nDone. Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
