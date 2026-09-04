"""
Plots for the SFT / DPO / GRPO comparison.

    # accuracy over training for both methods, on one axis
    python plot_results.py --run_dir results/seed42 --out results/curves.png

    # final bar chart across seeds, with error bars
    python plot_results.py --comparison_csv results/comparison.csv --out results/final.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path):
    rows = {"step": [], "eval_acc": [], "mean_reward": [], "loss": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows["step"].append(int(r["step"]))
            rows["eval_acc"].append(float(r["eval_acc"]))
            rows["loss"].append(float(r["loss"]))
            if "mean_reward" in r:
                rows["mean_reward"].append(float(r["mean_reward"]))
    return rows


def plot_curves(run_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sft_path = os.path.join(run_dir, "log_sft.csv")
    if os.path.exists(sft_path):
        sft = read_log(sft_path)
        axes[0].plot(sft["step"], sft["eval_acc"], "-", color="#888", label="SFT")

    for method, color in (("dpo", "#4c72b0"), ("grpo", "#c44e52")):
        p = os.path.join(run_dir, f"log_{method}.csv")
        if not os.path.exists(p):
            continue
        d = read_log(p)
        axes[1].plot(d["step"], d["eval_acc"], "-", color=color, label=f"{method.upper()} acc")
        if d["mean_reward"]:
            axes[1].plot(d["step"], d["mean_reward"], "--", color=color, alpha=0.5,
                          label=f"{method.upper()} sampled reward")

    axes[0].set_xlabel("SFT step"); axes[0].set_ylabel("held-out accuracy")
    axes[0].set_title("Stage 1: supervised fine-tuning")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("preference-optimization step"); axes[1].set_ylabel("accuracy / reward")
    axes[1].set_title("Stage 2: DPO vs GRPO (both from the same SFT checkpoint)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_final(csv_path, out_path):
    rows = list(csv.DictReader(open(csv_path)))
    seed_rows = [r for r in rows if r["seed"] not in ("mean", "std")]
    mean_row = next((r for r in rows if r["seed"] == "mean"), None)
    std_row = next((r for r in rows if r["seed"] == "std"), None)

    labels = ["SFT", "DPO", "GRPO"]
    keys = ["sft_acc", "dpo_acc", "grpo_acc"]

    if mean_row and std_row:
        means = [float(mean_row[k]) for k in keys]
        errs = [float(std_row[k]) for k in keys]
        title = f"Final accuracy (mean +/- std over {len(seed_rows)} seeds)"
    else:
        means = [float(seed_rows[0][k]) for k in keys]
        errs = [0, 0, 0]
        title = "Final accuracy (single seed -- no uncertainty estimate)"

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(labels, means, yerr=errs, capsize=6, color=["#888", "#4c72b0", "#c44e52"])
    ax.set_ylabel("held-out exact-match accuracy")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--comparison_csv", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if args.run_dir:
        plot_curves(args.run_dir, args.out)
    elif args.comparison_csv:
        plot_final(args.comparison_csv, args.out)
    else:
        raise SystemExit("provide --run_dir or --comparison_csv")


if __name__ == "__main__":
    main()
