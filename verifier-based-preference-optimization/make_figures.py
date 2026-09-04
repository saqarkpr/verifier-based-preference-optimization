"""
Figures for the group-size mechanism test.

    python make_figures.py
"""
import csv
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(k):
    rows = [r for r in csv.DictReader(open(f"results_k{k}/comparison.csv")) if r["seed"] not in ("mean","std")]
    dpo = [float(r["dpo_acc"]) for r in rows]
    grpo = [float(r["grpo_acc"]) for r in rows]
    return dpo, grpo


def main():
    ks = [2, 4, 8, 16]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    dpo_m, dpo_s, grpo_m, grpo_s, gap_m, gap_s = [], [], [], [], [], []
    for k in ks:
        dpo, grpo = read(k)
        dpo_m.append(st.mean(dpo)); dpo_s.append(st.stdev(dpo))
        grpo_m.append(st.mean(grpo)); grpo_s.append(st.stdev(grpo))
        diffs = [g - d for d, g in zip(dpo, grpo)]
        gap_m.append(st.mean(diffs)); gap_s.append(st.stdev(diffs) / len(diffs) ** 0.5)

    axs[0].errorbar(ks, dpo_m, yerr=dpo_s, fmt="o-", capsize=4, color="#4c72b0", label="DPO")
    axs[0].errorbar(ks, grpo_m, yerr=grpo_s, fmt="s-", capsize=4, color="#c44e52", label="GRPO")
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(ks); axs[0].set_xticklabels(ks)
    axs[0].set_xlabel("group_size (K)")
    axs[0].set_ylabel("final accuracy (5 seeds)")
    axs[0].set_title("Neither method's accuracy trends cleanly with K")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

    axs[1].errorbar(ks, gap_m, yerr=gap_s, fmt="o-", capsize=4, color="#55a868")
    axs[1].axhline(0, ls="--", color="grey", lw=1)
    axs[1].annotate("predicted: monotonic increase\n(not observed)",
                    (2, max(gap_m) * 0.7), fontsize=8, color="grey")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xticks(ks); axs[1].set_xticklabels(ks)
    axs[1].set_xlabel("group_size (K)")
    axs[1].set_ylabel("paired gap (GRPO - DPO), mean ± SE")
    axs[1].set_title("The gap spikes at K=8, not a monotonic trend")
    axs[1].grid(alpha=0.3)

    fig.suptitle("Group-size sweep, n=5 seeds per K: no support for the "
                 "sample-efficiency mechanism as stated", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/fig_group_size.png", dpi=150)
    print("saved results/fig_group_size.png")


if __name__ == "__main__":
    main()
