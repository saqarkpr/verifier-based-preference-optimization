"""
Runs the full comparison: SFT once, then DPO and GRPO *both starting from that
same checkpoint*, and reports all three on the identical held-out set.

Branching both methods off one SFT run is the point. If each method got its own
SFT run, the two policies would start from different accuracies and the
comparison would confound "which objective is better" with "which run happened
to have a better starting point."

    python run_pipeline.py --sft_steps 3000 --po_steps 600
    python run_pipeline.py --sft_steps 3000 --po_steps 600 --n_seeds 3
"""
import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np

from sft import run_sft, build_arg_parser as sft_parser
from preference_optim import run_preference_optimization


def run_one_seed(args, seed: int, out_root: str) -> dict:
    run_dir = os.path.join(out_root, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    sft_args = SimpleNamespace(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=0.0, steps=args.sft_steps, batch_size=args.batch_size,
        lr=args.sft_lr, eval_every=args.eval_every,
        out_dir=run_dir, seed=seed,
    )
    print(f"\n{'='*70}\n[seed {seed}] STAGE 1: supervised fine-tuning\n{'='*70}")
    sft_summary = run_sft(sft_args)
    sft_ckpt = sft_summary["checkpoint"]

    summaries = {"sft": sft_summary}
    for method in ("dpo", "grpo"):
        print(f"\n{'='*70}\n[seed {seed}] STAGE 2: {method.upper()}\n{'='*70}")
        po_args = SimpleNamespace(
            method=method, sft_ckpt=sft_ckpt, steps=args.po_steps,
            batch_size=args.po_batch_size, group_size=args.group_size,
            temperature=args.temperature, lr=args.po_lr, beta=args.beta,
            eval_every=args.eval_every,
            out_dir=run_dir, seed=seed,
        )
        summaries[method] = run_preference_optimization(po_args)

    with open(os.path.join(run_dir, "all_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--sft_steps", type=int, default=3000)
    p.add_argument("--sft_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--po_steps", type=int, default=600)
    p.add_argument("--po_batch_size", type=int, default=32)
    p.add_argument("--po_lr", type=float, default=1e-5)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--n_seeds", type=int, default=1,
                    help=">1 repeats the whole pipeline and reports mean +/- std per stage")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [42 + i for i in range(args.n_seeds)]
    all_runs = [run_one_seed(args, s, args.out_dir) for s in seeds]

    # ---- aggregate ------------------------------------------------------
    table_path = os.path.join(args.out_dir, "comparison.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "sft_acc", "dpo_acc", "grpo_acc",
                     "dpo_delta", "grpo_delta"])
        for s, runs in zip(seeds, all_runs):
            w.writerow([s,
                        runs["sft"]["final_accuracy"],
                        runs["dpo"]["final_accuracy"],
                        runs["grpo"]["final_accuracy"],
                        runs["dpo"]["delta_vs_sft"],
                        runs["grpo"]["delta_vs_sft"]])

        if len(seeds) > 1:
            for label, key in (("mean", np.mean), ("std", np.std)):
                w.writerow([label,
                            key([r["sft"]["final_accuracy"] for r in all_runs]),
                            key([r["dpo"]["final_accuracy"] for r in all_runs]),
                            key([r["grpo"]["final_accuracy"] for r in all_runs]),
                            key([r["dpo"]["delta_vs_sft"] for r in all_runs]),
                            key([r["grpo"]["delta_vs_sft"] for r in all_runs])])

    print(f"\n{'='*70}\nFINAL COMPARISON (held-out exact-match accuracy)\n{'='*70}")
    sft_a = [r["sft"]["final_accuracy"] for r in all_runs]
    dpo_a = [r["dpo"]["final_accuracy"] for r in all_runs]
    grpo_a = [r["grpo"]["final_accuracy"] for r in all_runs]
    if len(seeds) > 1:
        print(f"SFT   {np.mean(sft_a):.3f} +/- {np.std(sft_a):.3f}")
        print(f"DPO   {np.mean(dpo_a):.3f} +/- {np.std(dpo_a):.3f}")
        print(f"GRPO  {np.mean(grpo_a):.3f} +/- {np.std(grpo_a):.3f}")
        print("\nRead any DPO/GRPO gap against the seed spread above before "
              "calling it a result.")
    else:
        print(f"SFT {sft_a[0]:.3f} | DPO {dpo_a[0]:.3f} | GRPO {grpo_a[0]:.3f}")
        print("\nSingle seed: no uncertainty estimate. Use --n_seeds 3 before "
              "drawing any conclusion.")
    print(f"\ntable -> {table_path}")


if __name__ == "__main__":
    main()
