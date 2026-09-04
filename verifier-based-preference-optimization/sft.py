"""
Stage 1: supervised fine-tuning (SFT).

Trains the policy on (prompt, correct answer) pairs with cross-entropy on the
completion tokens only. This is the reference point everything else is
measured against: DPO and GRPO both start from the SFT checkpoint and both
use it as their frozen reference model, so "did preference optimization help?"
always means "beyond what supervised learning on correct answers already gave."

Deliberately trained to a *partially* correct policy rather than to
saturation. If SFT already solved the task, there would be no headroom for
preference optimization to show any effect, and the comparison would be
vacuous -- `--steps` is the knob controlling that headroom.
"""
import argparse
import csv
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from policy import build_policy, sample_completions, completion_strings, prompt_strings
from task import make_sft_batch, make_prompt_batch, verify


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


EVAL_BATCH = 64          # internal only; does not affect which problems are drawn
EVAL_PROBLEMS = 512      # size of the fixed held-out set


@torch.no_grad()
def eval_accuracy(model, eval_seed: int, device: str,
                   n_problems: int = EVAL_PROBLEMS, greedy: bool = True) -> float:
    """Exact-match accuracy under the programmatic verifier, on a FIXED
    held-out problem set.

    Two things are deliberately pinned here:

    1. A fresh `random.Random(eval_seed)` is built on every call, so every
       evaluation -- across stages, methods, and training steps -- scores the
       identical problems. An earlier version threaded one shared RNG through,
       so each eval drew *different* problems and the variance swamped the
       effects being measured.
    2. The problem count and internal batch size are constants, NOT the
       training batch size. When they were tied to training batch size, SFT
       and DPO evaluated the same checkpoint on different-sized sets and
       reported two different accuracies for one model.
    """
    rng = random.Random(eval_seed)
    correct, total = 0.0, 0
    while total < n_problems:
        b = min(EVAL_BATCH, n_problems - total)
        prompts, _ = make_prompt_batch(b, rng, device)
        seqs = sample_completions(model, prompts, greedy=greedy)
        for p, c in zip(prompt_strings(seqs), completion_strings(seqs)):
            correct += verify(p, c)
            total += 1
    return correct / max(total, 1)


def run_sft(args) -> dict:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    eval_seed = args.seed + 10_000  # fixed held-out problem set (see eval_accuracy)

    model = build_policy(args.d_model, args.n_heads, args.n_layers,
                          dropout=args.dropout, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sft] params: {n_params:,} | device: {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_path = os.path.join(args.out_dir, "log_sft.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "eval_acc"])

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y, comp_mask = make_sft_batch(args.batch_size, rng, device)
        logits, _ = model(x)
        loss_per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).view(y.shape)
        # restrict the loss to completion tokens -- see task.make_sft_batch
        loss = (loss_per_tok * comp_mask.float()).sum() / comp_mask.float().sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            acc = eval_accuracy(model, eval_seed, device)
            print(f"[sft] step {step:5d} | loss {loss.item():.4f} | eval_acc {acc:.3f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, loss.item(), acc])

    final_acc = eval_accuracy(model, eval_seed, device)
    ckpt_path = os.path.join(args.out_dir, "model_sft.pt")
    torch.save({"model_state": model.state_dict(), "args": vars(args), "stage": "sft"}, ckpt_path)

    summary = {
        "stage": "sft", "seed": args.seed, "steps": args.steps,
        "n_params": n_params, "final_accuracy": final_acc,
        "training_time_s": time.time() - t0, "checkpoint": ckpt_path, "log": log_path,
    }
    with open(os.path.join(args.out_dir, "summary_sft.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[sft] final greedy accuracy: {final_acc:.3f} -> {ckpt_path}")
    return summary


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=2000,
                    help="kept short on purpose: SFT should leave headroom for DPO/GRPO to act on")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--out_dir", type=str, default="results/run1")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run_sft(build_arg_parser().parse_args())
