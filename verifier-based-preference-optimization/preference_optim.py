"""
Stage 2: preference optimization -- DPO and GRPO.

Both methods live in ONE file behind a `--method` flag on purpose. They share
the same training loop, the same rollout code, the same evaluation, the same
logging, and the same frozen reference model; the only thing that differs is
the loss function. Implementing them as two separate scripts would make any
measured gap between them partly attributable to incidental differences in
plumbing, which would defeat the point of the comparison.

DPO -- offline, pairwise preference optimization:

    L = -log sigma( beta * [ (logp(y_w) - logp_ref(y_w))
                            - (logp(y_l) - logp_ref(y_l)) ] )

    Optimizes a preference between two completions. The reference model
    appears as a per-completion baseline, which is what keeps the policy from
    drifting arbitrarily far from SFT.

GRPO (group-relative policy optimization) -- online, group-normalized:

    A_i = (r_i - mean(r_group)) / std(r_group)
    L   = -mean( A_i * logp(y_i) ) + beta * KL_hat(policy || ref)

    Optimizes expected reward directly, using the group mean as the baseline
    instead of a learned value head. Uses *all* K samples per prompt, not just
    the best/worst pair.

The interesting asymmetry, and the reason to compare them on the same task:
DPO discards a group that is uniformly right or uniformly wrong (no pair to
form), while GRPO discards it too (zero variance -> zero advantage) but for a
different reason. Both therefore depend on the policy being *partially*
correct -- which is why sft.py deliberately stops short of solving the task.
"""
import argparse
import csv
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from policy import build_policy, completion_logprobs
from rollouts import rollout, build_preference_pairs, group_advantages, rollout_stats
from sft import eval_accuracy, set_seed
from task import make_prompt_batch


def load_reference(ckpt_path, device):
    """Frozen copy of the SFT policy. Both methods regularize toward it."""
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    ref = build_policy(a["d_model"], a["n_heads"], a["n_layers"], dropout=0.0, device=device)
    ref.load_state_dict(ckpt["model_state"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref, a


def dpo_loss(policy, ref, chosen, rejected, beta):
    pol_c = completion_logprobs(policy, chosen)
    pol_r = completion_logprobs(policy, rejected)
    with torch.no_grad():
        ref_c = completion_logprobs(ref, chosen)
        ref_r = completion_logprobs(ref, rejected)

    logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    # fraction of pairs the policy already ranks correctly -- the standard
    # DPO training diagnostic, and more interpretable than the loss value
    acc = (logits > 0).float().mean().item()
    return loss, acc


def grpo_loss(policy, ref, seqs, rewards, beta):
    B, G, L = seqs.shape
    flat = seqs.view(B * G, L)

    pol_lp = completion_logprobs(policy, flat)
    with torch.no_grad():
        ref_lp = completion_logprobs(ref, flat)

    adv = group_advantages(rewards).view(-1)
    pg_loss = -(adv * pol_lp).mean()

    # single-sample estimator of KL(policy || ref) on the sampled completions
    kl = (pol_lp - ref_lp).mean()
    loss = pg_loss + beta * kl
    return loss, pg_loss.item(), kl.item()


def run_preference_optimization(args) -> dict:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed + 1)
    eval_seed = args.seed + 10_000  # SAME fixed held-out set sft.py scores on

    ref, sft_args = load_reference(args.sft_ckpt, device)

    policy = build_policy(sft_args["d_model"], sft_args["n_heads"], sft_args["n_layers"],
                           dropout=0.0, device=device)
    policy.load_state_dict(torch.load(args.sft_ckpt, map_location=device)["model_state"])
    policy.train()

    # scored on the identical fixed held-out set sft.py uses, so this number
    # is exactly the SFT summary's final_accuracy for the same checkpoint
    start_acc = eval_accuracy(policy, eval_seed, device)
    print(f"[{args.method}] starting from SFT accuracy {start_acc:.3f} | device: {device}")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    log_path = os.path.join(args.out_dir, f"log_{args.method}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["step", "loss", "aux", "mean_reward", "usable_group_frac", "eval_acc"])

    t0 = time.time()
    for step in range(1, args.steps + 1):
        prompts, _ = make_prompt_batch(args.batch_size, rng, device)
        seqs, rewards = rollout(policy, prompts, args.group_size, args.temperature)

        if args.method == "dpo":
            chosen, rejected, n_pairs = build_preference_pairs(seqs, rewards)
            if n_pairs == 0:
                continue  # no signal this step
            loss, aux = dpo_loss(policy, ref, chosen, rejected, args.beta)
            usable = n_pairs / rewards.size(0)
        else:
            loss, aux, _kl = grpo_loss(policy, ref, seqs, rewards, args.beta)
            usable = (rewards.max(dim=1).values > rewards.min(dim=1).values).float().mean().item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            acc = eval_accuracy(policy, eval_seed, device)
            mean_r = rewards.mean().item()
            print(f"[{args.method}] step {step:5d} | loss {loss.item():+.4f} | aux {aux:+.4f} "
                  f"| mean_r {mean_r:.3f} | usable {usable:.2f} | eval_acc {acc:.3f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, loss.item(), aux, mean_r, usable, acc])

    final_acc = eval_accuracy(policy, eval_seed, device)
    stats = rollout_stats(policy, 4, args.batch_size, args.group_size, random.Random(eval_seed), device)

    ckpt_path = os.path.join(args.out_dir, f"model_{args.method}.pt")
    torch.save({"model_state": policy.state_dict(), "args": sft_args,
                "stage": args.method}, ckpt_path)

    summary = {
        "stage": args.method, "seed": args.seed, "steps": args.steps,
        "beta": args.beta, "group_size": args.group_size, "lr": args.lr,
        "sft_accuracy": start_acc, "final_accuracy": final_acc,
        "delta_vs_sft": final_acc - start_acc,
        "final_mean_reward": stats["mean_reward"],
        "final_usable_group_frac": stats["usable_group_frac"],
        "training_time_s": time.time() - t0, "checkpoint": ckpt_path, "log": log_path,
    }
    with open(os.path.join(args.out_dir, f"summary_{args.method}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{args.method}] SFT {start_acc:.3f} -> {args.method.upper()} {final_acc:.3f} "
          f"(delta {final_acc - start_acc:+.3f})")
    return summary


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["dpo", "grpo"], required=True)
    p.add_argument("--sft_ckpt", type=str, default="results/run1/model_sft.pt")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=32, help="prompts per step")
    p.add_argument("--group_size", type=int, default=8, help="completions sampled per prompt")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1,
                    help="DPO: preference sharpness. GRPO: KL penalty weight.")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="results/run1")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run_preference_optimization(build_arg_parser().parse_args())
