"""
On-policy rollouts scored by the programmatic verifier.

Shared by both DPO and GRPO so the two methods consume *identical* sampling
machinery. If each script sampled its own way, any measured difference between
DPO and GRPO could be an artifact of differing rollout code rather than of the
objectives themselves -- which is exactly the confound this file exists to
remove.
"""
import torch

from policy import sample_completions, completion_strings, prompt_strings
from task import verify, make_prompt_batch


@torch.no_grad()
def rollout(model, prompts: torch.Tensor, group_size: int, temperature: float = 1.0):
    """Sample `group_size` completions for each prompt and score them.

    prompts: (B, PROMPT_LEN)
    returns:
        seqs:    (B, group_size, SEQ_LEN)
        rewards: (B, group_size) float 0/1 from the verifier
    """
    B, P = prompts.shape
    # repeat_interleave so rows are grouped by prompt: [p0,p0,...,p1,p1,...]
    repeated = prompts.repeat_interleave(group_size, dim=0)
    seqs = sample_completions(model, repeated, temperature=temperature)

    p_strs = prompt_strings(seqs)
    c_strs = completion_strings(seqs)
    rewards = torch.tensor(
        [verify(p, c) for p, c in zip(p_strs, c_strs)],
        dtype=torch.float32, device=prompts.device,
    )

    seqs = seqs.view(B, group_size, -1)
    rewards = rewards.view(B, group_size)
    return seqs, rewards


def build_preference_pairs(seqs: torch.Tensor, rewards: torch.Tensor):
    """Turn scored rollouts into (chosen, rejected) pairs for DPO.

    For each prompt group, pair the highest-reward completion against the
    lowest-reward one. A group where every sample scored the same (all correct
    or all wrong) carries no preference signal and is dropped -- reporting how
    often that happens matters, since it is the real sample efficiency of
    verifier-based DPO.

    returns: chosen (N, SEQ_LEN), rejected (N, SEQ_LEN), n_usable_groups
    """
    chosen, rejected = [], []
    B, G = rewards.shape
    for b in range(B):
        r = rewards[b]
        if r.max() <= r.min():
            continue  # no signal in this group
        chosen.append(seqs[b, r.argmax()])
        rejected.append(seqs[b, r.argmin()])

    if not chosen:
        return None, None, 0
    return torch.stack(chosen), torch.stack(rejected), len(chosen)


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Group-relative advantages used by GRPO.

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

    Normalizing within the prompt group is the whole point of GRPO: it removes
    the need for a learned value baseline, because the group mean *is* the
    baseline. Groups with zero variance produce zero advantage and contribute
    no gradient, which is the intended behaviour.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


@torch.no_grad()
def rollout_stats(model, n_batches, batch_size, group_size, rng, device, temperature=1.0):
    """Diagnostics on the current policy's rollouts: mean reward, and the
    fraction of prompt groups that yield a usable preference pair."""
    total_reward, total_samples, usable, groups = 0.0, 0, 0, 0
    for _ in range(n_batches):
        prompts, _ = make_prompt_batch(batch_size, rng, device)
        seqs, rewards = rollout(model, prompts, group_size, temperature)
        total_reward += rewards.sum().item()
        total_samples += rewards.numel()
        _, _, n = build_preference_pairs(seqs, rewards)
        usable += n
        groups += rewards.size(0)
    return {
        "mean_reward": total_reward / max(total_samples, 1),
        "usable_group_frac": usable / max(groups, 1),
    }


if __name__ == "__main__":
    import random
    from policy import build_policy

    torch.manual_seed(0)
    model = build_policy(d_model=32, n_heads=2, n_layers=2)
    prompts, _ = make_prompt_batch(6, random.Random(0))

    seqs, rewards = rollout(model, prompts, group_size=4)
    print("seqs:", seqs.shape, "rewards:", rewards.shape)
    print("rewards:\n", rewards)

    adv = group_advantages(rewards)
    print("advantages (zero rows = no signal, as intended):\n", adv.round(decimals=3))

    ch, rj, n = build_preference_pairs(seqs, rewards)
    print(f"usable preference pairs from 6 groups: {n}")
    if n:
        print("chosen shape:", ch.shape, "rejected shape:", rj.shape)

    # a group with mixed rewards must produce a usable pair; an all-equal one must not
    fake_seqs = seqs.clone()
    mixed = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    _, _, n_mixed = build_preference_pairs(fake_seqs[:2], mixed)
    print("sanity: 1 mixed group + 1 flat group ->", n_mixed, "pair(s) (expected 1)")
    assert n_mixed == 1
    print("rollout machinery OK")
