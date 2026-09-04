"""
Policy wrapper around the from-scratch Transformer.

Both DPO and GRPO need the same primitive: the log-probability the policy
assigns to a *specific* completion given a prompt, summed over completion
tokens only. Getting this subtly wrong (off-by-one in the shift, or summing
over prompt tokens too) is the classic silent bug in preference-optimization
code -- it trains without error and produces a model that is quietly
optimizing the wrong objective. `completion_logprobs` is therefore isolated
here, used by every training script, and unit-tested in __main__ against a
brute-force recomputation.
"""
import torch
import torch.nn.functional as F

from model import TransformerFromScratch
from task import (VOCAB_SIZE, PROMPT_LEN, COMPLETION_LEN, SEQ_LEN,
                  EOS_ID, decode)


def build_policy(d_model=64, n_heads=4, n_layers=3, dropout=0.0, device="cpu"):
    return TransformerFromScratch(
        vocab_size=VOCAB_SIZE, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_len=SEQ_LEN, dropout=dropout,
    ).to(device)


def completion_logprobs(model, sequences: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token_t | tokens_<t) over COMPLETION tokens only.

    sequences: (B, SEQ_LEN) full prompt+completion token ids.
    returns:   (B,) summed log-probability of the completion.

    The model predicts position t+1 from position t, so the logits that
    predict completion token at absolute index i live at index i-1. The
    completion occupies absolute indices [PROMPT_LEN, SEQ_LEN), so the
    predicting logits are at [PROMPT_LEN-1, SEQ_LEN-1).
    """
    logits, _ = model(sequences)                      # (B, SEQ_LEN, V)
    pred_logits = logits[:, PROMPT_LEN - 1:SEQ_LEN - 1, :]   # (B, COMPLETION_LEN, V)
    targets = sequences[:, PROMPT_LEN:SEQ_LEN]               # (B, COMPLETION_LEN)

    logprobs = F.log_softmax(pred_logits, dim=-1)
    token_lp = torch.gather(logprobs, 2, targets.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(dim=-1)


@torch.no_grad()
def sample_completions(model, prompts: torch.Tensor, temperature: float = 1.0,
                        greedy: bool = False) -> torch.Tensor:
    """Autoregressively sample one completion per prompt.

    prompts: (B, PROMPT_LEN)
    returns: (B, SEQ_LEN) prompt with completion appended.

    Generation runs for a fixed COMPLETION_LEN steps rather than stopping at
    <eos>, so every sequence in the batch has identical length. This keeps the
    reward free of any length confound (see task.py) and makes batched
    logprob computation exact with no padding mask.
    """
    was_training = model.training
    model.eval()

    seq = prompts.clone()
    for _ in range(COMPLETION_LEN):
        logits, _ = model(seq)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        if greedy:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_id], dim=1)

    if was_training:
        model.train()
    return seq


def completion_strings(sequences: torch.Tensor):
    """Decode just the completion part of each sequence, for the verifier."""
    return [decode(s[PROMPT_LEN:SEQ_LEN].tolist()) for s in sequences]


def prompt_strings(sequences: torch.Tensor):
    return [decode(s[:PROMPT_LEN].tolist()) for s in sequences]


if __name__ == "__main__":
    import random
    from task import make_prompt_batch

    torch.manual_seed(0)
    device = "cpu"
    model = build_policy(d_model=32, n_heads=2, n_layers=2, device=device)

    prompts, meta = make_prompt_batch(4, random.Random(0), device)
    seqs = sample_completions(model, prompts)
    print("sampled shape:", seqs.shape, "(expected (4,%d))" % SEQ_LEN)
    print("prompts:    ", prompt_strings(seqs))
    print("completions:", completion_strings(seqs))

    # --- correctness check on completion_logprobs -------------------------
    # brute-force the same quantity one token at a time and compare.
    lp_fast = completion_logprobs(model, seqs)

    lp_slow = torch.zeros(seqs.size(0))
    for b in range(seqs.size(0)):
        for i in range(PROMPT_LEN, SEQ_LEN):
            prefix = seqs[b:b + 1, :i]
            logits, _ = model(prefix)
            lp = F.log_softmax(logits[0, -1, :], dim=-1)
            lp_slow[b] += lp[seqs[b, i]]

    print("fast:", lp_fast.detach().numpy().round(4))
    print("slow:", lp_slow.detach().numpy().round(4))
    print("max abs diff:", (lp_fast - lp_slow).abs().max().item())
    assert torch.allclose(lp_fast, lp_slow, atol=1e-4), "logprob mismatch!"
    print("completion_logprobs verified against brute force")
