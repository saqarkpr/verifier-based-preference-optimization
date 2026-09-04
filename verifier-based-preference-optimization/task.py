"""
A small task with an EXACT programmatic verifier.

Preference optimization normally needs human preference labels, which are
expensive, noisy, and impossible to reproduce. Here the reward is a Python
function: two-digit addition, where a completion is correct iff it states the
right sum. That removes the label-noise confound entirely, so any difference
between SFT / DPO / GRPO is attributable to the *optimization method* rather
than to disagreement about what a good answer is.

Format (character-level, fixed width so batching is trivial):

    prompt      "47+38="
    completion  "085<eos>"      <- always 3 digits, zero-padded

Zero-padding the answer to a fixed width means every completion has the same
token length, so a sampled completion is either right or wrong with no
length-bias term needed in the reward -- one fewer confound.
"""
import random
import torch

# vocabulary: digits, operator, equals, eos, pad
CHARS = list("0123456789+=") + ["<eos>", "<pad>"]
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(CHARS)
PAD_ID = STOI["<pad>"]
EOS_ID = STOI["<eos>"]

PROMPT_LEN = 6      # "47+38="
ANSWER_LEN = 3      # "085"
COMPLETION_LEN = ANSWER_LEN + 1  # + <eos>
SEQ_LEN = PROMPT_LEN + COMPLETION_LEN


def encode(s: str):
    """Encode a plain string (no special tokens) to ids."""
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[i] for i in ids if ITOS[i] not in ("<pad>",))


def make_prompt(a: int, b: int) -> str:
    return f"{a:02d}+{b:02d}="


def make_answer(a: int, b: int) -> str:
    return f"{a + b:03d}"


def sample_problem(rng: random.Random):
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    return a, b


def verify(prompt_str: str, completion_str: str) -> float:
    """The reward function. Returns 1.0 if the completion states the correct
    sum for this prompt, else 0.0. Deliberately strict: any malformed output
    scores 0, so the model cannot be rewarded for plausible-looking garbage."""
    try:
        lhs = prompt_str.replace("=", "")
        a_str, b_str = lhs.split("+")
        target = int(a_str) + int(b_str)
    except (ValueError, AttributeError):
        return 0.0

    digits = completion_str.replace("<eos>", "")
    if len(digits) != ANSWER_LEN or not digits.isdigit():
        return 0.0
    return 1.0 if int(digits) == target else 0.0


def make_sft_batch(batch_size: int, rng: random.Random, device: str = "cpu"):
    """Full (prompt + correct answer) sequences for supervised fine-tuning.

    Returns x, y for next-token prediction, plus a completion mask so the loss
    can be restricted to the answer tokens only -- training the model to
    predict the prompt back to itself would waste capacity on a task it is
    never evaluated on.
    """
    seqs, comp_masks = [], []
    for _ in range(batch_size):
        a, b = sample_problem(rng)
        ids = encode(make_prompt(a, b)) + encode(make_answer(a, b)) + [EOS_ID]
        seqs.append(ids)
        # mask marks positions whose *target* is a completion token
        mask = [0] * (PROMPT_LEN - 1) + [1] * COMPLETION_LEN
        comp_masks.append(mask)

    seqs = torch.tensor(seqs, dtype=torch.long, device=device)
    comp_masks = torch.tensor(comp_masks, dtype=torch.bool, device=device)
    x, y = seqs[:, :-1], seqs[:, 1:]
    return x, y, comp_masks


def make_prompt_batch(batch_size: int, rng: random.Random, device: str = "cpu"):
    """Prompts only -- the starting point for on-policy sampling."""
    prompts, meta = [], []
    for _ in range(batch_size):
        a, b = sample_problem(rng)
        p = make_prompt(a, b)
        prompts.append(encode(p))
        meta.append(p)
    return torch.tensor(prompts, dtype=torch.long, device=device), meta


if __name__ == "__main__":
    rng = random.Random(0)
    a, b = sample_problem(rng)
    p, ans = make_prompt(a, b), make_answer(a, b)
    print(f"prompt={p!r} answer={ans!r}")
    print("verify(correct):", verify(p, ans + "<eos>"))
    print("verify(wrong):  ", verify(p, "999<eos>"))
    print("verify(malformed):", verify(p, "7<eos>"))

    x, y, m = make_sft_batch(4, rng)
    print("sft batch:", x.shape, y.shape, m.shape)
    print("decoded seq 0:", decode(x[0].tolist()), "->", decode(y[0].tolist()))
    print("vocab size:", VOCAB_SIZE)
