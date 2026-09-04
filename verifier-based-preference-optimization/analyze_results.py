"""
Correct statistical analysis of the SFT / DPO / GRPO comparison.

`run_pipeline.py` prints each method's mean and std independently. That is the
WRONG analysis for this design, and it is wrong in a way that loses power.

Both methods branch from the *same* SFT checkpoint on every seed, so the
per-seed SFT quality is a shared nuisance variable: a seed whose SFT run
happened to land high will push DPO and GRPO up together. Comparing the two
marginal distributions throws that pairing away and buries a consistent
difference inside between-seed variance.

The paired difference (GRPO - DPO computed within each seed, then averaged)
cancels the shared SFT term and is the estimator the experimental design
actually supports.

    python analyze_results.py --csv results/comparison.csv
"""
import argparse
import csv
import statistics as st
from math import comb, sqrt


def read_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["seed"] in ("mean", "std"):
                continue
            rows.append({k: (int(v) if k == "seed" else float(v)) for k, v in r.items()})
    return rows


def _corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else 0.0


def sigma_str(delta, sigma):
    return "n/a" if sigma == 0 else f"{abs(delta) / sigma:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/comparison.csv")
    args = p.parse_args()

    rows = read_rows(args.csv)
    n = len(rows)
    sft = [r["sft_acc"] for r in rows]
    dpo = [r["dpo_acc"] for r in rows]
    grpo = [r["grpo_acc"] for r in rows]

    if n < 2:
        raise SystemExit("need at least 2 seeds; run with --n_seeds 3")

    m_s, s_s = st.mean(sft), st.stdev(sft)
    m_d, s_d = st.mean(dpo), st.stdev(dpo)
    m_g, s_g = st.mean(grpo), st.stdev(grpo)

    print(f"seeds: {n}\n")
    print("1) Marginal accuracies")
    print(f"   SFT   {m_s:.3f} +/- {s_s:.3f}")
    print(f"   DPO   {m_d:.3f} +/- {s_d:.3f}")
    print(f"   GRPO  {m_g:.3f} +/- {s_g:.3f}")

    print("\n2) Does preference optimization beat SFT?")
    for name, m in (("DPO", m_d), ("GRPO", m_g)):
        d = m - m_s
        print(f"   SFT -> {name:4s}  {d:+.3f}   ({sigma_str(d, s_s)} sigma of the SFT spread)")

    print("\n3) DPO vs GRPO -- unpaired (the misleading view)")
    gap = m_g - m_d
    print(f"   GRPO - DPO  {gap:+.4f}   ({sigma_str(gap, s_d)} sigma of DPO's own spread)")
    print("   Read alone this says: indistinguishable.")

    print("\n4) DPO vs GRPO -- paired (the view the design supports)")
    diffs = [g - d for g, d in zip(grpo, dpo)]
    md = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    wins = sum(d > 0 for d in diffs)
    for r, d in zip(rows, diffs):
        print(f"   seed {r['seed']}: {d:+.4f}")
    print(f"   mean {md:+.4f} +/- {sd:.4f}")
    if sd > 0:
        t = md / (sd / n ** 0.5)
        print(f"   paired t = {t:.2f}, df = {n-1}")
    print(f"   GRPO ahead on {wins}/{n} seeds")

    if sd > 0:
        wins_p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n
        print(f"   sign test: p = {wins_p:.5f} (one-sided)")

    print("\n5) Does the gain depend on how good the SFT checkpoint was?")
    gains = [g - s_ for g, s_ in zip(grpo, sft)]
    r_gain = _corr(sft, gains)
    r_final = _corr(sft, grpo)
    print(f"   corr(SFT, gain)          = {r_gain:+.3f}  <- PARTLY ARTIFACTUAL:")
    print("      gain = final - SFT, so SFT appears on both sides with opposite")
    print("      signs. A noisy SFT measurement alone produces a negative")
    print("      correlation here even if there is no real effect.")
    print(f"   corr(SFT, final accuracy) = {r_final:+.3f}  <- the clean test")
    if n > 2:
        t_final = r_final * sqrt((n - 2) / max(1 - r_final ** 2, 1e-12))
        print(f"      t = {t_final:.2f}, df = {n-2}")

    print("\n6) Verdict")
    sft_gain_sigma = sigma_str(m_d - m_s, s_s)
    print(f"   - SFT -> preference optimization: {sft_gain_sigma} sigma"
          f"{' of the SFT spread.' if sft_gain_sigma != 'n/a' else ' (SFT std is 0 -- too few/too-degenerate seeds to judge; this is a smoke-test-scale run, not a real result).'}")
    if wins == n and sd > 0:
        wins_p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n
        if wins_p < 0.01:
            print(f"   - GRPO leads on all {n}/{n} seeds by {md:+.4f} (sign test p = {wins_p:.5f}).")
            print("     This IS a finding. Note the unpaired view above calls the same")
            print("     data noise -- pairing is what makes the effect visible.")
        else:
            print(f"   - GRPO leads on all {n} seeds, but n={n} gives sign-test p = {wins_p:.3f}:")
            print("     directionally suggestive, not yet a finding.")
    else:
        print(f"   - GRPO ahead on {wins}/{n}: no consistent direction.")


if __name__ == "__main__":
    main()
