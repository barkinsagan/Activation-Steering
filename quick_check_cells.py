import numpy as np
import pandas as pd
from pathlib import Path

exp = Path("results/exp_20260626_phys_vs_bio_continuous_full")

# Cells to compare val vs test on Δ-correct / Δ-wrong
CELLS = [
    (11, 0.25),
    (12, 0.25),
    (13, 0.25),
    (14, 0.50),
    (16, 0.25),
    (20, 1.00),
    (25, 3.00),
]

# Layers to compare *baselines* (coef = 0) on val vs test
BASELINE_LAYERS = sorted({l for l, _ in CELLS})

# ── Load CF data ─────────────────────────────────────────────────────────────
dfs = []
for d in sorted((exp / "cf").glob("layer_*")):
    df = pd.read_csv(d / "detailed_wide.csv")
    df["layer"] = int(d.name.split("_")[1])
    dfs.append(df)
cf = pd.concat(dfs, ignore_index=True)

manifest = pd.read_csv(exp / "split_manifest.csv")
cf = cf.merge(
    manifest[["eval_question_id", "split"]].rename(columns={"eval_question_id": "question_id"}),
    on="question_id",
    how="left",
)

# ── Section 1: Δc / Δw at selected (layer, coef) cells ──────────────────────
print("=" * 84)
print("DELTA-CORRECT / DELTA-WRONG  (val vs test, selected cells)")
print("=" * 84)
print(
    f"{'cell':>12}  {'split':>11}  {'n':>3}  {'Dc':>9}  {'SE_Dc':>7}  "
    f"{'Dw':>9}  {'SE_Dw':>7}  {'Dc-Dw':>9}"
)
print("-" * 84)

for layer, coef in CELLS:
    cell = cf[(cf.layer == layer) & (abs(cf.coef - coef) < 1e-6)]
    label = f"L{layer:02d} {coef:+.2f}"
    for s in ["validation", "test"]:
        sub = cell[cell.split == s]
        if sub.empty:
            print(f"{label:>12}  {s:>11}  (no data)")
            continue
        dc = sub["delta_target_sum_lp"]
        dw = sub["delta_max_wrong_sum_lp"]
        n = len(sub)
        se_dc = dc.std() / n**0.5 if n else float("nan")
        se_dw = dw.std() / n**0.5 if n else float("nan")
        print(
            f"{label:>12}  {s:>11}  {n:>3}  "
            f"{dc.mean():+.4f}  {se_dc:.4f}  "
            f"{dw.mean():+.4f}  {se_dw:.4f}  "
            f"{(dc - dw).mean():+.4f}"
        )
    print()

# ── Section 2: baseline (coef = 0) comparison val vs test ─────────────────
# If val and test baselines diverge systematically, "Δ" comparisons across
# splits aren't apples-to-apples — the steered logprob is being subtracted
# from different baseline distributions. Suspect heterogeneity if:
#   - mean target_sum_lp differs by > 1σ between val and test
#   - mean best-wrong differs by > 1σ
#   - per-question margin (target − best-wrong) distribution shifts
print("=" * 84)
print("BASELINE COMPARISON  (coef = 0, val vs test, per layer)")
print("  If splits draw from different question distributions, baseline target")
print("  and best-wrong logprobs will differ. That would explain Δw flips.")
print("=" * 84)
print(
    f"{'layer':>5}  {'split':>11}  {'n':>3}  "
    f"{'tgt_mean':>9}  {'tgt_std':>8}  "
    f"{'bw_mean':>9}  {'bw_std':>8}  "
    f"{'margin':>8}  {'mar_std':>8}"
)
print("-" * 84)


def _best_wrong_at_zero(sub: pd.DataFrame) -> pd.Series:
    """Per-row best wrong logprob = max(false1, false2, false3) at coef=0."""
    cols = ["false1_sum_lp", "false2_sum_lp", "false3_sum_lp"]
    return sub[cols].max(axis=1)


for layer in BASELINE_LAYERS:
    base = cf[(cf.layer == layer) & (abs(cf.coef) < 1e-6)]
    for s in ["validation", "test"]:
        sub = base[base.split == s]
        if sub.empty:
            print(f"  L{layer:02d}  {s:>11}  (no data)")
            continue
        tgt = sub["target_sum_lp"]
        bw  = _best_wrong_at_zero(sub)
        margin = tgt - bw
        print(
            f"  L{layer:02d}  {s:>11}  {len(sub):>3}  "
            f"{tgt.mean():+.3f}  {tgt.std():7.3f}  "
            f"{bw.mean():+.3f}  {bw.std():7.3f}  "
            f"{margin.mean():+.3f}  {margin.std():7.3f}"
        )
    print()

# ── Section 3: 2-sample t-tests on baseline distributions ────────────────
# Cleanest "are splits comparable?" check: do val and test baselines have
# significantly different means? If so, deltas across splits aren't directly
# comparable.
print("=" * 84)
print("BASELINE T-TEST  (Welch's t-test, val vs test, per layer)")
print("  H0: same population mean for val and test. p<0.05 = splits not comparable.")
print("=" * 84)
print(
    f"{'layer':>5}  {'metric':>10}  "
    f"{'val_mean':>9}  {'test_mean':>10}  {'diff':>8}  "
    f"{'t':>6}  {'p':>7}"
)
print("-" * 84)

try:
    from scipy.stats import ttest_ind

    for layer in BASELINE_LAYERS:
        base = cf[(cf.layer == layer) & (abs(cf.coef) < 1e-6)]
        val = base[base.split == "validation"]
        tst = base[base.split == "test"]
        if val.empty or tst.empty:
            continue
        for metric_name, getter in [
            ("target",     lambda d: d["target_sum_lp"]),
            ("best_wrong", _best_wrong_at_zero),
            ("margin",     lambda d: d["target_sum_lp"] - _best_wrong_at_zero(d)),
        ]:
            v, t = getter(val).values, getter(tst).values
            res = ttest_ind(v, t, equal_var=False)
            sig = " *" if res.pvalue < 0.05 else "  "
            print(
                f"  L{layer:02d}  {metric_name:>10}  "
                f"{v.mean():+.3f}  {t.mean():+.4f}  {t.mean()-v.mean():+.3f}  "
                f"{res.statistic:+6.2f}  {res.pvalue:.4f}{sig}"
            )
        print()
except ImportError:
    print("  scipy not available — skipping t-tests")
