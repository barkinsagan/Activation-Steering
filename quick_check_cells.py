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


# ── Section 4: pooled k-fold CV on val+test ──────────────────────────────
# Workaround for the small-test-set sampling-variance problem identified
# in the baseline t-test: pool val and test questions, run k-fold CV on the
# pool. We lose the "true held-out" guarantee, but with σ≈22 on baseline
# logprobs and test n=25, a single random test draw is too noisy to trust
# anyway. Pooled CV gives a stable per-cell mean and a fold-stability check.
K_FOLDS = 5
N_BOOT  = 1000
SEED    = 42

print("=" * 84)
print(f"POOLED CROSS-VALIDATION  (val+test pooled, {K_FOLDS}-fold + bootstrap)")
print("  Workaround for small-n test set with heterogeneous baselines.")
print("  Per cell: per-fold Δc, stability check, and bootstrap 95% CI on the")
print("  pooled-set mean.")
print("=" * 84)
print(
    f"{'cell':>12}  {'fold':>4}  {'n':>3}  {'Dc':>9}  {'Dw':>9}  {'Dc-Dw':>9}"
)
print("-" * 84)

rng = np.random.default_rng(SEED)

for layer, coef in CELLS:
    cell = cf[(cf.layer == layer) & (abs(cf.coef - coef) < 1e-6)]
    pool = cell[cell.split.isin(["validation", "test"])].copy()
    if pool.empty:
        print(f"L{layer:02d} {coef:+.2f}: no val+test data")
        continue

    qids = pool["question_id"].unique()
    rng.shuffle(qids)
    folds = np.array_split(qids, K_FOLDS)

    fold_dc, fold_dw, fold_sel = [], [], []
    label = f"L{layer:02d} {coef:+.2f}"

    for fi, held in enumerate(folds, 1):
        f_sub = pool[pool.question_id.isin(held)]
        dc = f_sub["delta_target_sum_lp"].mean()
        dw = f_sub["delta_max_wrong_sum_lp"].mean()
        fold_dc.append(dc)
        fold_dw.append(dw)
        fold_sel.append(dc - dw)
        print(
            f"{label:>12}  {fi:>4}  {len(f_sub):>3}  "
            f"{dc:+.4f}  {dw:+.4f}  {dc-dw:+.4f}"
        )

    # Stability summary
    mn_dc, sd_dc = float(np.mean(fold_dc)), float(np.std(fold_dc, ddof=1))
    mn_dw, sd_dw = float(np.mean(fold_dw)), float(np.std(fold_dw, ddof=1))
    mn_sel       = float(np.mean(fold_sel))
    n_pos_dc     = sum(d > 0 for d in fold_dc)
    n_neg_dw     = sum(d < 0 for d in fold_dw)
    n_pos_sel    = sum(s > 0 for s in fold_sel)

    print(
        f"{label:>12}  {'mean':>4}        "
        f"{mn_dc:+.4f}  {mn_dw:+.4f}  {mn_sel:+.4f}"
    )
    print(
        f"{label:>12}  {'std':>4}         "
        f"{sd_dc:.4f}   {sd_dw:.4f}   {(sd_dc**2 + sd_dw**2)**0.5:.4f}"
    )
    print(
        f"  Fold consistency: Δc>0 in {n_pos_dc}/{K_FOLDS} folds, "
        f"Δw<0 in {n_neg_dw}/{K_FOLDS}, Δc-Δw>0 in {n_pos_sel}/{K_FOLDS}"
    )

    # Bootstrap 95% CI on the pooled per-question Δc and Δw
    dc_arr = pool["delta_target_sum_lp"].dropna().values
    dw_arr = pool["delta_max_wrong_sum_lp"].dropna().values
    if len(dc_arr) == 0 or len(dw_arr) == 0:
        print("  Bootstrap: no data")
    else:
        # Use the SAME bootstrap indices for Δc, Δw to preserve per-question pairing
        boot_dc, boot_dw, boot_sel = [], [], []
        n = len(dc_arr)
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            boot_dc.append(dc_arr[idx].mean())
            boot_dw.append(dw_arr[idx].mean())
            boot_sel.append((dc_arr[idx] - dw_arr[idx]).mean())
        lo_dc, hi_dc   = np.percentile(boot_dc,  [2.5, 97.5])
        lo_dw, hi_dw   = np.percentile(boot_dw,  [2.5, 97.5])
        lo_se, hi_se   = np.percentile(boot_sel, [2.5, 97.5])

        def _verdict(lo: float, hi: float, want_positive: bool) -> str:
            if lo > 0 and want_positive:  return "fully positive ✓"
            if hi < 0 and not want_positive: return "fully negative ✓"
            return "crosses zero"

        print(
            f"  Bootstrap 95% CI (n_pool={n}, B={N_BOOT}):\n"
            f"    Δc      : [{lo_dc:+.4f}, {hi_dc:+.4f}]  {_verdict(lo_dc, hi_dc, True)}\n"
            f"    Δw      : [{lo_dw:+.4f}, {hi_dw:+.4f}]  {_verdict(lo_dw, hi_dw, False)}\n"
            f"    Δc - Δw : [{lo_se:+.4f}, {hi_se:+.4f}]  {_verdict(lo_se, hi_se, True)}"
        )
    print()
