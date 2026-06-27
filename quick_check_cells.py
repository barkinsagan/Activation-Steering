import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DEFAULT_EXP = "results/exp_20260626_phys_vs_bio_continuous_full"

parser = argparse.ArgumentParser(
    description="Pooled CV + bootstrap quick-check on a steering experiment."
)
parser.add_argument(
    "exp",
    nargs="?",
    default=DEFAULT_EXP,
    help=f"Path to experiment results dir (default: {DEFAULT_EXP})",
)
args = parser.parse_args()

exp = Path(args.exp)
if not exp.exists():
    sys.exit(f"[!] Experiment dir not found: {exp}")
if not (exp / "cf").exists():
    sys.exit(f"[!] No cf/ subdirectory in {exp}; expected per-layer detailed_wide.csv files")
if not (exp / "split_manifest.csv").exists():
    sys.exit(f"[!] No split_manifest.csv in {exp}; cannot tag val/test rows")
print(f"Experiment: {exp}")

# Cells to compare val vs test on Δ-correct / Δ-wrong (kept narrow for table width)
CELLS = [
    (11, 0.25), (11, 0.50),
    (12, 0.25), (12, 0.50), (12, 0.75),
    (13, 0.25), (13, 0.50),
    (14, 0.25), (14, 0.50),
    (16, 0.25), (16, 0.50), (16, 0.75),
    (17, 0.50), (17, 0.75),
    (20, 1.00),
    (25, 1.00), (25, 3.00),
]

# Bands for pooled-layer cross-validation. Each entry: (name, layers, coefs).
# At each (band, coef) we average per-question Δc and Δw across the band's
# layers, then CV + bootstrap on the per-question averaged values. This lifts
# the SNR if the same vector behaves coherently across the band.
BANDS = [
    ("L11-L13 divergence-peak",   [11, 12, 13],     [0.25, 0.50, 0.75]),
    ("L14-L17 target-aware",      [14, 15, 16, 17], [0.25, 0.50, 0.75]),
    ("L19-L21 sharpening",        [19, 20, 21],     [0.50, 0.75, 1.00, 1.50]),
    ("L25-L26 late directional",  [25, 26],         [1.00, 1.50, 2.00, 3.00]),
]

K_FOLDS = 5
N_BOOT  = 1000
SEED    = 42

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


def _best_wrong_at_zero(sub: pd.DataFrame) -> pd.Series:
    cols = ["false1_sum_lp", "false2_sum_lp", "false3_sum_lp"]
    return sub[cols].max(axis=1)


def _verdict(lo: float, hi: float, want_positive: bool) -> str:
    if want_positive and lo > 0:      return "fully positive ✓"
    if (not want_positive) and hi < 0: return "fully negative ✓"
    return "crosses zero"


def _cv_and_bootstrap(per_q_dc: np.ndarray, per_q_dw: np.ndarray,
                      label: str, rng: np.random.Generator):
    """Run K-fold CV + bootstrap on per-question Δc / Δw arrays. Prints results."""
    n = len(per_q_dc)
    if n < K_FOLDS:
        print(f"  {label}: too few questions ({n}) for {K_FOLDS}-fold CV")
        return

    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, K_FOLDS)

    fold_dc, fold_dw, fold_sel = [], [], []
    for fi, held in enumerate(folds, 1):
        dc = float(per_q_dc[held].mean())
        dw = float(per_q_dw[held].mean())
        fold_dc.append(dc); fold_dw.append(dw); fold_sel.append(dc - dw)

    n_pos_dc  = sum(d > 0 for d in fold_dc)
    n_neg_dw  = sum(d < 0 for d in fold_dw)
    n_pos_sel = sum(s > 0 for s in fold_sel)

    boot_dc, boot_dw, boot_sel = [], [], []
    for _ in range(N_BOOT):
        bi = rng.integers(0, n, size=n)
        boot_dc.append(per_q_dc[bi].mean())
        boot_dw.append(per_q_dw[bi].mean())
        boot_sel.append((per_q_dc[bi] - per_q_dw[bi]).mean())
    lo_dc, hi_dc = np.percentile(boot_dc,  [2.5, 97.5])
    lo_dw, hi_dw = np.percentile(boot_dw,  [2.5, 97.5])
    lo_se, hi_se = np.percentile(boot_sel, [2.5, 97.5])

    mean_dc  = float(np.mean(fold_dc))
    mean_dw  = float(np.mean(fold_dw))
    mean_sel = float(np.mean(fold_sel))

    print(
        f"  {label}  n_pool={n}  "
        f"meanΔc={mean_dc:+.4f}  meanΔw={mean_dw:+.4f}  meanSel={mean_sel:+.4f}"
    )
    print(
        f"    folds:  Δc>0 in {n_pos_dc}/{K_FOLDS}, "
        f"Δw<0 in {n_neg_dw}/{K_FOLDS}, "
        f"Δc-Δw>0 in {n_pos_sel}/{K_FOLDS}"
    )
    print(
        f"    Δc      : [{lo_dc:+.4f}, {hi_dc:+.4f}]  {_verdict(lo_dc, hi_dc, True)}"
    )
    print(
        f"    Δw      : [{lo_dw:+.4f}, {hi_dw:+.4f}]  {_verdict(lo_dw, hi_dw, False)}"
    )
    print(
        f"    Δc - Δw : [{lo_se:+.4f}, {hi_se:+.4f}]  {_verdict(lo_se, hi_se, True)}"
    )


# ── Section 1: Δc / Δw at selected (layer, coef) cells (val vs test) ────────
print("=" * 84)
print("SECTION 1: DELTA-CORRECT / DELTA-WRONG  (val vs test, per cell)")
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
print("=" * 84)
print("SECTION 2: BASELINE COMPARISON  (coef = 0, val vs test, per layer)")
print("=" * 84)
print(
    f"{'layer':>5}  {'split':>11}  {'n':>3}  "
    f"{'tgt_mean':>9}  {'tgt_std':>8}  "
    f"{'bw_mean':>9}  {'bw_std':>8}  "
    f"{'margin':>8}  {'mar_std':>8}"
)
print("-" * 84)

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
print("=" * 84)
print("SECTION 3: BASELINE T-TEST  (Welch's t, val vs test, per layer)")
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


# ── Section 4: pooled CV + bootstrap at each individual cell ────────────
print("=" * 84)
print(f"SECTION 4: POOLED CV + BOOTSTRAP, PER CELL  ({K_FOLDS}-fold, B={N_BOOT})")
print("  Pools val+test, runs CV and bootstrap on per-question Δc / Δw.")
print("=" * 84)

rng = np.random.default_rng(SEED)

for layer, coef in CELLS:
    cell = cf[(cf.layer == layer) & (abs(cf.coef - coef) < 1e-6)]
    pool = cell[cell.split.isin(["validation", "test"])].copy()
    if pool.empty:
        print(f"L{layer:02d} {coef:+.2f}: no val+test data\n"); continue

    per_q_dc = pool["delta_target_sum_lp"].values.astype(float)
    per_q_dw = pool["delta_max_wrong_sum_lp"].values.astype(float)
    label = f"L{layer:02d} {coef:+.2f}"
    _cv_and_bootstrap(per_q_dc, per_q_dw, label, rng)
    print()


# ── Section 5: BAND-pooled CV + bootstrap ──────────────────────────────
# For each (band, coef): per question, average Δc and Δw across the band's
# layers, then CV+bootstrap on those per-question averages. If the band moves
# coherently with the vector, this should clear CI bars that no single cell
# could clear at n=76.
print("=" * 84)
print("SECTION 5: BAND-POOLED CV + BOOTSTRAP  (per-question averaged across band)")
print("  Each per-question observation = mean Δc / Δw across the band's layers.")
print("  Lifts SNR if the band behaves coherently; collapses noise across layers.")
print("=" * 84)

for band_name, layers, coefs in BANDS:
    print(f"\n── Band: {band_name}  (layers {layers}) ──")
    for coef in coefs:
        sub = cf[
            (cf.layer.isin(layers))
            & (abs(cf.coef - coef) < 1e-6)
            & (cf.split.isin(["validation", "test"]))
        ].copy()
        if sub.empty:
            print(f"  coef={coef:+.2f}: no data"); continue

        # Per-question average across the band's layers
        agg = sub.groupby("question_id").agg(
            dc=("delta_target_sum_lp",    "mean"),
            dw=("delta_max_wrong_sum_lp", "mean"),
            n_layers=("layer",            "nunique"),
        ).reset_index()
        # Only keep questions with full band coverage
        agg = agg[agg["n_layers"] == len(layers)]
        if agg.empty:
            print(f"  coef={coef:+.2f}: no questions with full band coverage"); continue

        per_q_dc = agg["dc"].values
        per_q_dw = agg["dw"].values
        label = f"coef={coef:+.2f}"
        _cv_and_bootstrap(per_q_dc, per_q_dw, label, rng)
        print()
