"""
Post-hoc analysis of MCF steering vector experiment results.

Tiers:
  1  Core stats       — baseline logprob, best setting, logprob grid, question flips
  2  Layer analysis   — best coef per layer, top 3 layers
  3  Deep dive        — per-question breakdown, rank change, logprob percentiles
  4  K-fold CV        — selection stability and sensitivity across rotating test folds

Usage:
    python analysis/analyze_results.py results/<exp_id>/mcf [--k 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def load_data(mcf_dir: Path) -> pd.DataFrame:
    combined = mcf_dir / "combined_results.csv"
    if combined.exists():
        df = pd.read_csv(combined)
    else:
        frames = [pd.read_csv(f) for f in sorted(mcf_dir.glob("layer_*_results.csv"))]
        if not frames:
            raise FileNotFoundError(f"No results found in {mcf_dir}")
        df = pd.concat(frames, ignore_index=True)
        print(f"Assembled {len(frames)} per-layer files → {df.shape}")
    return df


def _baseline_per_question(df: pd.DataFrame) -> pd.Series:
    """Return one baseline logprob per question_id (coef=0, deduped across layers)."""
    return (
        df[df["coef"] == 0.0]
        .groupby("question_id")["correct_label_logprob"]
        .first()
    )


def _sep(char: str = "=", width: int = 72):
    print(char * width)


def _header(title: str):
    print()
    _sep("=")
    print(f"  {title}")
    _sep("=")


def _section(title: str):
    print()
    _sep("-", 60)
    print(f"  {title}")
    _sep("-", 60)


# =============================================================================
# Tier 1 — Core stats
# =============================================================================

def tier1(df: pd.DataFrame):
    _header("TIER 1 — CORE STATS")

    # Baseline: deduplicate across layers (all coef=0 rows are identical per question)
    base_per_q = _baseline_per_question(df)
    baseline_logprob = base_per_q.mean()

    base_any_layer = (
        df[(df["coef"] == 0.0) & (df["layer"] == df["layer"].min())]
        .set_index("question_id")
    )
    baseline_acc = base_any_layer["correct"].mean()
    n_q = df["question_id"].nunique()

    print(f"\n  Baseline (coef = 0)")
    print(f"    Mean correct-label logprob : {baseline_logprob:.4f}")
    print(f"    Accuracy                   : {baseline_acc:.3f}  "
          f"({int(base_any_layer['correct'].sum())} / {n_q} questions)")

    # Best (layer, coef) by mean correct-label logprob
    steered = df[df["coef"] != 0.0]
    lp_by_setting = steered.groupby(["layer", "coef"])["correct_label_logprob"].mean()
    best_layer, best_coef = lp_by_setting.idxmax()
    best_logprob = lp_by_setting[(best_layer, best_coef)]

    best_rows = df[(df["layer"] == best_layer) & (df["coef"] == best_coef)]
    best_acc   = best_rows["correct"].mean()

    print(f"\n  Best setting  (by mean correct-label logprob)")
    print(f"    Layer  : {best_layer}")
    print(f"    Coef   : {best_coef}")
    print(f"    Mean correct-label logprob : {best_logprob:.4f}  "
          f"(Δ {best_logprob - baseline_logprob:+.4f})")
    print(f"    Accuracy                   : {best_acc:.3f}  "
          f"({int(best_rows['correct'].sum())} / {n_q} questions)")

    # Flips vs the same layer's baseline
    base_at_best_layer = (
        df[(df["layer"] == best_layer) & (df["coef"] == 0.0)]
        .set_index("question_id")["correct"]
    )
    best_at_best_layer = best_rows.set_index("question_id")["correct"]
    shared = base_at_best_layer.index.intersection(best_at_best_layer.index)
    wrong_to_correct = int((~base_at_best_layer[shared] &  best_at_best_layer[shared]).sum())
    correct_to_wrong = int(( base_at_best_layer[shared] & ~best_at_best_layer[shared]).sum())

    print(f"\n  Question flips at best setting")
    print(f"    Wrong → Correct : {wrong_to_correct}")
    print(f"    Correct → Wrong : {correct_to_wrong}")
    print(f"    Net             : {wrong_to_correct - correct_to_wrong:+d}")

    # Logprob grid
    _section("Correct-Label Logprob Grid  (layer × coef)")
    layers = sorted(df["layer"].unique())
    coefs  = sorted(df["coef"].unique())

    col_w = 8
    header_row = f"  {'Layer':>6} |" + "".join(f"{c:>{col_w}.2f}" for c in coefs)
    print(header_row)
    print("  " + "-" * (len(header_row) - 2))

    for layer in layers:
        row = f"  {layer:>6} |"
        for coef in coefs:
            subset = df[(df["layer"] == layer) & (df["coef"] == coef)]
            if subset.empty:
                row += f"{'N/A':>{col_w}}"
            else:
                val    = subset["correct_label_logprob"].mean()
                marker = "*" if (layer == best_layer and coef == best_coef) else " "
                row   += f"{val:>{col_w - 1}.3f}{marker}"
        print(row)

    print(f"\n  * = best setting  (layer={best_layer}, coef={best_coef})")

    return best_layer, best_coef, baseline_logprob


# =============================================================================
# Tier 2 — Layer analysis
# =============================================================================

def tier2(df: pd.DataFrame, baseline_logprob: float):
    _header("TIER 2 — LAYER ANALYSIS")

    steered = df[df["coef"] != 0.0]
    rows = []
    for layer in sorted(df["layer"].unique()):
        layer_df = steered[steered["layer"] == layer]
        by_coef  = layer_df.groupby("coef")["correct_label_logprob"].mean()
        best_coef   = by_coef.idxmax()
        best_subset = layer_df[layer_df["coef"] == best_coef]
        rows.append({
            "layer"      : layer,
            "best_coef"  : best_coef,
            "mean_logprob": by_coef[best_coef],
            "delta_logprob": by_coef[best_coef] - baseline_logprob,
            "accuracy"   : best_subset["correct"].mean(),
        })

    table = (
        pd.DataFrame(rows)
        .sort_values("mean_logprob", ascending=False)
        .reset_index(drop=True)
    )

    print()
    print(f"  {'Rank':>5}  {'Layer':>6}  {'Best Coef':>10}  "
          f"{'Mean LogP':>10}  {'Δ Baseline':>11}  {'Accuracy':>9}")
    print(f"  {'-' * 60}")
    for i, r in table.iterrows():
        tag = "  ← TOP 3" if i < 3 else ""
        print(f"  {i+1:>5}  {int(r.layer):>6}  {r.best_coef:>10.2f}  "
              f"{r.mean_logprob:>10.4f}  {r.delta_logprob:>+11.4f}  "
              f"{r.accuracy:>9.3f}{tag}")


# =============================================================================
# Tier 3 — Deep dive at best setting
# =============================================================================

def tier3(df: pd.DataFrame, best_layer: int, best_coef: float, baseline_logprob: float):
    _header(f"TIER 3 — DEEP DIVE  (layer={best_layer}, coef={best_coef})")

    best_rows = df[(df["layer"] == best_layer) & (df["coef"] == best_coef)].copy()
    base_rows = df[(df["layer"] == best_layer) & (df["coef"] == 0.0)].copy()

    merged = (
        best_rows[["question_id", "correct_label_logprob", "correct", "correct_label_rank"]]
        .merge(
            base_rows[["question_id", "correct_label_logprob", "correct", "correct_label_rank"]]
            .rename(columns={
                "correct_label_logprob": "base_logprob",
                "correct"              : "base_correct",
                "correct_label_rank"   : "base_rank",
            }),
            on="question_id",
        )
        .sort_values("question_id")
    )
    merged["delta"] = merged["correct_label_logprob"] - merged["base_logprob"]

    def flip_label(r):
        if not r.base_correct and r.correct:
            return "W→C"
        if r.base_correct and not r.correct:
            return "C→W"
        return "---"

    merged["flip"] = merged.apply(flip_label, axis=1)

    # Per-question table
    _section("Per-Question Breakdown")
    print(f"\n  {'Q':>5}  {'Base LogP':>10}  {'Steer LogP':>11}  "
          f"{'Delta':>8}  {'Base Rank':>10}  {'New Rank':>9}  {'Flip':>5}")
    print(f"  {'-' * 65}")
    for r in merged.itertuples(index=False):
        print(f"  {int(r.question_id):>5}  {r.base_logprob:>10.4f}  "
              f"{r.correct_label_logprob:>11.4f}  {r.delta:>+8.4f}  "
              f"{int(r.base_rank):>10}  {int(r.correct_label_rank):>9}  "
              f"{r.flip:>5}")

    # Rank change summary
    _section("Rank Change Summary")
    print(f"\n  Baseline rank distribution:")
    for rank, cnt in base_rows["correct_label_rank"].value_counts().sort_index().items():
        print(f"    Rank {int(rank)} : {cnt} questions")

    print(f"\n  Rank changes at best setting  (positive = improved):")
    for delta, cnt in best_rows["rank_change"].value_counts().sort_index().items():
        direction = "improved" if delta > 0 else ("degraded" if delta < 0 else "unchanged")
        print(f"    Δrank {int(delta):>+3} : {cnt:>3} questions  ({direction})")

    # Logprob percentiles
    _section("Logprob Percentiles  (baseline vs best setting)")
    pcts = [10, 25, 50, 75, 90]
    base_lp = base_rows["correct_label_logprob"]
    best_lp = best_rows["correct_label_logprob"]

    print(f"\n  {'':>12}  {'Baseline':>10}  {'Steered':>10}  {'Delta':>8}")
    print(f"  {'-' * 45}")
    for p in pcts:
        bv = np.percentile(base_lp, p)
        sv = np.percentile(best_lp, p)
        print(f"  p{p:<11}  {bv:>10.4f}  {sv:>10.4f}  {sv - bv:>+8.4f}")
    print(f"  {'mean':<12}  {base_lp.mean():>10.4f}  {best_lp.mean():>10.4f}  "
          f"{best_lp.mean() - base_lp.mean():>+8.4f}")
    print(f"  {'std':<12}  {base_lp.std():>10.4f}  {best_lp.std():>10.4f}  "
          f"{best_lp.std() - base_lp.std():>+8.4f}")


# =============================================================================
# Tier 4 — True k-fold CV
# =============================================================================

def tier4(df: pd.DataFrame, k: int = 5, seed: int = 42):
    _header(f"TIER 4 — {k}-FOLD CROSS-VALIDATION")

    question_ids = np.array(sorted(df["question_id"].unique()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(question_ids)
    folds = np.array_split(shuffled, k)

    steered = df[df["coef"] != 0.0]

    fold_results = []

    for fold_idx, test_ids in enumerate(folds):
        train_ids = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        train_df = steered[steered["question_id"].isin(train_ids)]
        test_df  = df[df["question_id"].isin(test_ids)]

        # Select best (layer, coef) on train by mean correct-label logprob
        train_lp = train_df.groupby(["layer", "coef"])["correct_label_logprob"].mean()
        top3     = train_lp.nlargest(3)

        sel_layer, sel_coef = top3.index[0]
        train_logprob = float(top3.iloc[0])

        # Evaluate selected on test
        test_sel  = test_df[(test_df["layer"] == sel_layer) & (test_df["coef"] == sel_coef)]
        test_base = test_df[(test_df["layer"] == sel_layer) & (test_df["coef"] == 0.0)]

        test_logprob  = test_sel["correct_label_logprob"].mean()
        test_acc      = test_sel["correct"].mean()
        base_logprob  = test_base["correct_label_logprob"].mean()
        base_acc      = test_base["correct"].mean()

        # Sensitivity: top 3 train choices evaluated on test
        sensitivity = []
        for rank_i, ((layer_i, coef_i), _) in enumerate(top3.items()):
            alt = test_df[(test_df["layer"] == layer_i) & (test_df["coef"] == coef_i)]
            sensitivity.append({
                "rank"        : rank_i + 1,
                "layer"       : layer_i,
                "coef"        : coef_i,
                "test_logprob": float(alt["correct_label_logprob"].mean()),
                "test_acc"    : float(alt["correct"].mean()),
            })

        fold_results.append({
            "fold"         : fold_idx + 1,
            "n_train"      : len(train_ids),
            "n_test"       : len(test_ids),
            "sel_layer"    : sel_layer,
            "sel_coef"     : sel_coef,
            "train_logprob": train_logprob,
            "test_logprob" : test_logprob,
            "test_acc"     : test_acc,
            "base_logprob" : base_logprob,
            "base_acc"     : base_acc,
            "delta_logprob": test_logprob - base_logprob,
            "sensitivity"  : sensitivity,
        })

    # Per-fold table
    _section("Per-Fold Results")
    col = f"  {'Fold':>5}  {'N_train':>8}  {'N_test':>7}  {'Sel Layer':>10}  " \
          f"{'Sel Coef':>9}  {'Train LogP':>11}  {'Test LogP':>10}  " \
          f"{'Base LogP':>10}  {'Δ LogP':>8}  {'Test Acc':>9}  {'Base Acc':>9}"
    print(f"\n{col}")
    print(f"  {'-' * (len(col) - 2)}")
    for r in fold_results:
        print(f"  {r['fold']:>5}  {r['n_train']:>8}  {r['n_test']:>7}  "
              f"{r['sel_layer']:>10}  {r['sel_coef']:>9.2f}  "
              f"{r['train_logprob']:>11.4f}  {r['test_logprob']:>10.4f}  "
              f"{r['base_logprob']:>10.4f}  {r['delta_logprob']:>+8.4f}  "
              f"{r['test_acc']:>9.3f}  {r['base_acc']:>9.3f}")

    # Summary stats
    _section("Summary Across Folds")
    test_lps    = [r["test_logprob"]  for r in fold_results]
    test_accs   = [r["test_acc"]      for r in fold_results]
    delta_lps   = [r["delta_logprob"] for r in fold_results]
    base_lps    = [r["base_logprob"]  for r in fold_results]
    base_accs   = [r["base_acc"]      for r in fold_results]

    print(f"\n  {'Metric':<30}  {'Mean':>10}  {'Std':>8}")
    print(f"  {'-' * 52}")
    print(f"  {'Baseline logprob':<30}  {np.mean(base_lps):>10.4f}  {np.std(base_lps):>8.4f}")
    print(f"  {'Test logprob (steered)':<30}  {np.mean(test_lps):>10.4f}  {np.std(test_lps):>8.4f}")
    print(f"  {'Delta logprob':<30}  {np.mean(delta_lps):>+10.4f}  {np.std(delta_lps):>8.4f}")
    print(f"  {'Baseline accuracy':<30}  {np.mean(base_accs):>10.3f}  {np.std(base_accs):>8.3f}")
    print(f"  {'Test accuracy (steered)':<30}  {np.mean(test_accs):>10.3f}  {np.std(test_accs):>8.3f}")

    # Selection stability
    _section("Selection Stability")
    print(f"\n  {'Fold':>5}  {'Layer':>7}  {'Coef':>7}")
    print(f"  {'-' * 25}")
    selections = []
    for r in fold_results:
        print(f"  {r['fold']:>5}  {r['sel_layer']:>7}  {r['sel_coef']:>7.2f}")
        selections.append((r["sel_layer"], r["sel_coef"]))

    counts = pd.Series(selections).value_counts()
    print(f"\n  Most selected  : layer={counts.index[0][0]}, coef={counts.index[0][1]}"
          f"  ({counts.iloc[0]}/{k} folds)")
    if len(counts) > 1:
        print(f"  Selection varied across {len(counts)} different (layer, coef) pairs:")
        for (layer, coef), cnt in counts.items():
            print(f"    layer={layer}, coef={coef:>6.2f}  →  {cnt} fold(s)")
    else:
        print(f"  Stable: same (layer, coef) selected in all {k} folds")

    # Sensitivity
    _section("Sensitivity — Top 3 Train Choices vs Test LogProb per Fold")
    print(f"\n  {'Fold':>5}  {'Train Rank':>11}  {'Layer':>7}  {'Coef':>7}  "
          f"{'Test LogP':>11}  {'Test Acc':>9}")
    print(f"  {'-' * 60}")
    for r in fold_results:
        for s in r["sensitivity"]:
            marker = " ←" if s["rank"] == 1 else "  "
            print(f"  {r['fold']:>5}  {s['rank']:>11}  {s['layer']:>7}  "
                  f"{s['coef']:>7.2f}  {s['test_logprob']:>11.4f}  "
                  f"{s['test_acc']:>9.3f}{marker}")
        print()


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc MCF steering analysis: logprob stats + k-fold CV"
    )
    parser.add_argument("mcf_dir", help="Path to results/<exp_id>/mcf/")
    parser.add_argument("--k",    type=int, default=5,  help="CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    mcf_dir = Path(args.mcf_dir)
    if not mcf_dir.exists():
        print(f"Error: {mcf_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    df = load_data(mcf_dir)
    print(f"\nLoaded  {len(df):,} rows  |  "
          f"{df['question_id'].nunique()} questions  |  "
          f"{df['layer'].nunique()} layers  |  "
          f"{df['coef'].nunique()} coefs")

    best_layer, best_coef, baseline_logprob = tier1(df)
    tier2(df, baseline_logprob)
    tier3(df, best_layer, best_coef, baseline_logprob)
    tier4(df, k=args.k, seed=args.seed)

    print()
    _sep()
    print("  Done.")
    _sep()
    print()


if __name__ == "__main__":
    main()
