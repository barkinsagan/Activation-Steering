"""
Post-hoc diagnostic: why are negative steering coefficients winning?

Reads results/<experiment_id>/{mcf,cf}/ and the split_manifest.csv, then reports:

  1. Per-layer val→test transfer:
       layer, best_val_coef, sign, val_gain, test_gain
  2. Aggregate: positive vs negative coefs on val and test
       (does positive help on average even though the per-layer oracle picks negative?)
  3. Val→test sign agreement (do val-best and test-best agree?)
  4. Per-question flips at the winning negative coefs (wrong→right / right→wrong)

Usage:
    python analysis/diagnose_negative_coefs.py results/exp_20260602_gpqa_physics_llama8b
    python analysis/diagnose_negative_coefs.py results/<exp> --formulation cf
    python analysis/diagnose_negative_coefs.py results/<exp> --save analysis_out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# Loading
# =============================================================================

def _attach_split(df: pd.DataFrame, exp_dir: Path) -> pd.DataFrame:
    """Add a 'split' column from split_manifest.csv (validation/test/unknown)."""
    manifest = exp_dir / "split_manifest.csv"
    if not manifest.exists():
        df["split"] = "all"
        return df
    m = pd.read_csv(manifest)
    id_col = next((c for c in ("eval_question_id", "question_id") if c in m.columns), None)
    if id_col is None or "split" not in m.columns:
        df["split"] = "all"
        return df
    split_map = dict(zip(m[id_col], m["split"]))
    df["split"] = df["question_id"].map(split_map).fillna("unknown")
    return df


def load_mcf(exp_dir: Path) -> pd.DataFrame:
    mcf_dir = exp_dir / "mcf"
    paths = sorted(mcf_dir.glob("layer_*_results.csv"))
    if not paths:
        raise FileNotFoundError(f"No layer_*_results.csv under {mcf_dir}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[["layer", "question_id", "coef", "correct"]].copy()
    df["correct"] = df["correct"].astype(bool)
    return _attach_split(df, exp_dir)


def load_cf(exp_dir: Path, norm: str = "char") -> pd.DataFrame:
    cf_dir = exp_dir / "cf"
    col_map = {"sum": "correct_sum", "mean": "correct_mean", "char": "correct_char", "pmi": "correct_pmi"}
    correct_col = col_map[norm]

    frames = []
    for layer_dir in sorted(cf_dir.glob("layer_*"), key=lambda d: int(d.name.split("_")[1])):
        wide = layer_dir / "detailed_wide.csv"
        if not wide.exists():
            continue
        frame = pd.read_csv(wide)
        frame["layer"] = int(layer_dir.name.split("_")[1])
        if correct_col not in frame.columns:
            raise ValueError(f"{wide} missing column {correct_col}")
        frame = frame[["layer", "question_id", "coef", correct_col]].rename(columns={correct_col: "correct"})
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No detailed_wide.csv under {cf_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["correct"].astype(bool)
    return _attach_split(df, exp_dir)


# =============================================================================
# Tables
# =============================================================================

def per_layer_transfer(df: pd.DataFrame) -> pd.DataFrame:
    """For each layer: best val coef and resulting val/test accuracies + gains."""
    val = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]
    if val.empty or test.empty:
        val = test = df  # single-split fallback

    val_per = val.groupby(["layer", "coef"])["correct"].mean().reset_index()
    test_per = test.groupby(["layer", "coef"])["correct"].mean().reset_index()

    def _acc(per: pd.DataFrame, layer: int, coef: float) -> float:
        row = per[(per["layer"] == layer) & (per["coef"] == coef)]
        return float(row["correct"].iloc[0]) if len(row) else float("nan")

    rows = []
    for layer, grp in val_per.groupby("layer"):
        non_base = grp[grp["coef"] != 0.0]
        if non_base.empty:
            continue
        best_row = non_base.loc[non_base["correct"].idxmax()]
        coef = float(best_row["coef"])

        val_base = _acc(val_per, layer, 0.0)
        val_best = float(best_row["correct"])
        test_base = _acc(test_per, layer, 0.0)
        test_best = _acc(test_per, layer, coef)

        rows.append({
            "layer":            int(layer),
            "best_val_coef":    coef,
            "sign":             "neg" if coef < 0 else ("pos" if coef > 0 else "zero"),
            "val_base":         round(val_base, 4),
            "val_best":         round(val_best, 4),
            "val_gain":         round(val_best - val_base, 4),
            "test_base":        round(test_base, 4),
            "test_best":        round(test_best, 4),
            "test_gain":        round(test_best - test_base, 4) if test_best == test_best else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def aggregate_sign(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy gain over baseline, partitioned into positive vs negative coefs.

    Tests the user's observation: "positive helps on average, but the per-layer
    oracle picks negative." If true, expect:
      - positive coefs have higher MEAN gain
      - negative coefs have a few large peaks that win the oracle selection
    """
    rows = []
    for split in ("validation", "test"):
        sub = df[df["split"] == split] if (df["split"] == split).any() else df
        per = sub.groupby(["layer", "coef"])["correct"].mean().reset_index()

        baselines = per[per["coef"] == 0.0].set_index("layer")["correct"].to_dict()
        per["gain"] = per.apply(
            lambda r: r["correct"] - baselines.get(r["layer"], float("nan")), axis=1,
        )
        nz = per[per["coef"] != 0.0]

        pos = nz[nz["coef"] > 0]
        neg = nz[nz["coef"] < 0]
        rows.append({
            "split": split,
            "n_pos_combos":     len(pos),
            "pos_mean_gain":    round(pos["gain"].mean(), 4) if len(pos) else float("nan"),
            "pos_max_gain":     round(pos["gain"].max(), 4) if len(pos) else float("nan"),
            "pos_pct_improved": round((pos["gain"] > 0).mean(), 4) if len(pos) else float("nan"),
            "n_neg_combos":     len(neg),
            "neg_mean_gain":    round(neg["gain"].mean(), 4) if len(neg) else float("nan"),
            "neg_max_gain":     round(neg["gain"].max(), 4) if len(neg) else float("nan"),
            "neg_pct_improved": round((neg["gain"] > 0).mean(), 4) if len(neg) else float("nan"),
        })
    return pd.DataFrame(rows)


def sign_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """For each layer, best non-zero coef on val vs on test. Disagreement = overfit/noise."""
    val = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]
    if val.empty or test.empty:
        return pd.DataFrame()

    val_per = val.groupby(["layer", "coef"])["correct"].mean().reset_index()
    test_per = test.groupby(["layer", "coef"])["correct"].mean().reset_index()

    rows = []
    for layer in sorted(val_per["layer"].unique()):
        v = val_per[(val_per["layer"] == layer) & (val_per["coef"] != 0.0)]
        t = test_per[(test_per["layer"] == layer) & (test_per["coef"] != 0.0)]
        if v.empty or t.empty:
            continue
        v_best = float(v.loc[v["correct"].idxmax()]["coef"])
        t_best = float(t.loc[t["correct"].idxmax()]["coef"])
        rows.append({
            "layer":           int(layer),
            "val_best_coef":   v_best,
            "test_best_coef":  t_best,
            "same_sign":       (v_best * t_best) > 0,
            "same_coef":       v_best == t_best,
        })
    return pd.DataFrame(rows)


def flipped_questions(df: pd.DataFrame, transfer: pd.DataFrame, *, only_negative: bool = True) -> pd.DataFrame:
    """For each layer where best_val_coef < 0 (or all layers): list test questions
    that went baseline-wrong → steered-right and vice versa.
    """
    target = transfer[transfer["best_val_coef"] < 0] if only_negative else transfer
    if target.empty:
        return pd.DataFrame()

    test = df[df["split"] == "test"]
    if test.empty:
        test = df  # fallback

    rows = []
    for _, lr in target.iterrows():
        layer = int(lr["layer"])
        coef = float(lr["best_val_coef"])
        ld = test[test["layer"] == layer]
        base = ld[ld["coef"] == 0.0].set_index("question_id")["correct"]
        steered = ld[ld["coef"] == coef].set_index("question_id")["correct"]
        common = base.index.intersection(steered.index)

        n_w2r = n_r2w = 0
        for qid in common:
            b, s = bool(base.loc[qid]), bool(steered.loc[qid])
            if not b and s:
                n_w2r += 1
            elif b and not s:
                n_r2w += 1
        rows.append({
            "layer":         layer,
            "best_val_coef": coef,
            "n_questions":   int(len(common)),
            "wrong_to_right": n_w2r,
            "right_to_wrong": n_r2w,
            "net_flips":      n_w2r - n_r2w,
        })
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


# =============================================================================
# Printing
# =============================================================================

def _hr(label: str = "", width: int = 78) -> None:
    if not label:
        print("─" * width)
        return
    pad = width - len(label) - 2
    print(f"── {label} " + "─" * max(pad, 0))


def _print_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print("  (empty)")
        return
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dir", type=Path, help="Path to results/<experiment_id>/")
    p.add_argument("--formulation", choices=("mcf", "cf"), default="mcf")
    p.add_argument("--cf-norm", choices=("sum", "mean", "char", "pmi"), default="char",
                   help="Which CF correctness column to use (default: char). Ignored for MCF.")
    p.add_argument("--save", type=Path, default=None,
                   help="If set, also save tables to this directory as CSVs.")
    args = p.parse_args(argv)

    if not args.exp_dir.exists():
        print(f"[!] {args.exp_dir} does not exist", file=sys.stderr)
        return 1

    if args.formulation == "mcf":
        df = load_mcf(args.exp_dir)
    else:
        df = load_cf(args.exp_dir, norm=args.cf_norm)

    n_layers = df["layer"].nunique()
    n_coefs  = df["coef"].nunique()
    splits   = df.groupby("split")["question_id"].nunique().to_dict()
    print(f"\nLoaded {len(df):,} rows from {args.exp_dir.name}/{args.formulation}/")
    print(f"  layers : {n_layers}    coefs : {n_coefs}    splits : {splits}\n")

    _hr("1. Aggregate: positive vs negative coefs")
    agg = aggregate_sign(df)
    _print_df(agg)
    print(
        "\n  Read: if pos_mean_gain > neg_mean_gain but neg_max_gain >> pos_max_gain,"
        "\n  the average favors positive while the oracle selection rides a few negative peaks."
    )

    print()
    _hr("2. Per-layer val→test transfer (best val coef applied to test)")
    transfer = per_layer_transfer(df)
    _print_df(transfer)
    if not transfer.empty:
        n_neg = int((transfer["best_val_coef"] < 0).sum())
        n_pos = int((transfer["best_val_coef"] > 0).sum())
        mean_neg_test = transfer.loc[transfer["best_val_coef"] < 0, "test_gain"].mean()
        mean_pos_test = transfer.loc[transfer["best_val_coef"] > 0, "test_gain"].mean()
        print(
            f"\n  Layers picking negative on val: {n_neg}   positive: {n_pos}"
            f"\n  Mean test gain among negative-picking layers : {mean_neg_test:.4f}"
            f"\n  Mean test gain among positive-picking layers : {mean_pos_test:.4f}"
        )

    print()
    _hr("3. Val/test sign agreement (per layer)")
    agree = sign_agreement(df)
    _print_df(agree)
    if not agree.empty:
        same_sign = int(agree["same_sign"].sum())
        total = len(agree)
        same_exact = int(agree["same_coef"].sum())
        print(
            f"\n  Same sign : {same_sign}/{total}    same exact coef : {same_exact}/{total}"
            "\n  Low same-sign count → multiple-comparison noise rather than real signal."
        )

    print()
    _hr("4. Flipped test questions at the winning negative coefs")
    flips = flipped_questions(df, transfer, only_negative=True)
    _print_df(flips)
    if not flips.empty:
        net = int(flips["net_flips"].sum())
        print(
            f"\n  Total net flips across negative-winning layers : {net:+d}"
            "\n  (positive = steering rescued more than it broke)"
        )

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.save / "aggregate_sign.csv", index=False)
        transfer.to_csv(args.save / "per_layer_transfer.csv", index=False)
        agree.to_csv(args.save / "sign_agreement.csv", index=False)
        flips.to_csv(args.save / "flips_negative_layers.csv", index=False)
        print(f"\nSaved CSVs to {args.save}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
