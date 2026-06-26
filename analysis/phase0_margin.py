"""
Phase 0: Margin analysis — does negative steering squeeze wrong labels harder?

For each (layer, coef, question): compute
    margin = correct_label_logprob - max_wrong_logprob

Then delta_margin = margin(coef) - margin(baseline=0).

Key question: positive coef raises correct logprob (delta_correct_logprob > 0)
but also raises wrong logprobs even more → margin shrinks → accuracy drops.
Negative coef should show the reverse: delta_margin > 0 even if
delta_correct_logprob < 0.

Also prints the text of layer-13 flips (wrong→right at best negative coef).

Usage:
    python analysis/phase0_margin.py results/exp_20260602_gpqa_physics_llama8b
    python analysis/phase0_margin.py results/<exp> --layer 13 --save analysis_out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


LABELS = ["A", "B", "C", "D"]


def load_mcf(exp_dir: Path) -> pd.DataFrame:
    paths = sorted((exp_dir / "mcf").glob("layer_*_results.csv"))
    if not paths:
        sys.exit(f"[!] No layer_*_results.csv under {exp_dir}/mcf/")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def add_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Add max_wrong_logprob and margin columns."""
    label_to_col = {"A": "logprob_A", "B": "logprob_B", "C": "logprob_C", "D": "logprob_D"}

    def _max_wrong(row):
        wrong_cols = [label_to_col[l] for l in LABELS if l != row["correct_label"]]
        return max(row[c] for c in wrong_cols)

    df = df.copy()
    df["max_wrong_logprob"] = df.apply(_max_wrong, axis=1)
    df["margin"] = df["correct_label_logprob"] - df["max_wrong_logprob"]
    return df


def add_delta_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta_margin = margin(coef) - margin(coef=0) per (layer, question_id)."""
    base = (
        df[df["coef"] == 0.0]
        .set_index(["layer", "question_id"])["margin"]
        .rename("margin_base")
    )
    df = df.join(base, on=["layer", "question_id"])
    df["delta_margin"] = df["margin"] - df["margin_base"]
    return df


def aggregate_by_sign(df: pd.DataFrame) -> pd.DataFrame:
    nz = df[df["coef"] != 0.0].copy()
    nz["sign"] = nz["coef"].apply(lambda c: "pos" if c > 0 else "neg")

    rows = []
    for sign, grp in nz.groupby("sign"):
        rows.append({
            "sign":                sign,
            "n_combos":            len(grp),
            "mean_delta_correct":  round(grp["delta_correct_logprob"].mean(), 4),
            "mean_delta_margin":   round(grp["delta_margin"].mean(), 4),
            "pct_margin_improved": round((grp["delta_margin"] > 0).mean(), 4),
            "pct_correct_improved":round((grp["delta_correct_logprob"] > 0).mean(), 4),
        })
    return pd.DataFrame(rows)


def per_layer_sign(df: pd.DataFrame) -> pd.DataFrame:
    nz = df[df["coef"] != 0.0].copy()
    nz["sign"] = nz["coef"].apply(lambda c: "pos" if c > 0 else "neg")

    rows = []
    for (layer, sign), grp in nz.groupby(["layer", "sign"]):
        rows.append({
            "layer":               int(layer),
            "sign":                sign,
            "mean_delta_correct":  round(grp["delta_correct_logprob"].mean(), 4),
            "mean_delta_margin":   round(grp["delta_margin"].mean(), 4),
            "pct_margin_improved": round((grp["delta_margin"] > 0).mean(), 4),
        })
    return pd.DataFrame(rows).sort_values(["layer", "sign"]).reset_index(drop=True)


def flip_details(df: pd.DataFrame, layer: int) -> None:
    """Print text of questions that flip wrong→right at the best negative coef on layer."""
    ld = df[df["layer"] == layer]
    nz_neg = ld[(ld["coef"] < 0)]
    if nz_neg.empty:
        print("  No negative coefs found.")
        return

    # Best negative coef by accuracy
    acc = nz_neg.groupby("coef")["correct"].mean()
    best_coef = float(acc.idxmax())
    print(f"  Best negative coef at layer {layer}: {best_coef}  (acc={acc[best_coef]:.3f})")

    base    = ld[ld["coef"] == 0.0].set_index("question_id")
    steered = ld[ld["coef"] == best_coef].set_index("question_id")
    common  = base.index.intersection(steered.index)

    flips_w2r = [qid for qid in common if not base.loc[qid, "correct"] and steered.loc[qid, "correct"]]
    flips_r2w = [qid for qid in common if base.loc[qid, "correct"] and not steered.loc[qid, "correct"]]

    print(f"  wrong→right: {len(flips_w2r)}   right→wrong: {len(flips_r2w)}\n")

    if flips_w2r:
        print("  ── wrong→right questions ──")
        for qid in flips_w2r:
            row = steered.loc[qid]
            prompt_snippet = str(row.get("prompt", ""))[:300].replace("\n", " ")
            print(f"\n  qid={qid}  correct_label={row['correct_label']}")
            print(f"  {prompt_snippet}")

    if flips_r2w:
        print("\n  ── right→wrong questions ──")
        for qid in flips_r2w:
            row = steered.loc[qid]
            prompt_snippet = str(row.get("prompt", ""))[:300].replace("\n", " ")
            print(f"\n  qid={qid}  correct_label={row['correct_label']}")
            print(f"  {prompt_snippet}")


def _hr(label: str = "", width: int = 78) -> None:
    pad = width - len(label) - 2
    print(f"── {label} " + "─" * max(pad, 0))


def _print_df(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 200, "display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dir", type=Path)
    p.add_argument("--layer", type=int, default=13, help="Layer to inspect for flips (default: 13)")
    p.add_argument("--save", type=Path, default=None)
    args = p.parse_args()

    if not args.exp_dir.exists():
        sys.exit(f"[!] {args.exp_dir} does not exist")

    print(f"\nLoading MCF results from {args.exp_dir.name}/mcf/ ...")
    df = load_mcf(args.exp_dir)

    required = {"logprob_A", "logprob_B", "logprob_C", "logprob_D", "correct_label_logprob",
                "delta_correct_logprob", "correct_label"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[!] Missing columns: {missing}\n    Available: {list(df.columns)}")

    df = add_margin(df)
    df = add_delta_margin(df)

    print(f"  {len(df):,} rows  |  layers: {df['layer'].nunique()}  |  coefs: {df['coef'].nunique()}  |  questions: {df['question_id'].nunique()}\n")

    _hr("1. Aggregate: delta_correct_logprob vs delta_margin by coef sign")
    agg = aggregate_by_sign(df)
    _print_df(agg)
    print(
        "\n  KEY: if pos has mean_delta_correct > 0 but mean_delta_margin < 0"
        "\n       → positive coef raises correct logprob but wrong logprobs rise MORE"
        "\n       → margin shrinks → accuracy drops despite higher correct logprob"
    )

    print()
    _hr("2. Per-layer breakdown (mean delta_margin by sign)")
    layer_tbl = per_layer_sign(df)
    _print_df(layer_tbl)

    print()
    _hr(f"3. Flip details at layer {args.layer}")
    flip_details(df, layer=args.layer)

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.save / "margin_aggregate.csv", index=False)
        layer_tbl.to_csv(args.save / "margin_per_layer.csv", index=False)
        df[df["coef"] != 0.0][["layer", "question_id", "coef",
                                "delta_correct_logprob", "delta_margin", "correct"]]\
            .to_csv(args.save / "margin_all_rows.csv", index=False)
        print(f"\nSaved to {args.save}/")


if __name__ == "__main__":
    main()
