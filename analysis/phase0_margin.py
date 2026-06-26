"""
Phase 0: Margin analysis — does negative steering squeeze wrong labels harder?

For each (layer, coef, question): compute
    margin = correct_label_logprob - max_wrong_logprob

Then delta_margin = margin(coef) - margin(baseline=0).

The key section is Section 1: for each layer, find the best val coef, then
report its margin story on the test set. Averaging over all coefs is
misleading — only the val-selected coef matters.

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


def _attach_split(df: pd.DataFrame, exp_dir: Path) -> pd.DataFrame:
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


def val_selected_margin(df: pd.DataFrame) -> pd.DataFrame:
    """For each layer: find best val coef (by accuracy), report its margin story on test.

    This is the meaningful analysis — averaging over all coefs is noise.
    Only the coef the val set would pick actually gets deployed to test.
    """
    has_splits = df["split"].nunique() > 1
    val = df[df["split"] == "validation"] if has_splits else df
    test = df[df["split"] == "test"] if has_splits else df

    val_per = val.groupby(["layer", "coef"])["correct"].mean()
    rows = []
    for layer in sorted(df["layer"].unique()):
        # best non-zero coef on val
        layer_val = val_per[val_per.index.get_level_values("layer") == layer]
        nz = layer_val[layer_val.index.get_level_values("coef") != 0.0]
        if nz.empty:
            continue
        best_coef = float(nz.idxmax()[1])
        sign = "neg" if best_coef < 0 else "pos"

        # val stats at best coef
        v = val[(val["layer"] == layer) & (val["coef"] == best_coef)]
        val_acc   = round(v["correct"].mean(), 4)
        val_dc    = round(v["delta_correct_logprob"].mean(), 4)
        val_dm    = round(v["delta_margin"].mean(), 4)

        # test stats at same coef
        t = test[(test["layer"] == layer) & (test["coef"] == best_coef)]
        if t.empty:
            test_acc = test_dc = test_dm = float("nan")
        else:
            test_acc  = round(t["correct"].mean(), 4)
            test_dc   = round(t["delta_correct_logprob"].mean(), 4)
            test_dm   = round(t["delta_margin"].mean(), 4)

        rows.append({
            "layer":          int(layer),
            "best_val_coef":  best_coef,
            "sign":           sign,
            "val_acc":        val_acc,
            "val_delta_correct": val_dc,
            "val_delta_margin":  val_dm,
            "test_acc":       test_acc,
            "test_delta_correct": test_dc,
            "test_delta_margin":  test_dm,
        })
    return pd.DataFrame(rows)


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

    df = _attach_split(df, args.exp_dir)
    df = add_margin(df)
    df = add_delta_margin(df)

    splits = df.groupby("split")["question_id"].nunique().to_dict()
    print(f"  {len(df):,} rows  |  layers: {df['layer'].nunique()}  |  coefs: {df['coef'].nunique()}  |  splits: {splits}\n")

    _hr("1. Val-selected coef → margin story on test (the meaningful analysis)")
    focused = val_selected_margin(df)
    _print_df(focused)
    print(
        "\n  Read: val_delta_margin > 0 → the val-best coef improves the margin on val"
        "\n        test_delta_margin > 0 → it also improves margin on test (confirms mechanism)"
        "\n        test_delta_margin < 0 → accuracy gain is not margin-based (noise / threshold luck)"
    )

    print()
    _hr(f"2. Flip details at layer {args.layer}")
    flip_details(df, layer=args.layer)

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        focused.to_csv(args.save / "val_selected_margin.csv", index=False)
        print(f"\nSaved to {args.save}/")


if __name__ == "__main__":
    main()
