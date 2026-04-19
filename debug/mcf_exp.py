"""
debug/mcf_exp.py — reusable MCF steering experiment harness (print-only).

A thin CLI that runs an MCF sweep over a chosen (prompts × token_position ×
normalize) variant and prints a per-layer asymmetry table. Nothing is written
to disk — this is pure diagnostic output.

Example — Experiment 1 (current prompts, last, normalize):
    python debug/mcf_exp.py \\
        --exp-id e1_normalize \\
        --eval-path data/eval/anatomy_sweep.csv \\
        --pos-file data/prompts/pos.txt \\
        --neg-file data/prompts/neg.txt \\
        --layers 12 18 \\
        --coefs -2 -1.5 -1 -0.5 -0.25 0.25 0.5 1 1.5 2 \\
        --token-position last \\
        --normalize \\
        --num-questions 200

To run Experiment 2, only the --pos-file / --neg-file / --exp-id change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dim import DifferenceInMeansSteering
from hook import ModelWithHooks
from olmes.formatter import build_formatter
from single_token_completion_test import MCFScorer


def load_prompts(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No prompts found in {path}")
    return lines


def run_layer(scorer, formatter, eval_df, dim_steerer, vector,
              layer_idx: int, coefs: list[float], verbose_every: int) -> list[dict]:
    """Score baseline + each coef for one layer; return in-memory records."""
    coefs = [0.0] + [c for c in coefs if c != 0.0]
    K = len(coefs)
    coefs_tensor = torch.tensor(coefs, dtype=torch.float32)
    dim_steerer.apply_steering(vector, coefficient=coefs_tensor)

    records: list[dict] = []
    baselines: dict[int, float] = {}

    try:
        for i, row in eval_df.iterrows():
            mcf_row = formatter.format_mcf(row, question_idx=i)
            results = scorer.score_mcf_batched(mcf_row.prompt, mcf_row.correct_label, K)
            for coef, result in zip(coefs, results):
                lp = result["correct_label_logprob"]
                if coef == 0.0:
                    baselines[i] = lp
                    delta = 0.0
                else:
                    delta = lp - baselines[i]
                records.append({
                    "layer": layer_idx,
                    "question_id": i,
                    "coef": coef,
                    "correct": result["correct"],
                    "correct_label_logprob": lp,
                    "delta_correct_logprob": delta,
                })
            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"    {i + 1}/{len(eval_df)}")
    finally:
        dim_steerer.reset_steering()

    return records


def print_summary(df: pd.DataFrame) -> None:
    """Per-(layer, coef) aggregates + asymmetry table."""
    agg_rows = []
    for (layer, coef), grp in df.groupby(["layer", "coef"]):
        row = {
            "layer": int(layer),
            "coef": float(coef),
            "n": len(grp),
            "accuracy": grp["correct"].mean(),
            "mean_correct_logprob": grp["correct_label_logprob"].mean(),
        }
        if coef != 0.0:
            row["mean_delta"] = grp["delta_correct_logprob"].mean()
            row["pct_improved"] = (grp["delta_correct_logprob"] > 0).mean()
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values(["layer", "coef"]).reset_index(drop=True)

    print("\n" + "=" * 78)
    print("  Per (layer, coef) summary")
    print("=" * 78)
    print(f"{'layer':>5} {'coef':>7} {'n':>5} {'acc':>7} {'mean_lp':>10} "
          f"{'mean_Δ':>10} {'pct+':>7}")
    print("-" * 78)
    for _, r in agg.iterrows():
        delta = f"{r['mean_delta']:>10.4f}" if pd.notna(r.get("mean_delta")) else f"{'-':>10s}"
        pct = f"{r['pct_improved']:>7.3f}" if pd.notna(r.get("pct_improved")) else f"{'-':>7s}"
        print(f"{int(r['layer']):>5d} {r['coef']:>+7.3f} {int(r['n']):>5d} "
              f"{r['accuracy']:>7.3f} {r['mean_correct_logprob']:>10.4f} "
              f"{delta} {pct}")

    nz = df[df["coef"] != 0.0].copy()
    if nz.empty:
        return
    nz["abs_coef"] = nz["coef"].abs()
    nz["sign"] = nz["coef"].apply(lambda c: "pos" if c > 0 else "neg")
    per_lc = nz.groupby(["layer", "abs_coef", "sign"]).agg(
        mean_delta=("delta_correct_logprob", "mean"),
        pct_improved=("delta_correct_logprob", lambda s: (s > 0).mean()),
    ).reset_index()
    wide = per_lc.pivot_table(
        index=["layer", "abs_coef"], columns="sign",
        values=["mean_delta", "pct_improved"],
    ).reset_index()

    print("\n" + "=" * 78)
    print("  Asymmetry table  (|Δ+ + Δ-| near 0 → symmetric-hurt)")
    print("=" * 78)
    print(f"{'layer':>5} {'|c|':>6} "
          f"{'Δ(+c)':>10} {'Δ(-c)':>10} {'Δ+ + Δ-':>10} "
          f"{'pct+(+c)':>10} {'pct+(-c)':>10}")
    print("-" * 78)
    for _, r in wide.iterrows():
        layer = int(r[("layer", "")])
        ac = float(r[("abs_coef", "")])
        d_pos = r[("mean_delta", "pos")]
        d_neg = r[("mean_delta", "neg")]
        p_pos = r[("pct_improved", "pos")]
        p_neg = r[("pct_improved", "neg")]
        print(f"{layer:>5d} {ac:>6.2f} "
              f"{d_pos:>10.4f} {d_neg:>10.4f} {(d_pos + d_neg):>10.4f} "
              f"{p_pos:>10.3f} {p_neg:>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True, help="Label shown in header (no files written).")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--eval-path", type=Path, required=True)
    ap.add_argument("--pos-file", type=Path, required=True)
    ap.add_argument("--neg-file", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--coefs", type=float, nargs="+", required=True,
                    help="Non-zero coefs; baseline 0.0 is added automatically")
    ap.add_argument("--token-position", default="last", choices=["last", "mean"])
    ap.add_argument("--normalize", dest="normalize", action="store_true")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=True)
    ap.add_argument("--norm-type", default="unit", choices=["unit", "std"])
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=0)
    ap.add_argument("--fewshot-source", default="")
    ap.add_argument("--task-prefix", default="question")
    ap.add_argument("--shuffle-choices", action="store_true", default=True)
    ap.add_argument("--layer-pattern", default="model.layers.{layer_idx}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    pos_prompts = load_prompts(args.pos_file)
    neg_prompts = load_prompts(args.neg_file)
    print(f"Loaded {len(pos_prompts)} pos / {len(neg_prompts)} neg prompts")

    eval_df = pd.read_csv(args.eval_path)
    if args.num_questions and args.num_questions < len(eval_df):
        eval_df = eval_df.head(args.num_questions).reset_index(drop=True)
    print(f"Eval rows: {len(eval_df)}  (from {args.eval_path})")

    print(f"\nLoading {args.model}  dtype={args.dtype}  device={args.device}")
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
    )
    model.eval()
    model_with_hooks = ModelWithHooks(model)
    scorer = MCFScorer(forward_fn=model_with_hooks, tokenizer=tokenizer)

    formatter = build_formatter(
        task_prefix=args.task_prefix,
        num_shots=args.num_shots,
        fewshot_source=args.fewshot_source,
        shuffle_choices=args.shuffle_choices,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {args.exp_id}  (print-only, nothing saved)")
    print(f"  layers={args.layers}  coefs={args.coefs}")
    print(f"  token_position={args.token_position}  normalize={args.normalize}"
          f"  norm_type={args.norm_type}")
    print("=" * 70)

    all_records: list[dict] = []
    for layer_idx in args.layers:
        print(f"\n--- Layer {layer_idx} ---")
        layer_name = args.layer_pattern.format(layer_idx=layer_idx)
        dim_steerer = DifferenceInMeansSteering(
            model_with_hooks=model_with_hooks,
            tokenizer=tokenizer,
            target_layer=layer_name,
            token_position=args.token_position,
        )
        dim_steerer.capture_positive_activations(pos_prompts)
        dim_steerer.capture_negative_activations(neg_prompts)
        vector = dim_steerer.compute_steering_vector(
            normalize=args.normalize, norm_type=args.norm_type,
        )
        all_records.extend(run_layer(
            scorer, formatter, eval_df, dim_steerer, vector,
            layer_idx, args.coefs, args.verbose_every,
        ))
        dim_steerer.cleanup()

    df = pd.DataFrame(all_records)
    print_summary(df)


if __name__ == "__main__":
    main()
