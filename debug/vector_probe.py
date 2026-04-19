"""
debug/vector_probe.py — DIM vector quality + generation eyeball test.

For each requested layer:
  1. Build the DIM vector from pos/neg prompt files.
  2. Print diagnostics:
       - cos_sim between pos-mean and neg-mean
       - pos/neg mean norms and ratio
       - between-class distance vs within-class variance
       - top-k contributing hidden dims
  3. (Optional) Generate N tokens from --gen-prompt at baseline / +coef / -coef
     and print side by side.

Run:
    python debug/vector_probe.py \
        --pos-file data/prompts/pos.txt \
        --neg-file data/prompts/neg.txt \
        --layers 5 15 20 \
        --coef 3 \
        --gen-prompt "The cranial nerve responsible for vision is the" \
        --out debug/out/vector_quality.csv

Fast pass criteria:
    cos_sim < 0.95, between/within > 0.5, generations at +coef differ coherently.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dim import DifferenceInMeansSteering
from hook import ModelWithHooks


def load_prompts(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No prompts found in {path}")
    return lines


def vector_diagnostics(dim_steerer: DifferenceInMeansSteering,
                       top_k: int = 10) -> dict:
    """Compute quality metrics from already-captured activations."""
    pos = torch.stack(dim_steerer.positive_activations).float()
    neg = torch.stack(dim_steerer.negative_activations).float()

    mu_pos, mu_neg = pos.mean(0), neg.mean(0)
    vec = mu_pos - mu_neg

    cos_sim = torch.nn.functional.cosine_similarity(
        mu_pos.unsqueeze(0), mu_neg.unsqueeze(0)
    ).item()

    between_dist_sq = (mu_pos - mu_neg).pow(2).sum().item()
    within_var_pos = pos.var(dim=0).sum().item()
    within_var_neg = neg.var(dim=0).sum().item()
    within_var = 0.5 * (within_var_pos + within_var_neg)
    bw_ratio = between_dist_sq / within_var if within_var > 0 else float("inf")

    top_idx = vec.abs().topk(top_k).indices.tolist()
    top_vals = [float(vec[i]) for i in top_idx]

    total_sq = vec.pow(2).sum().item()
    top_sq = sum(v * v for v in top_vals)
    top_frac = top_sq / total_sq if total_sq > 0 else 0.0

    return {
        "cos_sim": cos_sim,
        "pos_norm": mu_pos.norm().item(),
        "neg_norm": mu_neg.norm().item(),
        "norm_ratio": mu_pos.norm().item() / (mu_neg.norm().item() + 1e-8),
        "vec_norm": vec.norm().item(),
        "between_dist_sq": between_dist_sq,
        "within_var": within_var,
        "between_over_within": bw_ratio,
        f"top{top_k}_dims_energy_frac": top_frac,
        f"top{top_k}_dim_indices": top_idx,
    }


@torch.no_grad()
def steered_generation(model, tokenizer, model_with_hooks, dim_steerer,
                       vector, coef: float, prompt: str, max_new_tokens: int) -> str:
    """Generate tokens with the steering vector applied at fixed coef."""
    if coef == 0.0:
        model_with_hooks.reset_steering()
    else:
        dim_steerer.apply_steering(vector, coefficient=float(coef))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    model_with_hooks.reset_steering()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--pos-file", type=Path, required=True)
    ap.add_argument("--neg-file", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="Layer indices to probe")
    ap.add_argument("--layer-pattern", default="model.layers.{layer_idx}")
    ap.add_argument("--token-position", default="last", choices=["last", "mean"])
    ap.add_argument("--coef", type=float, default=3.0,
                    help="Coef for generation probe (+coef and -coef are used)")
    ap.add_argument("--gen-prompt", default=None,
                    help="If provided, generate at baseline / +coef / -coef")
    ap.add_argument("--gen-tokens", type=int, default=40)
    ap.add_argument("--top-k-dims", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("debug/out/vector_quality.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pos_prompts = load_prompts(args.pos_file)
    neg_prompts = load_prompts(args.neg_file)
    print(f"Loaded {len(pos_prompts)} pos / {len(neg_prompts)} neg prompts")

    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device
    )
    model.eval()
    model_with_hooks = ModelWithHooks(model)

    rows = []

    for layer_idx in args.layers:
        layer_name = args.layer_pattern.format(layer_idx=layer_idx)
        print(f"\n{'='*70}\n  Layer {layer_idx}  ({layer_name})\n{'='*70}")

        dim_steerer = DifferenceInMeansSteering(
            model_with_hooks=model_with_hooks,
            tokenizer=tokenizer,
            target_layer=layer_name,
            token_position=args.token_position,
        )
        dim_steerer.capture_positive_activations(pos_prompts)
        dim_steerer.capture_negative_activations(neg_prompts)
        vector = dim_steerer.compute_steering_vector(normalize=False)

        diag = vector_diagnostics(dim_steerer, top_k=args.top_k_dims)
        diag = {"layer": layer_idx, **diag}

        print(f"  cos_sim(μ+, μ−)         : {diag['cos_sim']:.4f}   "
              f"(< 0.95 = signal exists)")
        print(f"  pos_norm / neg_norm     : {diag['pos_norm']:.2f} / "
              f"{diag['neg_norm']:.2f}   (ratio {diag['norm_ratio']:.3f})")
        print(f"  vec_norm                : {diag['vec_norm']:.4f}")
        print(f"  between² / within_var   : {diag['between_over_within']:.4f}   "
              f"(> 0.5 = usable)")
        print(f"  top-{args.top_k_dims} dims energy    : "
              f"{diag[f'top{args.top_k_dims}_dims_energy_frac']*100:.1f}%   "
              f"(>50% = vector is sparse, suspicious)")
        print(f"  top-{args.top_k_dims} dim indices   : "
              f"{diag[f'top{args.top_k_dims}_dim_indices']}")

        if args.gen_prompt:
            print(f"\n  -- Generation probe (prompt: {args.gen_prompt!r}) --")
            for coef in [0.0, args.coef, -args.coef]:
                text = steered_generation(
                    model, tokenizer, model_with_hooks, dim_steerer,
                    vector, coef, args.gen_prompt, args.gen_tokens,
                )
                tag = "baseline" if coef == 0.0 else f"coef={coef:+g}"
                print(f"  [{tag:>12s}] {text.strip()}")

        rows.append(diag)
        dim_steerer.cleanup() if hasattr(dim_steerer, "cleanup") else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
