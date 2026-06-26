"""
Option A: Project question sets onto the per-layer DIM vector.

Recomputes the DIM vector from the original capture sets, then projects
additional question sets onto it. This reveals what the vector discriminates:

  - If direction is PHYSICS-DOMAIN:   MMLU-physics  ≈ GPQA-physics  >> MMLU-nonmed
  - If direction is GPQA-STYLE:       GPQA-biology  ≈ GPQA-physics  >> MMLU-anything
  - If direction is pure noise:       all four sets overlap heavily

Usage:
    python analysis/probe_dim_direction.py configs/exp_20260602_gpqa_physics_llama8b.yaml \\
        --probe gpqa_biology:data/eval/gpqa_main_biology_sweep.csv \\
        --probe gpqa_chemistry:data/eval/gpqa_main_chemistry_sweep.csv \\
        --layers 10 13 16 20 \\
        --n 40

    # minimal run (just pos vs neg, all default layers):
    python analysis/probe_dim_direction.py configs/exp_20260602_gpqa_physics_llama8b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mcf_prompts(csv_path: str, n: int, seed: int) -> List[str]:
    """Format up to n rows from csv_path as zero-shot MCF prompts."""
    import random
    from olmes.formatter import build_formatter

    df = pd.read_csv(csv_path)
    df = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)

    fmt = build_formatter(task_prefix="question", num_shots=0, shuffle_choices=True, seed=seed)
    prompts = []
    for i, (_, row) in enumerate(df.iterrows()):
        prompts.append(fmt.format_mcf(row, question_idx=i).prompt)
    return prompts


def _project(acts: List[torch.Tensor], vec: torch.Tensor) -> List[float]:
    """Dot-product projection of each activation onto vec. Returns list of floats."""
    stacked = torch.stack(acts).float()       # [N, H]
    v = vec.float()                            # [H]
    return (stacked @ v).tolist()             # [N]


def _project_cosine(acts: List[torch.Tensor], vec: torch.Tensor) -> List[float]:
    """Cosine-similarity projection of each activation onto vec.

    Removes the activation-norm confound: dot product grows with ||act||,
    cosine does not. Useful for comparing across layers where activation
    magnitudes differ.
    """
    stacked = torch.stack(acts).float()       # [N, H]
    v = vec.float()                            # [H]
    return torch.nn.functional.cosine_similarity(stacked, v.unsqueeze(0), dim=-1).tolist()


def _activation_norms(acts: List[torch.Tensor]) -> List[float]:
    """L2 norm of each activation. Used to expose layer-wise norm growth."""
    stacked = torch.stack(acts).float()
    return stacked.norm(dim=-1).tolist()


def _split_half_stability(
    pos_acts: List[torch.Tensor],
    neg_acts: List[torch.Tensor],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[float, float]:
    """Measure reproducibility of the DIM direction across random subsamples.

    Repeatedly splits the captured pos/neg activations into two random halves,
    computes a DIM vector from each half, and measures the cosine similarity
    between the two estimates.

    High mean cos → the direction is the same regardless of which half of the
    samples you use → the direction is real, not a sampling artifact.
    Low mean cos → different subsamples find different directions → the
    estimated direction is largely noise.

    Returns:
        (mean cosine, std cosine) over n_splits independent random partitions.
    """
    pos_stack = torch.stack(pos_acts).float()   # [N_p, H]
    neg_stack = torch.stack(neg_acts).float()   # [N_n, H]
    n_p, n_n = pos_stack.shape[0], neg_stack.shape[0]
    half_p, half_n = n_p // 2, n_n // 2

    if half_p < 2 or half_n < 2:
        return float("nan"), float("nan")

    cosines: List[float] = []
    for split_i in range(n_splits):
        g = torch.Generator().manual_seed(seed + split_i * 1000 + 1)
        p_perm = torch.randperm(n_p, generator=g)
        n_perm = torch.randperm(n_n, generator=g)

        idx_pA, idx_pB = p_perm[:half_p],     p_perm[half_p:2 * half_p]
        idx_nA, idx_nB = n_perm[:half_n],     n_perm[half_n:2 * half_n]

        v_A = pos_stack[idx_pA].mean(0) - neg_stack[idx_nA].mean(0)
        v_B = pos_stack[idx_pB].mean(0) - neg_stack[idx_nB].mean(0)

        c = torch.nn.functional.cosine_similarity(
            v_A.unsqueeze(0), v_B.unsqueeze(0)
        ).item()
        cosines.append(c)

    mean_c = sum(cosines) / len(cosines)
    var_c = sum((c - mean_c) ** 2 for c in cosines) / max(len(cosines) - 1, 1)
    return mean_c, var_c ** 0.5


def _stats(values: List[float]) -> Tuple[float, float, float, float]:
    t = torch.tensor(values)
    return (
        round(t.mean().item(), 4),
        round(t.std().item(), 4),
        round(t.min().item(), 4),
        round(t.max().item(), 4),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config", type=Path, help="Path to experiment YAML config")
    p.add_argument(
        "--probe", metavar="NAME:PATH", action="append", default=[],
        help="Extra question set to project. Repeat for multiple. E.g. gpqa_bio:data/eval/gpqa_main_biology_sweep.csv",
    )
    p.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layers to analyse (default: evenly spaced 8 layers across the model)",
    )
    p.add_argument(
        "--n", type=int, default=40,
        help="Max questions per set (default: 40)",
    )
    p.add_argument("--save", type=Path, default=None)
    args = p.parse_args()

    if not args.config.exists():
        sys.exit(f"[!] Config not found: {args.config}")

    # ------------------------------------------------------------------ setup
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from experiments.config import load_config
    from experiments.registry import load_model, build_capture_prompts_mcf
    from olmes.formatter import build_formatter
    from hook import ModelWithHooks
    from dim import DifferenceInMeansSteering

    cfg = load_config(str(args.config))
    s = cfg.sweep

    # ------------------------------------------------------------------ data
    pos_csv = cfg.dataset.eval_path
    neg_csv = cfg.dataset.neg_capture_path
    if not neg_csv:
        sys.exit("[!] This script requires dataset.neg_capture_path to be set in the config")

    # Reproduce capture prompts with zero-shot MCF formatter (matches runner.py)
    print(f"\nBuilding capture prompts (n={args.n}) ...")
    pos_prompts = _build_mcf_prompts(pos_csv, args.n, s.seed)
    neg_prompts = _build_mcf_prompts(neg_csv, args.n, s.seed)
    print(f"  pos ({Path(pos_csv).stem}): {len(pos_prompts)}")
    print(f"  neg ({Path(neg_csv).stem}): {len(neg_prompts)}")

    # Extra probe sets
    probe_sets: Dict[str, List[str]] = {}
    for spec in args.probe:
        if ":" not in spec:
            sys.exit(f"[!] --probe must be NAME:PATH, got: {spec!r}")
        name, path = spec.split(":", 1)
        if not Path(path).exists():
            sys.exit(f"[!] Probe file not found: {path}")
        probe_sets[name] = _build_mcf_prompts(path, args.n, s.seed)
        print(f"  probe {name}: {len(probe_sets[name])} prompts from {path}")

    all_sets: Dict[str, List[str]] = {
        Path(pos_csv).stem: pos_prompts,
        Path(neg_csv).stem: neg_prompts,
        **probe_sets,
    }

    # ------------------------------------------------------------------ model
    model, tokenizer = load_model(cfg)
    model_with_hooks = ModelWithHooks(model)
    n_layers = model.config.num_hidden_layers

    layers = args.layers
    if layers is None:
        # 8 evenly-spaced layers by default
        step = max(1, n_layers // 8)
        layers = list(range(0, n_layers, step))[:8]
    print(f"\nLayers to analyse: {layers}")

    # ------------------------------------------------------------------ sweep
    records = []
    diagnostics = []   # Per-layer vector geometry: cos(μ_p,μ_n), raw diff norm, etc.

    for layer_idx in layers:
        layer_name = s.layer_pattern.format(layer_idx=layer_idx)
        print(f"\n── Layer {layer_idx} ({layer_name}) ──")

        dim = DifferenceInMeansSteering(
            model_with_hooks=model_with_hooks,
            tokenizer=tokenizer,
            target_layer=layer_name,
            token_position=s.token_position,
            capture_batch_size=s.capture_batch_size,
        )

        # Compute DIM vector from pos/neg capture sets
        dim.capture_positive_activations(pos_prompts)
        dim.capture_negative_activations(neg_prompts)

        # ── Vector diagnostics: compute BEFORE the probe loop overwrites the
        #    activation buffers. These answer: are μ_pos and μ_neg already close
        #    at this layer? Is the within-class spread larger than the
        #    between-class separation (i.e., is the DIM vector mostly noise)?
        pos_stack = torch.stack(dim.positive_activations).float()   # [N_p, H]
        neg_stack = torch.stack(dim.negative_activations).float()   # [N_n, H]
        mu_pos_raw = pos_stack.mean(dim=0)
        mu_neg_raw = neg_stack.mean(dim=0)
        diff_raw = mu_pos_raw - mu_neg_raw
        cos_mu = torch.nn.functional.cosine_similarity(
            mu_pos_raw.unsqueeze(0), mu_neg_raw.unsqueeze(0)
        ).item()
        mu_pos_norm = mu_pos_raw.norm().item()
        mu_neg_norm = mu_neg_raw.norm().item()
        raw_diff_norm = diff_raw.norm().item()

        # Within-class spread: mean L2 distance of each activation from its
        # class centroid. Lets us compute a Fisher-style SNR for the DIM
        # direction — between-class separation divided by pooled within-class
        # noise. SNR < 1 means the unit DIM vector is mostly sampling noise.
        sigma_pos = (pos_stack - mu_pos_raw).norm(dim=-1).mean().item()
        sigma_neg = (neg_stack - mu_neg_raw).norm(dim=-1).mean().item()
        pooled_sigma = ((sigma_pos ** 2 + sigma_neg ** 2) / 2) ** 0.5
        snr = raw_diff_norm / pooled_sigma if pooled_sigma > 0 else float("inf")

        # Split-half reproducibility: does the DIM direction agree across
        # random subsamples of the captured prompts? Tests stability without
        # needing more data — answers "is the direction real, or sample-specific?"
        split_half_cos_mean, split_half_cos_std = _split_half_stability(
            dim.positive_activations,
            dim.negative_activations,
            n_splits=5,
            seed=s.seed,
        )

        diagnostics.append({
            "layer":            layer_idx,
            "cos_mu_pos_mu_neg": round(cos_mu, 4),
            "raw_diff_norm":    round(raw_diff_norm, 4),
            "sigma_pos":        round(sigma_pos, 4),
            "sigma_neg":        round(sigma_neg, 4),
            "snr":              round(snr, 4),
            "split_half_cos_mean": round(split_half_cos_mean, 4),
            "split_half_cos_std":  round(split_half_cos_std, 4),
            "mu_pos_norm":      round(mu_pos_norm, 4),
            "mu_neg_norm":      round(mu_neg_norm, 4),
        })
        print(
            f"  vector geometry: "
            f"cos(μ_p,μ_n)={cos_mu:+.4f}  "
            f"||μ_p−μ_n||={raw_diff_norm:.3f}  "
            f"σ_p={sigma_pos:.3f}  σ_n={sigma_neg:.3f}  "
            f"SNR={snr:.3f}  "
            f"split-half-cos={split_half_cos_mean:+.4f}±{split_half_cos_std:.4f}  "
            f"||μ_p||={mu_pos_norm:.2f}  ||μ_n||={mu_neg_norm:.2f}"
        )

        vec = dim.compute_steering_vector(normalize=s.normalize_vector, norm_type=s.norm_type)
        # vec is on CPU (compute_steering_vector calls .cpu() internally)

        # Project each set (dot product + cosine similarity, plus activation norm)
        for set_name, prompts in all_sets.items():
            # Reuse capture mechanism to get activations; overwrites stored lists (fine — DIM already computed)
            dim.capture_positive_activations(prompts)
            acts = dim.positive_activations   # list of [H] tensors, on CPU

            projs_dot = _project(acts, vec)
            projs_cos = _project_cosine(acts, vec)
            norms     = _activation_norms(acts)

            mean_dot, std_dot, lo_dot, hi_dot = _stats(projs_dot)
            mean_cos, std_cos, lo_cos, hi_cos = _stats(projs_cos)
            mean_norm = round(sum(norms) / len(norms), 3)

            print(
                f"  {set_name:35s}  "
                f"dot={mean_dot:+.4f}  "
                f"cos={mean_cos:+.4f}  "
                f"||act||={mean_norm:.2f}"
            )
            records.append({
                "layer":         layer_idx,
                "dataset":       set_name,
                "n":             len(projs_dot),
                "mean_dot":      mean_dot,
                "std_dot":       std_dot,
                "min_dot":       lo_dot,
                "max_dot":       hi_dot,
                "mean_cos":      mean_cos,
                "std_cos":       std_cos,
                "min_cos":       lo_cos,
                "max_cos":       hi_cos,
                "mean_act_norm": mean_norm,
            })

        dim.cleanup()

    # ------------------------------------------------------------------ summary
    results = pd.DataFrame(records)
    diag_df = pd.DataFrame(diagnostics)

    # ── Vector geometry per layer ──
    print("\n\n" + "═" * 78)
    print("VECTOR DIAGNOSTICS  (per layer, before unit-normalization)")
    print("═" * 78)
    print(
        "  cos(μ_p,μ_n) near 1  → pos and neg activation centroids overlap in direction."
    )
    print(
        "  raw_diff_norm        → between-class separation (centroid-to-centroid distance)."
    )
    print(
        "  σ_p, σ_n             → within-class spread (mean distance of each activation"
        " from its class centroid)."
    )
    print(
        "  SNR = raw_diff_norm / √((σ_p² + σ_n²)/2)"
    )
    print(
        "    SNR > 2  → between-class separation comfortably exceeds within-class noise."
        " DIM direction is meaningful."
    )
    print(
        "    SNR ≈ 1  → ambiguous; means may be sampling artifact in a 4096-dim space."
    )
    print(
        "    SNR < 1  → within-class spread dominates; DIM is mostly noise."
    )
    print(
        "  split_half_cos       → mean cosine between DIM vectors from two random halves"
        " of the data (5 splits)."
    )
    print(
        "    > 0.85  → direction is highly reproducible across subsamples. Real signal."
    )
    print(
        "    0.5–0.85 → partially reproducible; small-n wobble but underlying direction exists."
    )
    print(
        "    < 0.5   → different subsamples find different directions. Dominated by noise."
    )
    with pd.option_context("display.float_format", lambda x: f"{x:+.4f}",
                           "display.max_columns", None, "display.width", 200):
        print(diag_df.to_string(index=False))

    # ── Dot-product projection table (the original) ──
    print("\n" + "═" * 78)
    print("DOT-PRODUCT PROJECTIONS  (act · v_unit)")
    print("═" * 78)
    print(
        "  Mixes alignment with activation magnitude. ||act|| grows with"
        " layer depth in Llama, so cross-layer comparisons here are inflated."
    )
    pivot_dot = results.pivot(index="layer", columns="dataset", values="mean_dot")
    with pd.option_context("display.float_format", lambda x: f"{x:+.4f}",
                           "display.max_columns", None, "display.width", 200):
        print(pivot_dot.to_string())

    # ── Cosine projection table (length-normalized) ──
    print("\n" + "═" * 78)
    print("COSINE-SIMILARITY PROJECTIONS  (act · v_unit / ||act||)")
    print("═" * 78)
    print(
        "  Length-normalized: removes the ||act|| confound. If the L20 drift"
        " disappears here but persists in dot product, the drift was norm growth."
    )
    pivot_cos = results.pivot(index="layer", columns="dataset", values="mean_cos")
    with pd.option_context("display.float_format", lambda x: f"{x:+.4f}",
                           "display.max_columns", None, "display.width", 200):
        print(pivot_cos.to_string())

    # ── Mean activation norm per (layer, dataset) ──
    print("\n" + "═" * 78)
    print("ACTIVATION NORMS  (mean ||act|| per layer × dataset)")
    print("═" * 78)
    print(
        "  Confirms layer-wise norm growth. If norms grow ~k× across layers,"
        " expect dot-product magnitudes to grow ~k× too, independent of alignment."
    )
    pivot_norm = results.pivot(index="layer", columns="dataset", values="mean_act_norm")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}",
                           "display.max_columns", None, "display.width", 200):
        print(pivot_norm.to_string())

    print(
        "\nReading the four tables together:"
        f"\n  pos capture ({Path(pos_csv).stem}) is the reference high."
        f"\n  neg capture ({Path(neg_csv).stem}) is the reference low."
        "\n  Probes clustering with pos → same signal as pos."
        "\n  Probes clustering with neg → same signal as neg."
        "\n  cos(μ_p,μ_n) trending toward 1 at deep layers + cosine projections"
        "\n    staying separated → late-layer dot-product drift was norm-driven."
        "\n  cos(μ_p,μ_n) → 1 + cosine projections also drifting → format/universal"
        "\n    direction is leaking past the style-matched cancellation."
    )

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.save / "dim_projections.csv", index=False)
        pivot_dot.to_csv(args.save / "dim_projections_dot_pivot.csv")
        pivot_cos.to_csv(args.save / "dim_projections_cos_pivot.csv")
        pivot_norm.to_csv(args.save / "activation_norms_pivot.csv")
        diag_df.to_csv(args.save / "vector_diagnostics.csv", index=False)
        print(f"\nSaved to {args.save}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
