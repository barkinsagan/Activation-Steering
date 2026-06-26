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
        vec = dim.compute_steering_vector(normalize=s.normalize_vector, norm_type=s.norm_type)
        # vec is on CPU (compute_steering_vector calls .cpu() internally)

        # Project each set
        for set_name, prompts in all_sets.items():
            # Reuse capture mechanism to get activations; overwrites stored lists (fine — DIM already computed)
            dim.capture_positive_activations(prompts)
            acts = dim.positive_activations   # list of [H] tensors, on CPU

            projs = _project(acts, vec)
            mean, std, lo, hi = _stats(projs)

            print(f"  {set_name:35s}  mean={mean:+.4f}  std={std:.4f}  [{lo:+.4f}, {hi:+.4f}]")
            records.append({
                "layer":    layer_idx,
                "dataset":  set_name,
                "n":        len(projs),
                "mean":     mean,
                "std":      std,
                "min":      lo,
                "max":      hi,
            })

        dim.cleanup()

    # ------------------------------------------------------------------ summary
    results = pd.DataFrame(records)

    print("\n\n" + "═" * 78)
    print("SUMMARY TABLE")
    print("═" * 78)
    # Pivot: rows = layer, columns = dataset, values = mean projection
    pivot = results.pivot(index="layer", columns="dataset", values="mean")
    with pd.option_context("display.float_format", lambda x: f"{x:+.4f}",
                           "display.max_columns", None, "display.width", 200):
        print(pivot.to_string())

    print(
        "\nInterpretation:"
        f"\n  pos capture ({Path(pos_csv).stem}) is the reference — should always score highest."
        f"\n  neg capture ({Path(neg_csv).stem}) is the reference — should always score lowest."
        "\n  Probes that cluster with pos → captured same signal as pos."
        "\n  Probes that cluster with neg → captured same signal as neg."
    )

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.save / "dim_projections.csv", index=False)
        pivot.to_csv(args.save / "dim_projections_pivot.csv")
        print(f"\nSaved to {args.save}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
