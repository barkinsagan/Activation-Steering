"""
knn_locality.py — per-layer k-NN label locality analysis.

Captures activations from 8 physics/biology datasets in a single
forward-pass sweep per dataset, then for each layer computes:

  Purity     : fraction of k nearest neighbours sharing the same label
               (cosine similarity, averaged over all points)
  Accuracy   : leave-one-out k-NN accuracy (cosine metric, manual LOO
               over precomputed similarity matrix — no sklearn overhead)
  Silhouette : sklearn silhouette_score on cosine distance matrix

Two label granularities:
  binary     : physics (1) vs. biology (0)   — 8 datasets, 2 classes
  fine       : each dataset as its own class  — 8 datasets, 8 classes

Reference sets (all_phys, mmlu_general) are intentionally excluded:
all_phys duplicates the individual physics embeddings; mmlu_general has
no physics/bio label.

Usage:
    python analysis/knn_locality.py \\
        --config configs/exp_20260626_gpqa_phys_vs_bio_llama8b.yaml \\
        --layers 0 4 8 12 16 20 24 28 31 \\
        --save results/knn_locality/

    # Custom k values:
    python analysis/knn_locality.py ... --k 5 15 30

    # Skip model load (dry-run with random embeddings for testing):
    python analysis/knn_locality.py ... --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from hook import ModelWithHooks
from analysis.geometry_sweep import (
    _DS_SPECS, DS_KEYS, DS_NAME, DS_COLOR, DS_MARKER, DS_IS_REF,
    load_prompts, capture_all_layers,
)

# ── Dataset registry (non-reference only) ────────────────────────────────────

# 8 non-ref datasets used for locality analysis
ANALYSIS_KEYS: List[str] = [s[0] for s in _DS_SPECS if not s[5]]

# binary label: 1=physics, 0=biology (same ordering as _DS_SPECS)
_BINARY_LABEL: Dict[str, int] = {
    s[0]: (1 if s[2] is True else 0)
    for s in _DS_SPECS if not s[5]
}

# fine-grained label: integer index per dataset
_FINE_LABEL: Dict[str, int] = {k: i for i, k in enumerate(ANALYSIS_KEYS)}

# colours used in plots (one per k value)
_K_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ── Core metrics (all share a precomputed similarity matrix) ─────────────────

def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Return the N×N cosine similarity matrix for rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xu = X / norms
    return Xu @ Xu.T  # [N, N]


def knn_purity(sim: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Mean fraction of a point's k nearest neighbours that share its label.
    Uses precomputed cosine similarity matrix (self excluded via fill_diagonal).
    """
    s = sim.copy()
    np.fill_diagonal(s, -2.0)
    N = len(labels)
    total = 0.0
    for i in range(N):
        top_k = np.argsort(s[i])[-k:]
        total += np.sum(labels[top_k] == labels[i]) / k
    return total / N


def knn_loo_accuracy(sim: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    LOO k-NN accuracy using precomputed similarity matrix.
    For each point i: exclude it, find k closest among remainder, majority-vote.
    """
    s = sim.copy()
    np.fill_diagonal(s, -2.0)
    n_classes = int(labels.max()) + 1
    correct = 0
    for i in range(len(labels)):
        top_k = np.argsort(s[i])[-k:]
        counts = np.bincount(labels[top_k], minlength=n_classes)
        if np.argmax(counts) == labels[i]:
            correct += 1
    return correct / len(labels)


def silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score on a precomputed cosine distance matrix."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(dist, labels, metric="precomputed"))


# ── Per-layer analysis ────────────────────────────────────────────────────────

def analyse_layer(
    layer_acts: Dict[str, Optional[List[torch.Tensor]]],
    k_values: List[int],
) -> Dict:
    """
    Build X + label arrays from layer_acts, compute all metrics.
    Returns a flat dict: purity_binary_k5, accuracy_fine_k10, silhouette_binary, …
    """
    vecs, bin_labels, fine_labels = [], [], []
    for key in ANALYSIS_KEYS:
        acts = layer_acts.get(key)
        if not acts:
            continue
        b = _BINARY_LABEL[key]
        f = _FINE_LABEL[key]
        for v in acts:
            vecs.append(v.float().numpy())
            bin_labels.append(b)
            fine_labels.append(f)

    if not vecs:
        return {}

    X          = np.stack(vecs)                   # [N, H]
    bin_arr    = np.array(bin_labels, dtype=int)  # [N]
    fine_arr   = np.array(fine_labels, dtype=int) # [N]

    # Compute similarity + distance matrices once
    sim  = _cosine_sim_matrix(X)
    dist = np.clip(1.0 - sim, 0.0, 2.0)

    result: Dict = {"n_points": len(X)}

    result["silhouette_binary"] = silhouette(dist, bin_arr)
    result["silhouette_fine"]   = silhouette(dist, fine_arr)

    for k in k_values:
        tag = f"k{k}"
        if len(X) < k + 1:
            result[f"purity_binary_{tag}"]   = float("nan")
            result[f"accuracy_binary_{tag}"] = float("nan")
            result[f"purity_fine_{tag}"]     = float("nan")
            result[f"accuracy_fine_{tag}"]   = float("nan")
        else:
            result[f"purity_binary_{tag}"]   = knn_purity(sim, bin_arr, k)
            result[f"accuracy_binary_{tag}"] = knn_loo_accuracy(sim, bin_arr, k)
            result[f"purity_fine_{tag}"]     = knn_purity(sim, fine_arr, k)
            result[f"accuracy_fine_{tag}"]   = knn_loo_accuracy(sim, fine_arr, k)

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_metrics(
    records: List[Dict],
    k_values: List[int],
    save_dir: Optional[Path] = None,
) -> None:
    layers = [r["layer"] for r in records]
    n_fine_classes = len(ANALYSIS_KEYS)
    chance_binary  = 0.5
    chance_fine    = 1.0 / n_fine_classes

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("k-NN Label Locality Across Layers", fontsize=14, fontweight="bold")

    def _plot_k_lines(ax, metric_prefix: str, chance: float, ylabel: str, marker: str) -> None:
        for ki, k in enumerate(k_values):
            ys = [r.get(f"{metric_prefix}_k{k}", float("nan")) for r in records]
            ax.plot(layers, ys, marker=marker,
                    color=_K_COLORS[ki % len(_K_COLORS)],
                    label=f"k={k}", linewidth=1.8, markersize=5)
        ax.axhline(chance, color="gray", lw=0.8, linestyle="--", alpha=0.6, label="chance")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.set_xticks(layers)
        ax.tick_params(axis="x", labelsize=7)

    def _plot_silhouette(ax, metric_key: str, title: str, color: str) -> None:
        ys = [r.get(metric_key, float("nan")) for r in records]
        ax.plot(layers, ys, marker="D", color=color, linewidth=1.8, markersize=5)
        ax.axhline(0.0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Silhouette score")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.set_xticks(layers)
        ax.tick_params(axis="x", labelsize=7)

    # Row 0: binary labels (physics vs biology)
    axes[0, 0].set_title("Purity — Binary (phys vs bio)", fontsize=10)
    _plot_k_lines(axes[0, 0], "purity_binary",   chance_binary, "k-NN purity",   "o")

    axes[0, 1].set_title("LOO Accuracy — Binary", fontsize=10)
    _plot_k_lines(axes[0, 1], "accuracy_binary", chance_binary, "LOO accuracy",  "s")

    _plot_silhouette(axes[0, 2], "silhouette_binary",
                     "Silhouette — Binary", "#9467bd")

    # Row 1: fine-grained labels (8 dataset classes)
    axes[1, 0].set_title(f"Purity — Fine ({n_fine_classes} classes)", fontsize=10)
    _plot_k_lines(axes[1, 0], "purity_fine",   chance_fine, "k-NN purity",  "o")

    axes[1, 1].set_title(f"LOO Accuracy — Fine ({n_fine_classes} classes)", fontsize=10)
    _plot_k_lines(axes[1, 1], "accuracy_fine", chance_fine, "LOO accuracy", "s")

    _plot_silhouette(axes[1, 2], "silhouette_fine",
                     "Silhouette — Fine", "#8c564b")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / "knn_locality.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {p}")
    else:
        plt.show()


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(records: List[Dict], k_values: List[int]) -> None:
    col_w = 9

    header_parts = ["Layer", "N"] + [f"pur_b_k{k}" for k in k_values] + \
                   [f"acc_b_k{k}" for k in k_values] + ["sil_b"] + \
                   [f"pur_f_k{k}" for k in k_values] + \
                   [f"acc_f_k{k}" for k in k_values] + ["sil_f"]
    header = "  ".join(f"{h:>{col_w}}" for h in header_parts)
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print("k-NN LOCALITY SUMMARY")
    print(f"  b = binary (phys/bio)  |  f = fine ({len(ANALYSIS_KEYS)} classes)")
    print(sep)
    print(header)
    print(sep)

    for r in records:
        def _f(key: str) -> str:
            v = r.get(key, float("nan"))
            return f"{v:.4f}" if not np.isnan(v) else "  nan"

        row_parts = (
            [f"{r['layer']:>{col_w}}", f"{r.get('n_points', 0):>{col_w}}"] +
            [_f(f"purity_binary_k{k}")   for k in k_values] +
            [_f(f"accuracy_binary_k{k}") for k in k_values] +
            [_f("silhouette_binary")] +
            [_f(f"purity_fine_k{k}")     for k in k_values] +
            [_f(f"accuracy_fine_k{k}")   for k in k_values] +
            [_f("silhouette_fine")]
        )
        print("  ".join(f"{v:>{col_w}}" for v in row_parts))

    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True,
                    help="Experiment YAML (model settings only)")
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="Layer indices to analyse, e.g. --layers 0 4 8 12")
    ap.add_argument("--k",    type=int, nargs="+", default=[5, 10, 20],
                    help="k values for k-NN metrics (default: 5 10 20)")
    ap.add_argument("--n",    type=int, default=100,
                    help="Max prompts per dataset (default 100)")
    ap.add_argument("--save", type=Path, default=None,
                    help="Output directory for plot and CSV (omit to display plot only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true",
                    help="Skip model load; use random embeddings (dim=64) for testing")

    # Dataset path overrides (same defaults as geometry_sweep.py)
    ap.add_argument("--gpqa_phys",  default="data/eval/gpqa_main_physics_sweep.csv")
    ap.add_argument("--gpqa_bio",   default="data/eval/gpqa_main_biology_sweep.csv")
    ap.add_argument("--mmlu_phys",  default="data/eval/mmlu_physics_sweep.csv")
    ap.add_argument("--mmlu_bio",   default="data/eval/mmlu_biology_sweep.csv")
    ap.add_argument("--text_phys",  default="data/prompts/physics_pos.txt")
    ap.add_argument("--text_bio",   default="data/prompts/biology_neg.txt")
    ap.add_argument("--arxiv_phys", default="data/eval/arxiv_physics.txt")
    ap.add_argument("--arxiv_bio",  default="data/eval/biorxiv_biology.txt")
    ap.add_argument("--sublayer",   default=None,
                    help="Sub-component appended to layer name pattern "
                         "(e.g. mlp, self_attn, mlp.down_proj)")
    args = ap.parse_args()

    k_values = sorted(set(args.k))
    print(f"k values: {k_values}")
    print(f"Layers  : {args.layers}")
    print(f"n/dataset: {args.n}")

    # ── Read model config ──
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    model_raw  = raw["model"]
    sweep_raw  = raw.get("sweep", {})
    layer_pat  = sweep_raw.get("layer_name_pattern", "model.layers.{layer_idx}")
    tok_pos    = sweep_raw.get("token_position", "last")
    cap_bs     = sweep_raw.get("capture_batch_size", 16)

    sublayer = args.sublayer or sweep_raw.get("sublayer") or None
    if sublayer:
        layer_pat = f"{layer_pat}.{sublayer}"

    layer_names = [layer_pat.format(layer_idx=i) for i in args.layers]
    print(f"Layer names: {layer_names}")

    # ── Load prompts ──
    ds_path_args: Dict[str, str] = {
        "gpqa_phys":  args.gpqa_phys,
        "gpqa_bio":   args.gpqa_bio,
        "mmlu_phys":  args.mmlu_phys,
        "mmlu_bio":   args.mmlu_bio,
        "text_phys":  args.text_phys,
        "text_bio":   args.text_bio,
        "arxiv_phys": args.arxiv_phys,
        "arxiv_bio":  args.arxiv_bio,
    }

    print("\nLoading datasets:")
    ds_prompts: Dict[str, Optional[List[str]]] = {}
    for key in ANALYSIS_KEYS:
        path = ds_path_args[key]
        prompts = load_prompts(path, args.n, args.seed)
        ds_prompts[key] = prompts
        status = f"{len(prompts)} prompts" if prompts else "MISSING"
        print(f"  {key:12s}  {status}  ({path})")

    # ── Activation capture (or dry-run stub) ──
    # acts_store[key][layer_name] = List[torch.Tensor([H])]
    acts_store: Dict[str, Optional[Dict[str, List[torch.Tensor]]]] = {}

    if args.dry_run:
        print("\n[dry_run] Using random embeddings (dim=64).")
        rng = np.random.default_rng(args.seed)
        # physics clusters around +1 in first dim, bio around -1
        for ki, key in enumerate(ANALYSIS_KEYS):
            n = args.n
            is_phys = _BINARY_LABEL[key] == 1
            X = rng.standard_normal((n, 64)).astype(np.float32)
            X[:, 0] += 3.0 if is_phys else -3.0
            acts_store[key] = {
                ln: [torch.from_numpy(X[j]) for j in range(n)]
                for ln in layer_names
            }
    else:
        dtype_map = {"float16": torch.float16,
                     "bfloat16": torch.bfloat16,
                     "float32": torch.float32}
        dtype      = dtype_map[model_raw.get("dtype", "bfloat16")]
        model_name = model_raw["name"]
        device     = model_raw.get("device", "cuda")

        print(f"\nLoading model: {model_name}  (dtype={model_raw.get('dtype')}, device={device})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device
        )
        model.eval()
        print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
        mwh = ModelWithHooks(model)

        print("\nCapturing activations (all layers in one sweep per dataset)...")
        for key in ANALYSIS_KEYS:
            prompts = ds_prompts.get(key)
            if not prompts:
                acts_store[key] = None
                continue
            print(f"  {key} ({len(prompts)} prompts)...")
            acts_store[key] = capture_all_layers(
                mwh, tokenizer, prompts, layer_names,
                token_position=tok_pos, batch_size=cap_bs,
            )

    # ── Per-layer metrics ──
    records: List[Dict] = []

    for layer_idx, layer_name in zip(args.layers, layer_names):
        print(f"\nLayer {layer_idx}  ({layer_name})")

        layer_acts: Dict[str, Optional[List[torch.Tensor]]] = {}
        for key in ANALYSIS_KEYS:
            store = acts_store.get(key)
            layer_acts[key] = store.get(layer_name) if store else None

        metrics = analyse_layer(layer_acts, k_values)
        if not metrics:
            print("  [SKIP] No activations available.")
            continue

        metrics["layer"] = layer_idx
        records.append(metrics)

        # Quick per-layer summary
        n = metrics.get("n_points", 0)
        sil_b = metrics.get("silhouette_binary", float("nan"))
        sil_f = metrics.get("silhouette_fine",   float("nan"))
        print(f"  N={n}  sil_binary={sil_b:.4f}  sil_fine={sil_f:.4f}")
        for k in k_values:
            pur_b = metrics.get(f"purity_binary_k{k}",   float("nan"))
            acc_b = metrics.get(f"accuracy_binary_k{k}", float("nan"))
            pur_f = metrics.get(f"purity_fine_k{k}",     float("nan"))
            acc_f = metrics.get(f"accuracy_fine_k{k}",   float("nan"))
            print(f"    k={k:2d}  pur_b={pur_b:.4f}  acc_b={acc_b:.4f}"
                  f"  |  pur_f={pur_f:.4f}  acc_f={acc_f:.4f}")

    if not records:
        print("\nNo records — nothing to plot or save.")
        return

    # ── Sort by layer (supports multi-run merging) ──
    records.sort(key=lambda r: r["layer"])

    # ── Summary table ──
    print_table(records, k_values)

    # ── Save CSV ──
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        csv_path = args.save / "knn_locality.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path}")

    # ── Plot ──
    plot_metrics(records, k_values, save_dir=args.save)

    print("Done.")


if __name__ == "__main__":
    main()
