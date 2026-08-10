"""
knn_locality.py — per-layer k-NN label locality analysis.

Captures activations from 8 physics/biology datasets in a SINGLE forward-pass
sweep (all datasets pooled, then split by index), then for each layer computes:

  Purity     : fraction of k nearest neighbours sharing the same label
  Accuracy   : leave-one-out k-NN accuracy (cosine metric)
  Silhouette : sklearn silhouette_score on cosine distance matrix

Two label granularities:
  binary     : physics (1) vs. biology (0)   — 2 classes
  fine       : each dataset as its own class  — 8 classes

Per-dataset breakdown + confusion analysis for misclassified points:
for each dataset's wrong predictions, shows what fraction of their k-NN
came from each other dataset (the "predicted class" distribution).

Reference sets (all_phys, mmlu_general) excluded — all_phys duplicates
the individual physics embeddings.

Usage:
    python analysis/knn_locality.py \\
        --config configs/exp_20260626_gpqa_phys_vs_bio_llama8b.yaml \\
        --layers 0 4 8 12 16 20 24 28 31 \\
        --save results/knn_locality/

    python analysis/knn_locality.py ... --k 5 15 30
    python analysis/knn_locality.py ... --dry_run
"""

from __future__ import annotations

import argparse
import math
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
    _DS_SPECS, DS_KEYS, DS_NAME, DS_COLOR, DS_IS_REF,
    load_prompts, capture_all_layers,
)

# ── Dataset registry (non-reference only) ────────────────────────────────────

ANALYSIS_KEYS: List[str] = [s[0] for s in _DS_SPECS if not s[5]]
N_DS = len(ANALYSIS_KEYS)

_BINARY_LABEL: Dict[str, int] = {
    s[0]: (1 if s[2] is True else 0)
    for s in _DS_SPECS if not s[5]
}

_FINE_LABEL: Dict[str, int] = {k: i for i, k in enumerate(ANALYSIS_KEYS)}

_K_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ── Core metrics ──────────────────────────────────────────────────────────────

def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xu = X / norms
    return Xu @ Xu.T  # [N, N]


def knn_purity(sim: np.ndarray, labels: np.ndarray, k: int,
               row_mask: Optional[np.ndarray] = None) -> float:
    s = sim.copy()
    np.fill_diagonal(s, -2.0)
    rows = np.where(row_mask)[0] if row_mask is not None else np.arange(len(labels))
    if len(rows) == 0:
        return float("nan")
    total = 0.0
    for i in rows:
        top_k = np.argsort(s[i])[-k:]
        total += np.sum(labels[top_k] == labels[i]) / k
    return total / len(rows)


def knn_loo_accuracy(sim: np.ndarray, labels: np.ndarray, k: int) -> float:
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
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(dist, labels, metric="precomputed"))


def confusion_matrix_knn(
    sim: np.ndarray,
    fine_arr: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute two N_DS × N_DS confusion matrices.

    all_conf[i, j]   = mean fraction of k-NN from dataset j, for ALL points
                       in dataset i. Diagonal = fine purity (same-dataset fraction).
    wrong_conf[i, j] = same, but only for points whose majority-vote prediction
                       ≠ true dataset i. NaN row if no wrong predictions.
    wrong_n[i]       = number of wrong predictions for dataset i.

    Rows  = true dataset
    Cols  = where k-NN come from
    """
    s = sim.copy()
    np.fill_diagonal(s, -2.0)

    all_conf   = np.zeros((N_DS, N_DS))
    wrong_conf = np.zeros((N_DS, N_DS))
    wrong_n    = np.zeros(N_DS, dtype=int)
    ds_counts  = np.zeros(N_DS, dtype=int)

    for i in range(len(fine_arr)):
        true_ds = fine_arr[i]
        top_k   = np.argsort(s[i])[-k:]
        neighbor_labels = fine_arr[top_k]

        counts  = np.bincount(neighbor_labels, minlength=N_DS)
        pred_ds = int(np.argmax(counts))

        # all_conf: full k-NN distribution (shows neighbourhood composition)
        all_conf[true_ds] += counts.astype(float) / k
        ds_counts[true_ds] += 1

        # wrong_conf: one vote for the winning predicted class only
        if pred_ds != true_ds:
            wrong_conf[true_ds, pred_ds] += 1
            wrong_n[true_ds]             += 1

    # Normalise
    for i in range(N_DS):
        if ds_counts[i] > 0:
            all_conf[i] /= ds_counts[i]
        if wrong_n[i] > 0:
            wrong_conf[i] /= wrong_n[i]
        else:
            wrong_conf[i] = float("nan")

    return all_conf, wrong_conf, wrong_n


# ── Per-layer analysis ────────────────────────────────────────────────────────

def analyse_layer(
    layer_acts: Dict[str, Optional[List[torch.Tensor]]],
    k_values: List[int],
) -> Dict:
    """
    Returns a flat metrics dict plus numpy arrays for confusion matrices:
      conf_all_k{k}   : ndarray [N_DS, N_DS]  all-points k-NN distribution
      conf_wrong_k{k} : ndarray [N_DS, N_DS]  wrong-only k-NN distribution
      conf_wrong_n_k{k}: ndarray [N_DS]        count of wrong predictions per dataset
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

    X        = np.stack(vecs)
    bin_arr  = np.array(bin_labels, dtype=int)
    fine_arr = np.array(fine_labels, dtype=int)

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
            for key in ANALYSIS_KEYS:
                result[f"per_ds_binary_{key}_{tag}"] = float("nan")
                result[f"per_ds_fine_{key}_{tag}"]   = float("nan")
            result[f"conf_all_k{k}"]    = np.full((N_DS, N_DS), float("nan"))
            result[f"conf_wrong_k{k}"]  = np.full((N_DS, N_DS), float("nan"))
            result[f"conf_wrong_n_k{k}"]= np.zeros(N_DS, dtype=int)
        else:
            result[f"purity_binary_{tag}"]   = knn_purity(sim, bin_arr, k)
            result[f"accuracy_binary_{tag}"] = knn_loo_accuracy(sim, bin_arr, k)
            result[f"purity_fine_{tag}"]     = knn_purity(sim, fine_arr, k)
            result[f"accuracy_fine_{tag}"]   = knn_loo_accuracy(sim, fine_arr, k)

            for key in ANALYSIS_KEYS:
                mask = fine_arr == _FINE_LABEL[key]
                result[f"per_ds_binary_{key}_{tag}"] = knn_purity(sim, bin_arr,  k, mask)
                result[f"per_ds_fine_{key}_{tag}"]   = knn_purity(sim, fine_arr, k, mask)

            all_c, wrong_c, wrong_n = confusion_matrix_knn(sim, fine_arr, k)
            result[f"conf_all_k{k}"]     = all_c
            result[f"conf_wrong_k{k}"]   = wrong_c
            result[f"conf_wrong_n_k{k}"] = wrong_n

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_aggregate(records, k_values, save_dir=None):
    layers = [r["layer"] for r in records]
    chance_binary = 0.5
    chance_fine   = 1.0 / N_DS

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("k-NN Label Locality — Aggregate", fontsize=14, fontweight="bold")

    def _lines(ax, prefix, chance, ylabel, marker):
        for ki, k in enumerate(k_values):
            ys = [r.get(f"{prefix}_k{k}", float("nan")) for r in records]
            ax.plot(layers, ys, marker=marker,
                    color=_K_COLORS[ki % len(_K_COLORS)],
                    label=f"k={k}", linewidth=1.8, markersize=5)
        ax.axhline(chance, color="gray", lw=0.8, linestyle="--", alpha=0.6, label="chance")
        ax.set_xlabel("Layer"); ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, linestyle=":"); ax.set_xticks(layers)
        ax.tick_params(axis="x", labelsize=7)

    def _sil(ax, key, title, color):
        ys = [r.get(key, float("nan")) for r in records]
        ax.plot(layers, ys, marker="D", color=color, linewidth=1.8, markersize=5)
        ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Layer"); ax.set_ylabel("Silhouette"); ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2, linestyle=":"); ax.set_xticks(layers)
        ax.tick_params(axis="x", labelsize=7)

    axes[0, 0].set_title("Purity — Binary (phys vs bio)", fontsize=10)
    _lines(axes[0, 0], "purity_binary",   chance_binary, "k-NN purity",  "o")
    axes[0, 1].set_title("LOO Accuracy — Binary", fontsize=10)
    _lines(axes[0, 1], "accuracy_binary", chance_binary, "LOO accuracy", "s")
    _sil(axes[0, 2], "silhouette_binary", "Silhouette — Binary", "#9467bd")

    axes[1, 0].set_title(f"Purity — Fine ({N_DS} classes)", fontsize=10)
    _lines(axes[1, 0], "purity_fine",   chance_fine, "k-NN purity",  "o")
    axes[1, 1].set_title(f"LOO Accuracy — Fine ({N_DS} classes)", fontsize=10)
    _lines(axes[1, 1], "accuracy_fine", chance_fine, "LOO accuracy", "s")
    _sil(axes[1, 2], "silhouette_fine", "Silhouette — Fine", "#8c564b")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / "knn_locality_aggregate.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Plot saved: {p}")
    else:
        plt.show()


def plot_per_dataset(records, k_values, save_dir=None):
    """Heatmap: datasets × layers, one row pair per k (binary purity, fine purity)."""
    layers    = [r["layer"] for r in records]
    ds_labels = [DS_NAME[k] for k in ANALYSIS_KEYS]
    n_k       = len(k_values)

    fig, axes = plt.subplots(n_k, 2,
                             figsize=(max(10, len(layers) * 0.9 + 3), n_k * 5),
                             constrained_layout=True)
    if n_k == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("k-NN Purity — Per Dataset", fontsize=13, fontweight="bold")

    for row, k in enumerate(k_values):
        for col, (ptype, title_suffix) in enumerate([
            ("binary", "Binary purity (phys/bio)"),
            ("fine",   "Fine purity (same dataset)"),
        ]):
            ax  = axes[row, col]
            mat = np.full((N_DS, len(layers)), float("nan"))
            for li, r in enumerate(records):
                for di, ds_key in enumerate(ANALYSIS_KEYS):
                    mat[di, li] = r.get(f"per_ds_{ptype}_{ds_key}_k{k}", float("nan"))

            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                           vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=8)
            ax.set_yticks(range(N_DS));        ax.set_yticklabels(ds_labels, fontsize=8)
            ax.set_xlabel("Layer", fontsize=9)
            ax.set_title(f"k={k}  —  {title_suffix}", fontsize=9)

            for di in range(N_DS):
                for li in range(len(layers)):
                    v = mat[di, li]
                    if not np.isnan(v):
                        tc = "white" if v < 0.35 or v > 0.85 else "black"
                        ax.text(li, di, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color=tc)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="purity")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / "knn_locality_per_dataset.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Plot saved: {p}")
    else:
        plt.show()


def plot_confusion(records, k_values, save_dir=None):
    """
    For each k: one figure with n_layers rows × 2 cols.
    Left  = all-points confusion matrix (row=true, col=predicted distribution).
    Right = wrong-only confusion matrix (only misclassified points).
    Diagonal of the left matrix = fine purity.
    """
    ds_labels = [DS_NAME[k] for k in ANALYSIS_KEYS]

    for k in k_values:
        n_layers = len(records)
        fig, axes = plt.subplots(
            n_layers, 2,
            figsize=(14, n_layers * 4.5),
            constrained_layout=True,
        )
        if n_layers == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f"k-NN Confusion (fine, k={k})\n"
                     f"Left: row=true, col=k-NN fraction  |  "
                     f"Right: row=true, col=predicted class (wrong points only)",
                     fontsize=12, fontweight="bold")

        for row, r in enumerate(records):
            layer_idx = r["layer"]
            all_c   = r.get(f"conf_all_k{k}",     np.full((N_DS, N_DS), float("nan")))
            wrong_c = r.get(f"conf_wrong_k{k}",   np.full((N_DS, N_DS), float("nan")))
            wrong_n = r.get(f"conf_wrong_n_k{k}",  np.zeros(N_DS, dtype=int))

            for col, (mat, title) in enumerate([
                (all_c,   f"Layer {layer_idx} — All points"),
                (wrong_c, f"Layer {layer_idx} — Wrong predictions only"),
            ]):
                ax = axes[row, col]

                # Mask rows that are all-NaN (datasets not present or no wrong preds)
                valid_rows = ~np.all(np.isnan(mat), axis=1)
                display_mat = mat.copy()

                im = ax.imshow(display_mat, aspect="auto", cmap="YlOrRd",
                               vmin=0.0, vmax=1.0, interpolation="nearest")

                ax.set_xticks(range(N_DS))
                ax.set_xticklabels(ds_labels, fontsize=6.5, rotation=35, ha="right")
                ax.set_yticks(range(N_DS))

                # Show wrong_n next to row labels for the wrong-only panel
                if col == 1:
                    ylabels = [f"{DS_NAME[k_]} (n={wrong_n[i]})"
                               for i, k_ in enumerate(ANALYSIS_KEYS)]
                else:
                    ylabels = ds_labels
                ax.set_yticklabels(ylabels, fontsize=6.5)

                ax.set_title(title, fontsize=8.5)

                for ri in range(N_DS):
                    for ci in range(N_DS):
                        v = display_mat[ri, ci]
                        if not np.isnan(v):
                            tc = "white" if v > 0.55 else "black"
                            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                                    fontsize=6, color=tc)
                        else:
                            ax.text(ci, ri, "—", ha="center", va="center",
                                    fontsize=6, color="#aaaaaa")

                # Highlight the diagonal (true-class column)
                for d in range(N_DS):
                    ax.add_patch(plt.Rectangle(
                        (d - 0.5, d - 0.5), 1, 1,
                        fill=False, edgecolor="blue", linewidth=1.5, zorder=5
                    ))

                label = "fraction of k-NN" if col == 0 else "fraction predicted as"
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            p = save_dir / f"knn_confusion_k{k}.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  Plot saved: {p}")
        else:
            plt.show()


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(records: List[Dict], k_values: List[int]) -> None:
    col_w = 9

    # ── Aggregate summary ──
    header_parts = (["Layer", "N"] +
                    [f"pur_b_k{k}" for k in k_values] +
                    [f"acc_b_k{k}" for k in k_values] + ["sil_b"] +
                    [f"pur_f_k{k}" for k in k_values] +
                    [f"acc_f_k{k}" for k in k_values] + ["sil_f"])
    header = "  ".join(f"{h:>{col_w}}" for h in header_parts)
    sep    = "─" * len(header)

    print(f"\n{sep}")
    print("k-NN LOCALITY — AGGREGATE  (b=binary phys/bio, f=fine 8 classes)")
    print(sep); print(header); print(sep)

    for r in records:
        def _fv(key):
            v = r.get(key, float("nan"))
            return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "   nan"
        row = ([f"{r['layer']:>{col_w}}", f"{r.get('n_points',0):>{col_w}}"] +
               [_fv(f"purity_binary_k{k}")   for k in k_values] +
               [_fv(f"accuracy_binary_k{k}") for k in k_values] +
               [_fv("silhouette_binary")] +
               [_fv(f"purity_fine_k{k}")     for k in k_values] +
               [_fv(f"accuracy_fine_k{k}")   for k in k_values] +
               [_fv("silhouette_fine")])
        print("  ".join(f"{v:>{col_w}}" for v in row))
    print(sep)

    # ── Per-dataset purity ──
    for k in k_values:
        print(f"\nPER-DATASET PURITY  k={k}")
        ds_w = max(len(DS_NAME[dk]) for dk in ANALYSIS_KEYS) + 1
        hdr  = f"  {'Dataset':<{ds_w}}  {'binary':>8}  {'fine':>8}"
        print("─" * len(hdr)); print(hdr); print("─" * len(hdr))
        for r in records:
            print(f"  Layer {r['layer']}")
            for dk in ANALYSIS_KEYS:
                bp = r.get(f"per_ds_binary_{dk}_k{k}", float("nan"))
                fp = r.get(f"per_ds_fine_{dk}_k{k}",   float("nan"))
                bp_s = f"{bp:.4f}" if not np.isnan(bp) else "   nan"
                fp_s = f"{fp:.4f}" if not np.isnan(fp) else "   nan"
                print(f"    {DS_NAME[dk]:<{ds_w}}  {bp_s:>8}  {fp_s:>8}")
        print("─" * len(hdr))

    # ── Wrong-prediction confusion ──
    ds_labels = [DS_NAME[k] for k in ANALYSIS_KEYS]
    ds_w      = max(len(l) for l in ds_labels) + 1
    col_w2    = 7

    for k in k_values:
        print(f"\nWRONG-PREDICTION CONFUSION  k={k}")
        print("  Row = true dataset.  Values = fraction of wrong points predicted as each column class.")
        print("  Each row sums to 1.0. Only rows with wrong predictions shown.\n")
        col_header = "  ".join(f"{l[:col_w2]:>{col_w2}}" for l in ds_labels)
        hdr = f"  {'True dataset':<{ds_w}}  {'n_wrong':>7}  {col_header}"
        sep2 = "─" * len(hdr)

        for r in records:
            wrong_c = r.get(f"conf_wrong_k{k}", None)
            wrong_n = r.get(f"conf_wrong_n_k{k}", np.zeros(N_DS, dtype=int))
            if wrong_c is None:
                continue
            print(f"  Layer {r['layer']}")
            print(sep2); print(hdr); print(sep2)
            for di, dk in enumerate(ANALYSIS_KEYS):
                n_wrong = wrong_n[di]
                if n_wrong == 0:
                    continue
                row_vals = wrong_c[di]
                vals_str = "  ".join(
                    f"{v:>{col_w2}.3f}" if not np.isnan(v) else f"{'—':>{col_w2}}"
                    for v in row_vals
                )
                print(f"  {DS_NAME[dk]:<{ds_w}}  {n_wrong:>7}  {vals_str}")
            print(sep2 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config",    type=Path, required=True)
    ap.add_argument("--layers",    type=int, nargs="+", required=True)
    ap.add_argument("--k",         type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--n",         type=int, default=100)
    ap.add_argument("--save",      type=Path, default=None)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--dry_run",   action="store_true")
    ap.add_argument("--gpqa_phys", default="data/eval/gpqa_main_physics_sweep.csv")
    ap.add_argument("--gpqa_bio",  default="data/eval/gpqa_main_biology_sweep.csv")
    ap.add_argument("--mmlu_phys", default="data/eval/mmlu_physics_sweep.csv")
    ap.add_argument("--mmlu_bio",  default="data/eval/mmlu_biology_sweep.csv")
    ap.add_argument("--text_phys", default="data/prompts/physics_pos.txt")
    ap.add_argument("--text_bio",  default="data/prompts/biology_neg.txt")
    ap.add_argument("--arxiv_phys",default="data/eval/arxiv_physics.txt")
    ap.add_argument("--arxiv_bio", default="data/eval/biorxiv_biology.txt")
    ap.add_argument("--sublayer",  default=None)
    args = ap.parse_args()

    k_values = sorted(set(args.k))
    print(f"k values : {k_values}")
    print(f"Layers   : {args.layers}")
    print(f"n/dataset: {args.n}")

    # ── Config ──
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    model_raw = raw["model"]
    sweep_raw = raw.get("sweep", {})
    layer_pat = sweep_raw.get("layer_name_pattern", "model.layers.{layer_idx}")
    tok_pos   = sweep_raw.get("token_position", "last")
    cap_bs    = sweep_raw.get("capture_batch_size", 16)
    sublayer  = args.sublayer or sweep_raw.get("sublayer") or None
    if sublayer:
        layer_pat = f"{layer_pat}.{sublayer}"
    layer_names = [layer_pat.format(layer_idx=i) for i in args.layers]

    # ── Load prompts ──
    ds_path_args = {
        "gpqa_phys":  args.gpqa_phys,  "gpqa_bio":  args.gpqa_bio,
        "mmlu_phys":  args.mmlu_phys,  "mmlu_bio":  args.mmlu_bio,
        "text_phys":  args.text_phys,  "text_bio":  args.text_bio,
        "arxiv_phys": args.arxiv_phys, "arxiv_bio": args.arxiv_bio,
    }
    print("\nLoading datasets:")
    ds_prompts: Dict[str, Optional[List[str]]] = {}
    for key in ANALYSIS_KEYS:
        prompts = load_prompts(ds_path_args[key], args.n, args.seed)
        ds_prompts[key] = prompts
        status = f"{len(prompts)} prompts" if prompts else "MISSING"
        print(f"  {key:12s}  {status}")

    # ── Single forward-pass sweep ──
    acts_store: Dict[str, Dict[str, List[torch.Tensor]]] = {
        key: {ln: [] for ln in layer_names} for key in ANALYSIS_KEYS
    }

    if args.dry_run:
        print("\n[dry_run] Using random embeddings (dim=64).")
        rng = np.random.default_rng(args.seed)
        for key in ANALYSIS_KEYS:
            n = args.n
            X = rng.standard_normal((n, 64)).astype(np.float32)
            X[:, 0] += 3.0 if _BINARY_LABEL[key] == 1 else -3.0
            for ln in layer_names:
                acts_store[key][ln] = [torch.from_numpy(X[j]) for j in range(n)]
    else:
        dtype_map  = {"float16": torch.float16,
                      "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype      = dtype_map[model_raw.get("dtype", "bfloat16")]
        model_name = model_raw["name"]
        device     = model_raw.get("device", "cuda")

        print(f"\nLoading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device)
        model.eval()
        mwh = ModelWithHooks(model)

        pooled_prompts: List[str] = []
        pooled_keys:   List[str] = []
        for key in ANALYSIS_KEYS:
            p = ds_prompts.get(key)
            if p:
                pooled_prompts.extend(p)
                pooled_keys.extend([key] * len(p))

        print(f"\nCapturing — single sweep over {len(pooled_prompts)} prompts "
              f"({len(ANALYSIS_KEYS)} datasets pooled)...")
        all_layer_acts = capture_all_layers(
            mwh, tokenizer, pooled_prompts, layer_names,
            token_position=tok_pos, batch_size=cap_bs,
        )

        for ln in layer_names:
            layer_vecs = all_layer_acts.get(ln, [])
            for idx, key in enumerate(pooled_keys):
                if idx < len(layer_vecs):
                    acts_store[key][ln].append(layer_vecs[idx])

    # ── Per-layer metrics ──
    records: List[Dict] = []

    for layer_idx, layer_name in zip(args.layers, layer_names):
        print(f"\nLayer {layer_idx}  ({layer_name})")
        layer_acts = {key: acts_store[key].get(layer_name) or None
                      for key in ANALYSIS_KEYS}

        metrics = analyse_layer(layer_acts, k_values)
        if not metrics:
            print("  [SKIP] No activations."); continue

        metrics["layer"] = layer_idx
        records.append(metrics)

        n   = metrics.get("n_points", 0)
        s_b = metrics.get("silhouette_binary", float("nan"))
        s_f = metrics.get("silhouette_fine",   float("nan"))
        print(f"  N={n}  sil_binary={s_b:.4f}  sil_fine={s_f:.4f}")
        for k in k_values:
            pb = metrics.get(f"purity_binary_k{k}", float("nan"))
            ab = metrics.get(f"accuracy_binary_k{k}", float("nan"))
            pf = metrics.get(f"purity_fine_k{k}", float("nan"))
            af = metrics.get(f"accuracy_fine_k{k}", float("nan"))
            print(f"    k={k:2d}  pur_b={pb:.4f}  acc_b={ab:.4f}"
                  f"  |  pur_f={pf:.4f}  acc_f={af:.4f}")

    if not records:
        print("\nNo records to plot or save."); return

    records.sort(key=lambda r: r["layer"])
    print_table(records, k_values)

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        # Drop numpy arrays before saving CSV (not serialisable)
        flat = []
        for r in records:
            flat.append({k: v for k, v in r.items()
                         if not isinstance(v, np.ndarray)})
        pd.DataFrame(flat).to_csv(args.save / "knn_locality.csv", index=False)
        print(f"  CSV saved: {args.save / 'knn_locality.csv'}")

    plot_aggregate(records, k_values, save_dir=args.save)
    plot_per_dataset(records, k_values, save_dir=args.save)
    plot_confusion(records, k_values, save_dir=args.save)

    print("Done.")


if __name__ == "__main__":
    main()
