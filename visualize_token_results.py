"""
Visualization for steering vector layer-sweep experiments.

Expects folder structure:
    complete_sweep/
        layer_0/<timestamp>/detailed_wide.csv, detailed_long.csv, summary.csv
        layer_1/<timestamp>/...
        ...
        layer_31/<timestamp>/...

Colab usage:
    1. Run the LOAD DATA cell first
    2. Then run any plot cell independently
"""

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================================
# CELL 1: LOAD DATA  (run this first in Colab)
# =========================================================================

def load_sweep(sweep_dir: str):
    """
    Load all layer results from a sweep directory into combined DataFrames.

    Handles nested timestamp folders:
        sweep_dir/layer_X/<timestamp>/detailed_wide.csv

    Returns:
        wide_all:    combined detailed_wide with 'layer' column
        long_all:    combined detailed_long with 'layer' column
        summary_all: combined summary with 'layer' column
    """
    sweep_path = Path(sweep_dir)

    wide_frames, long_frames, summary_frames = [], [], []

    for layer_dir in sorted(sweep_path.glob("layer_*")):
        # Extract layer number from folder name
        layer_idx = int(layer_dir.name.split("_")[-1])

        # Handle optional timestamp subdirectory
        wide_candidates = list(layer_dir.rglob("detailed_wide.csv"))
        long_candidates = list(layer_dir.rglob("detailed_long.csv"))
        summary_candidates = list(layer_dir.rglob("summary.csv"))

        if wide_candidates:
            df = pd.read_csv(wide_candidates[0])
            df["layer"] = layer_idx
            wide_frames.append(df)
        if long_candidates:
            df = pd.read_csv(long_candidates[0])
            df["layer"] = layer_idx
            long_frames.append(df)
        if summary_candidates:
            df = pd.read_csv(summary_candidates[0])
            df["layer"] = layer_idx
            summary_frames.append(df)

    wide_all = pd.concat(wide_frames, ignore_index=True) if wide_frames else pd.DataFrame()
    long_all = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

    n_layers = len(wide_frames)
    print(f"Loaded {n_layers} layers from {sweep_path}")
    print(f"  wide_all:    {wide_all.shape}")
    print(f"  long_all:    {long_all.shape}")
    print(f"  summary_all: {summary_all.shape}")

    return wide_all, long_all, summary_all


# Example Colab cell:
# SWEEP_DIR = "/content/complete_sweep"
# wide_all, long_all, summary_all = load_sweep(SWEEP_DIR)


# =========================================================================
# CROSS-LAYER PLOTS  (the main value of a 32-layer sweep)
# =========================================================================

# ---- Plot 1: Accuracy Heatmap (layers x coefficients) -------------------

def plot_accuracy_heatmap(summary_all: pd.DataFrame, metric: str = "first"):
    """
    Heatmap with layers on y-axis, coefficients on x-axis, colored by accuracy.

    Args:
        summary_all: combined summary DataFrame with 'layer' column
        metric: "sum" or "first"
    """
    acc_col = f"accuracy_{metric}"
    pivot = summary_all.pivot(index="layer", columns="coef", values=acc_col)
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title(f"MCQ Accuracy by Layer and Coefficient ({metric} scoring)")
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    return fig


# ---- Plot 2: Accuracy vs Coefficient, one line per layer ----------------

def plot_accuracy_lines_by_layer(summary_all: pd.DataFrame, metric: str = "first",
                                 highlight_layers: list = None):
    """
    Line plot: accuracy vs coefficient, with one line per layer.

    Args:
        highlight_layers: list of layer indices to draw bold; others are faded.
                          If None, all layers shown equally.
    """
    acc_col = f"accuracy_{metric}"
    layers = sorted(summary_all["layer"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for layer in layers:
        sub = summary_all[summary_all["layer"] == layer].sort_values("coef")
        if highlight_layers is not None:
            if layer in highlight_layers:
                ax.plot(sub["coef"], sub[acc_col], marker="o", linewidth=2,
                        label=f"Layer {layer}")
            else:
                ax.plot(sub["coef"], sub[acc_col], color="lightgray",
                        linewidth=0.5, alpha=0.5)
        else:
            ax.plot(sub["coef"], sub[acc_col], marker=".", markersize=3,
                    linewidth=1, alpha=0.6, label=f"L{layer}")

    ax.axhline(0.25, linestyle=":", color="gray", alpha=0.5, label="Random chance")

    baseline = summary_all[summary_all["coef"] == 0.0]
    if not baseline.empty:
        ax.axhline(baseline[acc_col].iloc[0], linestyle="--", color="black",
                    alpha=0.7, label="Baseline")

    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Coefficient per Layer ({metric} scoring)")
    if highlight_layers is not None or len(layers) <= 8:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---- Plot 3: Best coefficient per layer ---------------------------------

def plot_best_coef_per_layer(summary_all: pd.DataFrame, metric: str = "first"):
    """
    Bar chart showing the best (non-zero) coefficient and its accuracy for each layer.
    """
    acc_col = f"accuracy_{metric}"
    steered = summary_all[summary_all["coef"] != 0.0]
    best = steered.loc[steered.groupby("layer")[acc_col].idxmax()]
    best = best.sort_values("layer")

    baseline_acc = summary_all[summary_all["coef"] == 0.0][acc_col].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: best accuracy per layer
    colors = ["green" if a > baseline_acc else "red" for a in best[acc_col]]
    axes[0].bar(best["layer"], best[acc_col], color=colors, alpha=0.7)
    axes[0].axhline(baseline_acc, linestyle="--", color="black", alpha=0.7,
                     label=f"Baseline ({baseline_acc:.3f})")
    axes[0].set_ylabel("Best Accuracy")
    axes[0].set_title(f"Best Steering Result per Layer ({metric} scoring)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Bottom: which coefficient was best
    axes[1].bar(best["layer"], best["coef"], color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Best Coefficient")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ---- Plot 4: Delta heatmap (layers x coefficients) ----------------------

def plot_delta_heatmap(summary_all: pd.DataFrame, metric: str = "first"):
    """
    Heatmap of mean delta target log-prob (steered - baseline) across layers
    and coefficients. Only shows non-zero coefficients.
    """
    delta_col = f"mean_delta_target_{metric}_lp"
    steered = summary_all[summary_all["coef"] != 0.0].copy()

    if delta_col not in steered.columns:
        print(f"Column {delta_col} not found.")
        return None

    pivot = steered.pivot(index="layer", columns="coef", values=delta_col)
    pivot = pivot.sort_index()

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu",
                    vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title(f"Mean Δ Target Log-Prob ({metric} scoring)\nBlue = improved, Red = hurt")
    plt.colorbar(im, ax=ax, label="Δ log-prob")
    plt.tight_layout()
    return fig


# ---- Plot 5: % Improved heatmap (layers x coefficients) -----------------

def plot_pct_improved_heatmap(summary_all: pd.DataFrame, metric: str = "first"):
    """
    Heatmap of % questions improved by steering, across layers and coefficients.
    """
    col = f"pct_improved_{metric}_lp"
    steered = summary_all[summary_all["coef"] != 0.0].copy()

    if col not in steered.columns:
        print(f"Column {col} not found.")
        return None

    pivot = steered.pivot(index="layer", columns="coef", values=col)
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title(f"% Questions with Improved Target Log-Prob ({metric} scoring)")
    plt.colorbar(im, ax=ax, label="Fraction improved")
    plt.tight_layout()
    return fig


# ---- Plot 6: Accuracy change from baseline (bar per layer) --------------

def plot_max_accuracy_change(summary_all: pd.DataFrame, metric: str = "first"):
    """
    For each layer, show the max accuracy gain and max accuracy loss
    compared to baseline (coef=0).
    """
    acc_col = f"accuracy_{metric}"
    baseline = summary_all[summary_all["coef"] == 0.0].set_index("layer")[acc_col]
    steered = summary_all[summary_all["coef"] != 0.0].copy()

    layers = sorted(steered["layer"].unique())
    max_gains, max_losses = [], []

    for layer in layers:
        sub = steered[steered["layer"] == layer]
        base_acc = baseline.get(layer, 0.0)
        deltas = sub[acc_col] - base_acc
        max_gains.append(deltas.max())
        max_losses.append(deltas.min())

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(layers))
    ax.bar(x, max_gains, color="green", alpha=0.7, label="Max accuracy gain")
    ax.bar(x, max_losses, color="red", alpha=0.7, label="Max accuracy loss")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy Change from Baseline")
    ax.set_title(f"Max Accuracy Gain/Loss per Layer ({metric} scoring)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# =========================================================================
# SINGLE-LAYER PLOTS  (filter wide_all / summary_all to one layer)
# =========================================================================

# ---- Plot 7: Accuracy vs Coefficient (single layer) ---------------------

def plot_accuracy_vs_coef(summary_all: pd.DataFrame, layer: int,
                          mode: str = "both"):
    """
    Line plot of MCQ accuracy vs coefficient for a single layer.

    Args:
        layer: which layer to plot
        mode: "sum", "first", or "both"
    """
    summary = summary_all[summary_all["layer"] == layer].sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, 5))

    if mode in ("sum", "both"):
        ax.plot(summary["coef"], summary["accuracy_sum"],
                marker="o", label="Sum log-prob scoring")
    if mode in ("first", "both"):
        ax.plot(summary["coef"], summary["accuracy_first"],
                marker="s", label="First-token scoring")

    ax.axhline(0.25, linestyle=":", color="gray", alpha=0.5, label="Random chance")
    baseline = summary[summary["coef"] == 0.0]
    if not baseline.empty:
        ax.axhline(baseline["accuracy_sum"].iloc[0], linestyle="--",
                    color="gray", alpha=0.7, label="Baseline")

    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Layer {layer}: MCQ Accuracy vs Coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---- Plot 8: Target probability vs Coefficient (single layer) -----------

def plot_target_prob_vs_coef(summary_all: pd.DataFrame, layer: int):
    """Mean target probability vs coefficient for a single layer."""
    summary = summary_all[summary_all["layer"] == layer].sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["coef"], summary["mean_target_prob_sum"],
            marker="o", label="Sum log-prob scoring")
    ax.plot(summary["coef"], summary["mean_target_prob_first"],
            marker="s", label="First-token scoring")

    ax.axhline(0.25, linestyle=":", color="gray", alpha=0.5, label="Random chance")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Mean Target Probability")
    ax.set_title(f"Layer {layer}: Mean Target Probability vs Coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---- Plot 9: Delta distribution boxplot (single layer) ------------------

def plot_delta_distribution(wide_all: pd.DataFrame, layer: int,
                            metric: str = "sum"):
    """
    Box plot of per-question target log-prob deltas at each coefficient.

    Args:
        layer: which layer to plot
        metric: "sum" or "first"
    """
    col = f"delta_target_{metric}_lp"
    wide = wide_all[(wide_all["layer"] == layer) & (wide_all["coef"] != 0.0)]

    if wide.empty or col not in wide.columns:
        print(f"No data for layer {layer} or column {col} not found.")
        return None

    coefs = sorted(wide["coef"].unique())
    data = [wide[wide["coef"] == c][col].dropna().values for c in coefs]

    fig, ax = plt.subplots(figsize=(max(8, len(coefs) * 1.2), 5))
    bp = ax.boxplot(data, labels=[f"{c:.1f}" for c in coefs],
                    patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.axhline(0.0, linestyle="--", color="red", alpha=0.7, label="No change")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel(f"Δ Target Log-Prob ({metric})")
    ax.set_title(f"Layer {layer}: Per-Question Target Log-Prob Shift ({metric})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---- Plot 10: Rank distribution stacked bar (single layer) ---------------

def plot_rank_distribution(wide_all: pd.DataFrame, layer: int,
                           metric: str = "sum"):
    """
    Stacked bar: fraction of questions at target rank 1/2/3/4 per coefficient.
    """
    rank_col = f"target_rank_{metric}"
    wide = wide_all[wide_all["layer"] == layer]
    coefs = sorted(wide["coef"].unique())

    rank_counts = {}
    for coef in coefs:
        subset = wide[wide["coef"] == coef]
        counts = subset[rank_col].value_counts().reindex([1, 2, 3, 4], fill_value=0)
        rank_counts[coef] = counts / max(len(subset), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(coefs) * 1.2), 5))
    x = np.arange(len(coefs))
    labels = [f"{c:.1f}" for c in coefs]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    bottom = np.zeros(len(coefs))
    for rank in [1, 2, 3, 4]:
        vals = [rank_counts[c][rank] for c in coefs]
        ax.bar(x, vals, bottom=bottom, label=f"Rank {rank}",
               color=colors[rank - 1], alpha=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title(f"Layer {layer}: Target Rank Distribution ({metric})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---- Plot 11: Per-question heatmap (single layer) -----------------------

def plot_question_heatmap(wide_all: pd.DataFrame, layer: int,
                          metric: str = "sum", max_questions: int = 50):
    """
    Heatmap: questions x coefficients, colored by target probability.
    """
    prob_col = f"target_prob_{metric}"
    wide = wide_all[wide_all["layer"] == layer]

    question_ids = sorted(wide["question_id"].unique())[:max_questions]
    subset = wide[wide["question_id"].isin(question_ids)]
    pivot = subset.pivot(index="question_id", columns="coef", values=prob_col)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                     max(6, len(question_ids) * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Question ID")
    ax.set_title(f"Layer {layer}: Target Probability per Question ({metric})")
    plt.colorbar(im, ax=ax, label="P(target)")
    plt.tight_layout()
    return fig


# ---- Plot 12: Sum vs first-token delta scatter (single layer) -----------

def plot_sum_vs_first_scatter(wide_all: pd.DataFrame, layer: int):
    """
    Scatter: delta_target_sum_lp vs delta_target_first_lp, colored by coefficient.
    """
    steered = wide_all[(wide_all["layer"] == layer) & (wide_all["coef"] != 0.0)]
    if steered.empty:
        print(f"No steered data for layer {layer}.")
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        steered["delta_target_sum_lp"],
        steered["delta_target_first_lp"],
        c=steered["coef"], cmap="coolwarm", alpha=0.6, edgecolors="none", s=20,
    )
    plt.colorbar(scatter, ax=ax, label="Steering Coefficient")

    ax.axhline(0, linestyle=":", color="gray", alpha=0.4)
    ax.axvline(0, linestyle=":", color="gray", alpha=0.4)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")

    ax.set_xlabel("Δ Target Sum Log-Prob")
    ax.set_ylabel("Δ Target First-Token Log-Prob")
    ax.set_title(f"Layer {layer}: Sum vs First-Token Scoring Agreement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    return fig


# ---- Plot 13: Improved / Hurt stacked bar (single layer) ----------------

def plot_improved_hurt(summary_all: pd.DataFrame, layer: int,
                       metric: str = "sum"):
    """
    Stacked bar: fraction of questions improved / unchanged / hurt per coefficient.
    """
    pct_imp_col = f"pct_improved_{metric}_lp"
    pct_hurt_col = f"pct_hurt_{metric}_lp"

    steered = summary_all[(summary_all["layer"] == layer) &
                           (summary_all["coef"] != 0.0)].sort_values("coef")

    if pct_imp_col not in steered.columns:
        print(f"Column {pct_imp_col} not found.")
        return None

    improved = steered[pct_imp_col].values
    hurt = steered[pct_hurt_col].values
    unchanged = 1.0 - improved - hurt

    labels = [f"{c:.1f}" for c in steered["coef"]]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    x = np.arange(len(labels))

    ax.bar(x, improved, label="Improved", color="green", alpha=0.7)
    ax.bar(x, unchanged, bottom=improved, label="Unchanged", color="gray", alpha=0.5)
    ax.bar(x, hurt, bottom=improved + unchanged, label="Hurt", color="red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title(f"Layer {layer}: Steering Impact ({metric} scoring)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---- Plot 14: Margin vs Coefficient (single layer) ----------------------

def plot_margin_vs_coef(summary_all: pd.DataFrame, layer: int):
    """Confidence margin (target prob - best wrong prob) vs coefficient."""
    summary = summary_all[summary_all["layer"] == layer].sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["coef"], summary["mean_margin_sum"],
            marker="o", label="Sum log-prob scoring")
    ax.plot(summary["coef"], summary["mean_margin_first"],
            marker="s", label="First-token scoring")

    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Mean Margin (target − best wrong)")
    ax.set_title(f"Layer {layer}: Confidence Margin vs Coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =========================================================================
# VALIDATION / TEST SPLIT & GENERALIZATION CHECK
# =========================================================================

def split_val_test(wide_all: pd.DataFrame, test_frac: float = 0.15,
                   seed: int = 42):
    """
    Randomly split question IDs into validation (85%) and test (15%) sets.

    Returns:
        val_ids: set of question IDs for validation
        test_ids: set of question IDs for test
    """
    question_ids = np.array(sorted(wide_all["question_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(question_ids)

    n_test = max(1, int(len(question_ids) * test_frac))
    test_ids = set(question_ids[:n_test])
    val_ids = set(question_ids[n_test:])

    print(f"Split: {len(val_ids)} validation, {len(test_ids)} test "
          f"({len(test_ids)/len(question_ids)*100:.1f}% test)")
    return val_ids, test_ids


def find_best_coef_per_layer(wide_all: pd.DataFrame, val_ids: set,
                             metric: str = "first"):
    """
    For each layer, find the steering coefficient that maximises accuracy
    on the validation set.

    Returns:
        dict  {layer: {"best_coef": float,
                        "val_accuracy": float,
                        "val_baseline": float}}
    """
    acc_col = f"correct_{metric}"
    val_df = wide_all[wide_all["question_id"].isin(val_ids)]

    best = {}
    for layer in sorted(val_df["layer"].unique()):
        layer_df = val_df[val_df["layer"] == layer]
        acc_per_coef = layer_df.groupby("coef")[acc_col].mean()
        best_coef = acc_per_coef.idxmax()
        best[layer] = {
            "best_coef": best_coef,
            "val_accuracy": acc_per_coef[best_coef],
            "val_baseline": acc_per_coef.get(0.0, np.nan),
        }
    return best


def evaluate_on_test(wide_all: pd.DataFrame, test_ids: set,
                     best_per_layer: dict, metric: str = "first"):
    """
    Evaluate the validation-selected best coefficient on the held-out test set.

    Returns:
        pd.DataFrame with one row per layer and columns:
            layer, best_coef,
            val_accuracy, val_baseline, val_improvement,
            test_accuracy, test_baseline, test_improvement
    """
    acc_col = f"correct_{metric}"
    test_df = wide_all[wide_all["question_id"].isin(test_ids)]

    rows = []
    for layer in sorted(best_per_layer.keys()):
        info = best_per_layer[layer]
        best_coef = info["best_coef"]
        layer_test = test_df[test_df["layer"] == layer]

        test_at_best = layer_test[layer_test["coef"] == best_coef]
        test_acc = test_at_best[acc_col].mean() if not test_at_best.empty else np.nan

        test_baseline = layer_test[layer_test["coef"] == 0.0]
        test_base = test_baseline[acc_col].mean() if not test_baseline.empty else np.nan

        rows.append({
            "layer": layer,
            "best_coef": best_coef,
            "val_accuracy": info["val_accuracy"],
            "val_baseline": info["val_baseline"],
            "val_improvement": info["val_accuracy"] - info["val_baseline"],
            "test_accuracy": test_acc,
            "test_baseline": test_base,
            "test_improvement": test_acc - test_base,
        })

    return pd.DataFrame(rows)


# ---- Plot 15: Val vs Test accuracy at best coefficient per layer ---------

def plot_val_test_accuracy(results_df: pd.DataFrame, metric: str = "first"):
    """
    Grouped bar chart comparing validation-selected accuracy with test accuracy
    and baseline for every layer.
    """
    layers = results_df["layer"].values
    x = np.arange(len(layers))
    w = 0.25

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # --- Top: absolute accuracies ---
    axes[0].bar(x - w, results_df["val_accuracy"], w,
                label="Val (best coef)", color="steelblue")
    axes[0].bar(x, results_df["test_accuracy"], w,
                label="Test (best coef)", color="darkorange")
    axes[0].bar(x + w, results_df["test_baseline"], w,
                label="Baseline (coef=0)", color="gray", alpha=0.6)
    axes[0].axhline(0.25, ls=":", color="black", alpha=0.4, label="Random chance")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Val-Selected Best Coefficient: Generalization to Test "
                      f"({metric} scoring)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Middle: improvement over baseline ---
    axes[1].bar(x - 0.15, results_df["val_improvement"], 0.3,
                label="Val improvement", color="steelblue", alpha=0.7)
    axes[1].bar(x + 0.15, results_df["test_improvement"], 0.3,
                label="Test improvement", color="darkorange", alpha=0.7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Accuracy \u0394 from baseline")
    axes[1].set_title("Improvement over Baseline per Layer")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    # --- Bottom: which coefficient was selected ---
    axes[2].bar(x, results_df["best_coef"], color="teal", alpha=0.7)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Best Coefficient")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(layers)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ---- Plot 16: Val improvement vs Test improvement scatter ----------------

def plot_val_test_scatter(results_df: pd.DataFrame, metric: str = "first"):
    """
    Scatter of val improvement vs test improvement per layer.
    Points above y=x line generalise better than expected.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(results_df["val_improvement"], results_df["test_improvement"],
               c=results_df["layer"], cmap="viridis", s=60, edgecolors="black",
               linewidths=0.5, zorder=3)

    for _, row in results_df.iterrows():
        ax.annotate(f"L{int(row['layer'])}", (row["val_improvement"],
                    row["test_improvement"]), fontsize=7, alpha=0.7,
                    textcoords="offset points", xytext=(4, 4))

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
    ax.axhline(0, ls=":", color="gray", alpha=0.4)
    ax.axvline(0, ls=":", color="gray", alpha=0.4)

    ax.set_xlabel("Validation Improvement")
    ax.set_ylabel("Test Improvement")
    ax.set_title(f"Generalization: Val vs Test Improvement ({metric} scoring)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    return fig


# ---- Convenience runner -------------------------------------------------

def run_val_test_analysis(wide_all: pd.DataFrame, metric: str = "first",
                          test_frac: float = 0.15, seed: int = 42):
    """
    End-to-end: split -> find best coefs on val -> evaluate on test -> plot.

    Returns:
        results_df, fig_bars, fig_scatter
    """
    val_ids, test_ids = split_val_test(wide_all, test_frac=test_frac, seed=seed)
    best_per_layer = find_best_coef_per_layer(wide_all, val_ids, metric=metric)
    results_df = evaluate_on_test(wide_all, test_ids, best_per_layer,
                                  metric=metric)

    print("\n" + "=" * 70)
    print("VALIDATION / TEST RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    n_gen = (results_df["test_improvement"] > 0).sum()
    print(f"\nLayers where test improved: {n_gen}/{len(results_df)}")

    fig_bars = plot_val_test_accuracy(results_df, metric=metric)
    fig_scatter = plot_val_test_scatter(results_df, metric=metric)

    return results_df, fig_bars, fig_scatter


# =========================================================================
# K-FOLD CROSS-VALIDATION
# =========================================================================

def kfold_val_test(wide_all: pd.DataFrame, k: int = 5, metric: str = "first",
                   seed: int = 42):
    """
    K-Fold cross-validated coefficient selection.

    For each fold the held-out questions are the test set and the remaining
    questions are used to pick the best coefficient per layer.  Results are
    averaged across all K folds for a stable estimate.

    Args:
        wide_all: combined detailed_wide DataFrame with 'layer' column
        k: number of folds
        metric: "sum" or "first"
        seed: random seed for shuffling

    Returns:
        fold_results: list of per-fold DataFrames (same schema as
                      evaluate_on_test output)
        avg_results:  DataFrame averaged over folds, one row per layer
    """
    acc_col = f"correct_{metric}"
    question_ids = np.array(sorted(wide_all["question_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(question_ids)

    folds = np.array_split(question_ids, k)
    fold_results = []

    for fold_idx in range(k):
        test_ids = set(folds[fold_idx])
        val_ids = set(qid for i, fold in enumerate(folds)
                      if i != fold_idx for qid in fold)

        print(f"\n--- Fold {fold_idx + 1}/{k}  "
              f"(val={len(val_ids)}, test={len(test_ids)}) ---")

        best_per_layer = find_best_coef_per_layer(wide_all, val_ids,
                                                  metric=metric)
        fold_df = evaluate_on_test(wide_all, test_ids, best_per_layer,
                                   metric=metric)
        fold_df["fold"] = fold_idx
        fold_results.append(fold_df)

    all_folds = pd.concat(fold_results, ignore_index=True)

    # Average across folds per layer
    avg_results = (
        all_folds
        .groupby("layer")
        .agg(
            best_coef_mean=("best_coef", "mean"),
            best_coef_std=("best_coef", "std"),
            val_accuracy=("val_accuracy", "mean"),
            val_baseline=("val_baseline", "mean"),
            val_improvement=("val_improvement", "mean"),
            test_accuracy=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_baseline=("test_baseline", "mean"),
            test_improvement=("test_improvement", "mean"),
            test_improvement_std=("test_improvement", "std"),
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print(f"{k}-FOLD CROSS-VALIDATED RESULTS")
    print("=" * 70)
    print(avg_results.to_string(index=False))

    n_gen = (avg_results["test_improvement"] > 0).sum()
    print(f"\nLayers where mean test improved: {n_gen}/{len(avg_results)}")

    return fold_results, avg_results


# ---- Plot 17: K-Fold accuracy bars per layer -----------------------------

def plot_kfold_accuracy(avg_results: pd.DataFrame, metric: str = "first"):
    """
    Grouped bar chart of mean val / test / baseline accuracy per layer,
    with error bars from cross-validation std.
    """
    layers = avg_results["layer"].values
    x = np.arange(len(layers))
    w = 0.25

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # --- Top: absolute accuracies with error bars ---
    axes[0].bar(x - w, avg_results["val_accuracy"], w,
                label="Val (best coef)", color="steelblue")
    axes[0].bar(x, avg_results["test_accuracy"], w,
                yerr=avg_results["test_accuracy_std"],
                capsize=3, label="Test (best coef)", color="darkorange")
    axes[0].bar(x + w, avg_results["test_baseline"], w,
                label="Baseline (coef=0)", color="gray", alpha=0.6)
    axes[0].axhline(0.25, ls=":", color="black", alpha=0.4, label="Random chance")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"K-Fold CV: Val-Selected Best Coef Generalization "
                      f"({metric} scoring)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Middle: improvement over baseline with error bars ---
    axes[1].bar(x - 0.15, avg_results["val_improvement"], 0.3,
                label="Val improvement", color="steelblue", alpha=0.7)
    axes[1].bar(x + 0.15, avg_results["test_improvement"], 0.3,
                yerr=avg_results["test_improvement_std"],
                capsize=3, label="Test improvement", color="darkorange", alpha=0.7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Accuracy \u0394 from baseline")
    axes[1].set_title("Mean Improvement over Baseline per Layer")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    # --- Bottom: mean best coefficient with std ---
    axes[2].bar(x, avg_results["best_coef_mean"], color="teal", alpha=0.7,
                yerr=avg_results["best_coef_std"], capsize=3)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Best Coefficient (mean)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(layers)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ---- Plot 18: K-Fold val vs test improvement scatter ---------------------

def plot_kfold_scatter(avg_results: pd.DataFrame, metric: str = "first"):
    """
    Scatter of mean val improvement vs mean test improvement per layer,
    with horizontal error bars (test_improvement_std).
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.errorbar(avg_results["val_improvement"],
                avg_results["test_improvement"],
                yerr=avg_results["test_improvement_std"],
                fmt="none", ecolor="gray", alpha=0.4, zorder=1)

    sc = ax.scatter(avg_results["val_improvement"],
                    avg_results["test_improvement"],
                    c=avg_results["layer"], cmap="viridis", s=60,
                    edgecolors="black", linewidths=0.5, zorder=3)

    for _, row in avg_results.iterrows():
        ax.annotate(f"L{int(row['layer'])}",
                    (row["val_improvement"], row["test_improvement"]),
                    fontsize=7, alpha=0.7,
                    textcoords="offset points", xytext=(4, 4))

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
    ax.axhline(0, ls=":", color="gray", alpha=0.4)
    ax.axvline(0, ls=":", color="gray", alpha=0.4)

    ax.set_xlabel("Mean Validation Improvement")
    ax.set_ylabel("Mean Test Improvement")
    ax.set_title(f"K-Fold Generalization: Val vs Test ({metric} scoring)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    return fig


# ---- Convenience runner for K-Fold --------------------------------------

def run_kfold_analysis(wide_all: pd.DataFrame, k: int = 5,
                       metric: str = "first", seed: int = 42):
    """
    End-to-end K-Fold: split -> select coefs on val -> evaluate on test
    -> average across folds -> plot.

    Returns:
        fold_results, avg_results, fig_bars, fig_scatter
    """
    fold_results, avg_results = kfold_val_test(wide_all, k=k, metric=metric,
                                               seed=seed)
    fig_bars = plot_kfold_accuracy(avg_results, metric=metric)
    fig_scatter = plot_kfold_scatter(avg_results, metric=metric)

    return fold_results, avg_results, fig_bars, fig_scatter
