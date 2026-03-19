"""
Visualization for Mehmet's steering vector experiment results.

Expects folder structure:
    cs2ml/layer_X/main_output/model_layers_X_mlp_*.csv

Usage (Colab):
    from mehmet_exp import load_sweep, plot_all
    df = load_sweep("/content/cs2ml")
    plot_all(df)
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_sweep(sweep_dir: str) -> pd.DataFrame:
    """Load all layer CSVs from cs2ml/layer_X/main_output/ into one DataFrame."""
    sweep_path = Path(sweep_dir)
    frames = []

    for csv_path in sorted(sweep_path.glob("layer_*/main_output/model_layers_*_mlp_*.csv")):
        df = pd.read_csv(csv_path)
        df["prob_diffs"] = df["prob_diffs"].apply(ast.literal_eval)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["layer"] = combined["target_layer"].str.extract(r"layers\.(\d+)").astype(int)
    print(f"Loaded {combined['layer'].nunique()} layers, {len(combined)} rows")
    return combined


# -- Plots -----------------------------------------------------------------

def plot_accuracy_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index="layer", columns="coefficient",
                           values="steered_accuracy").sort_index()
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title("Steered Accuracy (%)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_accuracy_change_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index="layer", columns="coefficient",
                           values="accuracy_change").sort_index()
    vmax = np.abs(pivot.values).max()
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title("Accuracy Change (pp) — Blue=improved, Red=hurt")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_accuracy_lines(df: pd.DataFrame, highlight_layers=None):
    baseline = df["baseline_accuracy"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer].sort_values("coefficient")
        if highlight_layers is not None:
            if layer in highlight_layers:
                ax.plot(sub["coefficient"], sub["steered_accuracy"],
                        "o-", lw=2, label=f"Layer {layer}")
            else:
                ax.plot(sub["coefficient"], sub["steered_accuracy"],
                        color="lightgray", lw=0.5, alpha=0.5)
        else:
            ax.plot(sub["coefficient"], sub["steered_accuracy"],
                    ".-", lw=1, alpha=0.6, label=f"L{layer}")
    ax.axhline(baseline, ls="--", color="black", alpha=0.7, label="Baseline")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Coefficient per Layer")
    if highlight_layers or df["layer"].nunique() <= 8:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_best_per_layer(df: pd.DataFrame):
    baseline = df["baseline_accuracy"].iloc[0]
    best = df.loc[df.groupby("layer")["steered_accuracy"].idxmax()].sort_values("layer")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = ["green" if a > baseline else "red" for a in best["steered_accuracy"]]
    axes[0].bar(best["layer"], best["steered_accuracy"], color=colors, alpha=0.7)
    axes[0].axhline(baseline, ls="--", color="black", alpha=0.7, label=f"Baseline ({baseline:.1f}%)")
    axes[0].set_ylabel("Best Accuracy (%)")
    axes[0].set_title("Best Steering Result per Layer")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(best["layer"], best["coefficient"], color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Best Coefficient")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_max_gain_loss(df: pd.DataFrame):
    layers = sorted(df["layer"].unique())
    gains = [df[df["layer"] == l]["accuracy_change"].max() for l in layers]
    losses = [df[df["layer"] == l]["accuracy_change"].min() for l in layers]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(layers))
    ax.bar(x, gains, color="green", alpha=0.7, label="Max gain")
    ax.bar(x, losses, color="red", alpha=0.7, label="Max loss")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy Change (pp)")
    ax.set_title("Max Accuracy Gain/Loss per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_pct_improved_heatmap(df: pd.DataFrame):
    pct = df.apply(lambda r: sum(d > 1e-6 for d in r["prob_diffs"]) / len(r["prob_diffs"]), axis=1)
    tmp = df[["layer", "coefficient"]].copy()
    tmp["pct_improved"] = pct
    pivot = tmp.pivot_table(index="layer", columns="coefficient",
                            values="pct_improved").sort_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title("Fraction of Questions Improved")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_metric_heatmaps(df: pd.DataFrame):
    metrics = [
        ("avg_prob_change", "Prob Change", "RdBu"),
        ("avg_entropy_change", "Entropy Change", "RdBu_r"),
        ("avg_margin_change", "Margin Change", "RdBu"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, (col, title, cmap) in zip(axes, metrics):
        pivot = df.pivot_table(index="layer", columns="coefficient",
                               values=col).sort_index()
        vmax = np.abs(pivot.values).max()
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Metric Changes Across Layers", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def print_summary(df: pd.DataFrame):
    layers = sorted(df["layer"].unique())
    baseline = df["baseline_accuracy"].iloc[0]
    best = df.loc[df["steered_accuracy"].idxmax()]

    print(f"Layers: {len(layers)} ({min(layers)}-{max(layers)})")
    print(f"Coefficients: {sorted(df['coefficient'].unique())}")
    print(f"Baseline accuracy: {baseline:.2f}%")
    print(f"Best: Layer {int(best['layer'])}, coef={int(best['coefficient'])}, "
          f"acc={best['steered_accuracy']:.2f}% ({best['accuracy_change']:+.2f}pp)")

    improved = sum(1 for l in layers if df[df["layer"] == l]["accuracy_change"].max() > 0)
    print(f"Layers with improvement: {improved}/{len(layers)}")


def plot_all(df: pd.DataFrame):
    """Generate all plots."""
    print_summary(df)
    plot_accuracy_heatmap(df)
    plot_accuracy_change_heatmap(df)
    plot_accuracy_lines(df)
    plot_best_per_layer(df)
    plot_max_gain_loss(df)
    plot_pct_improved_heatmap(df)
    plot_metric_heatmaps(df)
    plt.show()
