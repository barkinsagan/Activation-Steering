"""
Cross-experiment comparison tool.

Usage:
    python analysis/compare.py results/exp_a results/exp_b [results/exp_c ...]
    python analysis/compare.py results/exp_a results/exp_b --metric mean_delta_logprob
    python analysis/compare.py results/exp_a results/exp_b --save plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Data loading
# =============================================================================

def _load_experiment(result_dir: Path) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load combined_summary and combined_results from a result directory.
    Checks both single_token/ and continuation/ subdirs, then root.

    Returns:
        (experiment_id, summary_df, results_df)
    """
    exp_id = result_dir.name

    summary, results = None, None

    # Priority: single_token subdir > continuation subdir > root
    for subdir in ["single_token", "continuation", ""]:
        base = result_dir / subdir if subdir else result_dir

        s_path = base / "combined_summary.csv"
        r_path = base / "combined_results.csv"

        if s_path.exists() and summary is None:
            summary = pd.read_csv(s_path)
            summary["experiment"] = exp_id
            print(f"  [{exp_id}] summary from {s_path.relative_to(result_dir.parent)}: {summary.shape}")

        if r_path.exists() and results is None:
            results = pd.read_csv(r_path)
            results["experiment"] = exp_id
            print(f"  [{exp_id}] results from {r_path.relative_to(result_dir.parent)}: {results.shape}")

        if summary is not None:
            break

    if summary is None:
        print(f"  [{exp_id}] WARNING: no combined_summary.csv found, skipping.")

    return exp_id, summary, results


def load_experiments(result_dirs: List[str]) -> Dict[str, dict]:
    """Load all experiments. Returns dict keyed by experiment_id."""
    experiments = {}
    for d in result_dirs:
        path = Path(d)
        if not path.exists():
            print(f"WARNING: {path} does not exist, skipping.")
            continue
        exp_id, summary, results = _load_experiment(path)
        if summary is not None:
            experiments[exp_id] = {"summary": summary, "results": results, "path": path}
    return experiments


# =============================================================================
# Metric helpers
# =============================================================================

def _detect_metric(summary: pd.DataFrame) -> str:
    """Auto-detect the primary metric column from a summary DataFrame."""
    candidates = [
        "mean_delta_logprob",       # single_token sweep
        "mean_delta_sum_logprob",   # continuation target-only sweep
        "accuracy_first",           # full MCQ sweep
        "accuracy_sum",
    ]
    for c in candidates:
        if c in summary.columns:
            return c
    raise ValueError(f"Could not detect metric column. Columns: {list(summary.columns)}")


# =============================================================================
# Plots
# =============================================================================

def plot_metric_by_layer(experiments: Dict[str, dict],
                         metric: Optional[str] = None,
                         coef: Optional[float] = None,
                         save_dir: Optional[Path] = None) -> plt.Figure:
    """
    Line plot: chosen metric vs layer, one line per experiment.
    If coef is None, uses the best non-zero coefficient per layer per experiment.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_id, data in experiments.items():
        summary = data["summary"]
        m = metric or _detect_metric(summary)

        if m not in summary.columns:
            print(f"  [{exp_id}] metric '{m}' not found, skipping.")
            continue

        if coef is not None:
            sub = summary[summary["coef"] == coef].sort_values("layer")
        else:
            # Best non-zero coef per layer
            steered = summary[summary["coef"] != 0.0]
            best_idx = steered.groupby("layer")[m].idxmax()
            sub = steered.loc[best_idx].sort_values("layer")

        ax.plot(sub["layer"], sub[m], marker="o", linewidth=2, label=exp_id)

    coef_label = f" (coef={coef})" if coef is not None else " (best coef per layer)"
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric or "metric")
    ax.set_title(f"Cross-Experiment Comparison: {metric or 'metric'}{coef_label}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        fname = save_dir / "compare_by_layer.png"
        fig.savefig(fname, dpi=150)
        print(f"Saved: {fname}")

    return fig


def plot_best_layer_bar(experiments: Dict[str, dict],
                        metric: Optional[str] = None,
                        save_dir: Optional[Path] = None) -> plt.Figure:
    """
    Bar chart: best achievable metric (across all layers and coefs) per experiment.
    """
    exp_ids, best_values, best_layers, best_coefs = [], [], [], []

    for exp_id, data in experiments.items():
        summary = data["summary"]
        m = metric or _detect_metric(summary)

        if m not in summary.columns:
            continue

        steered = summary[summary["coef"] != 0.0]
        if steered.empty:
            continue

        best_row = steered.loc[steered[m].idxmax()]
        exp_ids.append(exp_id)
        best_values.append(best_row[m])
        best_layers.append(int(best_row["layer"]))
        best_coefs.append(best_row["coef"])

    fig, ax = plt.subplots(figsize=(max(8, len(exp_ids) * 1.5), 5))
    x = np.arange(len(exp_ids))
    bars = ax.bar(x, best_values, color="steelblue", alpha=0.8)

    # Annotate with layer and coef
    for bar, layer, coef in zip(bars, best_layers, best_coefs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"L{layer}\nc={coef:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=20, ha="right")
    ax.set_ylabel(metric or "best metric")
    ax.set_title("Best Achievable Result per Experiment")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_dir:
        fname = save_dir / "compare_best_bar.png"
        fig.savefig(fname, dpi=150)
        print(f"Saved: {fname}")

    return fig


def plot_heatmap_grid(experiments: Dict[str, dict],
                      metric: Optional[str] = None,
                      save_dir: Optional[Path] = None) -> plt.Figure:
    """
    Grid of heatmaps (one per experiment): layers × coefs, colored by metric.
    """
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 8), squeeze=False)

    for ax, (exp_id, data) in zip(axes[0], experiments.items()):
        summary = data["summary"]
        m = metric or _detect_metric(summary)

        if m not in summary.columns:
            ax.set_title(f"{exp_id}\n(metric not found)")
            continue

        steered = summary[summary["coef"] != 0.0]
        if steered.empty:
            continue

        pivot = steered.pivot(index="layer", columns="coef", values=m).sort_index()
        vmax = np.nanmax(np.abs(pivot.values))
        cmap = "RdBu" if "delta" in m else "RdYlGn"
        vmin = -vmax if "delta" in m else 0

        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns],
                            rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_xlabel("Coef")
        ax.set_ylabel("Layer")
        ax.set_title(exp_id)
        plt.colorbar(im, ax=ax, label=m)

    fig.suptitle(f"Heatmap Comparison: {metric or 'metric'}", fontsize=13)
    plt.tight_layout()

    if save_dir:
        fname = save_dir / "compare_heatmaps.png"
        fig.savefig(fname, dpi=150)
        print(f"Saved: {fname}")

    return fig


def print_summary_table(experiments: Dict[str, dict], metric: Optional[str] = None):
    """Print a simple text table of best results per experiment."""
    rows = []
    for exp_id, data in experiments.items():
        summary = data["summary"]
        m = metric or _detect_metric(summary)

        if m not in summary.columns:
            continue

        steered = summary[summary["coef"] != 0.0]
        baseline = summary[summary["coef"] == 0.0][m].mean() if not summary[summary["coef"] == 0.0].empty else float("nan")

        if steered.empty:
            continue

        best_row = steered.loc[steered[m].idxmax()]
        rows.append({
            "experiment": exp_id,
            "baseline": round(float(baseline), 4),
            "best_value": round(float(best_row[m]), 4),
            "best_layer": int(best_row["layer"]),
            "best_coef": best_row["coef"],
            "metric": m,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results to display.")
        return

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare results across multiple steering experiments."
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="Paths to experiment result directories",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric column to compare (auto-detected if not specified)",
    )
    parser.add_argument(
        "--coef",
        type=float,
        default=None,
        help="Fix a specific coefficient for the by-layer plot (default: best per layer)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Directory to save plots (default: show interactively)",
    )
    args = parser.parse_args()

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {len(args.result_dirs)} experiment(s)...")
    experiments = load_experiments(args.result_dirs)

    if not experiments:
        print("No valid experiments found.", file=sys.stderr)
        sys.exit(1)

    print_summary_table(experiments, metric=args.metric)

    plot_metric_by_layer(experiments, metric=args.metric, coef=args.coef, save_dir=save_dir)
    plot_best_layer_bar(experiments, metric=args.metric, save_dir=save_dir)
    plot_heatmap_grid(experiments, metric=args.metric, save_dir=save_dir)

    if not save_dir:
        plt.show()


if __name__ == "__main__":
    main()
