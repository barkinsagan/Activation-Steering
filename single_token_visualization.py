"""
Visualization for single-token steering vector layer-sweep experiments.

Expects output from sweep_layers_single_token():
    out_dir/combined_results.csv   — one row per (layer, question, coef)
    out_dir/combined_summary.csv   — one row per (layer, coef)
    out_dir/layer_N_results.csv    — fallback if combined not present

Results columns:
    layer, question_id, coef,
    prompt, target_text, first_token_id, first_token_str,
    logprob, rank, delta_logprob, rank_change

Summary columns:
    layer, coef, n, mean_logprob, std_logprob, mean_rank, median_rank
    (coef != 0 only): mean_delta_logprob, std_delta_logprob,
                      pct_improved_logprob, pct_hurt_logprob,
                      mean_rank_change, pct_improved_rank, pct_hurt_rank

Colab usage:
    1. Run the LOAD DATA cell first
    2. Then run any plot cell independently
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================================
# CELL 1: LOAD DATA  (run this first)
# =========================================================================

def load_results(out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load results and summary from a single_token sweep output directory.

    Tries combined_results.csv / combined_summary.csv first, then falls
    back to assembling from per-layer layer_N_results.csv files.

    Returns:
        results: full record DataFrame (one row per layer/question/coef)
        summary: aggregated DataFrame  (one row per layer/coef)
    """
    path = Path(out_dir)

    # --- results ---
    combined_path = path / "combined_results.csv"
    if combined_path.exists():
        results = pd.read_csv(combined_path)
        print(f"Loaded combined_results.csv  {results.shape}")
    else:
        frames = [pd.read_csv(f) for f in sorted(path.glob("layer_*_results.csv"))]
        if not frames:
            raise FileNotFoundError(f"No results files found in {path}")
        results = pd.concat(frames, ignore_index=True)
        print(f"Assembled {len(frames)} per-layer files → {results.shape}")

    # --- summary ---
    summary_path = path / "combined_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        print(f"Loaded combined_summary.csv  {summary.shape}")
    else:
        summary = _compute_summary(results)
        print(f"Recomputed summary from results → {summary.shape}")

    layers = sorted(results["layer"].unique())
    coefs  = sorted(results["coef"].unique())
    print(f"  Layers      : {layers[0]}–{layers[-1]}  ({len(layers)} layers)")
    print(f"  Coefficients: {len(coefs)} values  [{coefs[0]} … {coefs[-1]}]")
    print(f"  Questions   : {results['question_id'].nunique()}")

    return results, summary


def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute summary table from a results DataFrame."""
    rows = []
    for (layer, coef), grp in df.groupby(["layer", "coef"]):
        row: Dict = {
            "layer":        int(layer),
            "coef":         float(coef),
            "n":            len(grp),
            "mean_logprob": grp["logprob"].mean(),
            "std_logprob":  grp["logprob"].std(),
            "mean_rank":    grp["rank"].mean(),
            "median_rank":  grp["rank"].median(),
        }
        if coef != 0.0:
            row.update({
                "mean_delta_logprob":   grp["delta_logprob"].mean(),
                "std_delta_logprob":    grp["delta_logprob"].std(),
                "pct_improved_logprob": (grp["delta_logprob"] > 0).mean(),
                "pct_hurt_logprob":     (grp["delta_logprob"] < 0).mean(),
                "mean_rank_change":     grp["rank_change"].mean(),
                "pct_improved_rank":    (grp["rank_change"] > 0).mean(),
                "pct_hurt_rank":        (grp["rank_change"] < 0).mean(),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["layer", "coef"]).reset_index(drop=True)


# =========================================================================
# CROSS-LAYER PLOTS
# =========================================================================

# ---- Plot 1: Delta log-prob heatmap  (layers × coefficients) ------------

def plot_delta_logprob_heatmap(summary: pd.DataFrame) -> plt.Figure:
    """
    Heatmap: mean delta log-prob of the target's first token across layers
    and steering coefficients.  Blue = improved, Red = hurt.
    Analogous to the accuracy heatmap in the MCQ version.
    """
    steered = summary[summary["coef"] != 0.0].copy()
    pivot   = steered.pivot(index="layer", columns="coef",
                            values="mean_delta_logprob").sort_index()

    vmax = np.nanmax(np.abs(pivot.values))

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns],
                        rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title("Mean Δ Log-Prob of Target's First Token\nBlue = improved, Red = hurt")
    plt.colorbar(im, ax=ax, label="Δ log-prob (steered − baseline)")
    plt.tight_layout()
    return fig


# ---- Plot 2: Mean log-prob lines, one line per layer --------------------

def plot_logprob_lines_by_layer(summary: pd.DataFrame,
                                 highlight_layers: Optional[List[int]] = None
                                 ) -> plt.Figure:
    """
    Line plot: mean log-prob vs coefficient, one line per layer.
    Analogous to accuracy lines by layer in the MCQ version.

    Args:
        highlight_layers: layers to draw bold and labelled; all others faded.
                          If None, all layers shown equally.
    """
    layers = sorted(summary["layer"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for layer in layers:
        sub = summary[summary["layer"] == layer].sort_values("coef")
        if highlight_layers is not None:
            if layer in highlight_layers:
                ax.plot(sub["coef"], sub["mean_logprob"],
                        marker="o", linewidth=2, label=f"Layer {layer}")
            else:
                ax.plot(sub["coef"], sub["mean_logprob"],
                        color="lightgray", linewidth=0.5, alpha=0.5)
        else:
            ax.plot(sub["coef"], sub["mean_logprob"],
                    marker=".", markersize=3, linewidth=1, alpha=0.6,
                    label=f"L{layer}")

    baseline = summary[summary["coef"] == 0.0]
    if not baseline.empty:
        ax.axhline(baseline["mean_logprob"].mean(), linestyle="--",
                    color="black", alpha=0.7, label="Baseline")

    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Mean Log-Prob  [log P(first token | prompt)]")
    ax.set_title("Mean Target First-Token Log-Prob vs Coefficient per Layer")
    if highlight_layers is not None or len(layers) <= 8:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---- Plot 3: Best coefficient per layer ---------------------------------

def plot_best_coef_per_layer(summary: pd.DataFrame) -> plt.Figure:
    """
    Bar chart: for each layer, which coefficient gives the largest
    mean_delta_logprob improvement, and how large is that improvement.
    Analogous to the best-coef-per-layer plot in the MCQ version.
    """
    steered  = summary[summary["coef"] != 0.0].copy()
    best     = steered.loc[steered.groupby("layer")["mean_delta_logprob"].idxmax()]
    best     = best.sort_values("layer")
    baseline = summary[summary["coef"] == 0.0].groupby("layer")["mean_logprob"].mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = ["#2ecc71" if d > 0 else "#e74c3c"
              for d in best["mean_delta_logprob"]]
    axes[0].bar(best["layer"], best["mean_delta_logprob"],
                color=colors, alpha=0.8)
    axes[0].axhline(0, linestyle="--", color="black", alpha=0.7)
    axes[0].set_ylabel("Best Mean Δ Log-Prob")
    axes[0].set_title("Best Steering Result per Layer (by max mean Δ log-prob)")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(best["layer"], best["coef"], color="steelblue", alpha=0.7)
    axes[1].axhline(0, linestyle=":", color="gray", alpha=0.4)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Best Coefficient")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ---- Plot 4: % Improved heatmap  (layers × coefficients) ----------------

def plot_pct_improved_heatmap(summary: pd.DataFrame) -> plt.Figure:
    """
    Heatmap: fraction of questions whose target first-token log-prob
    improved (delta > 0) under steering, across layers and coefficients.
    Analogous to the % improved heatmap in the MCQ version.
    """
    steered = summary[summary["coef"] != 0.0].copy()
    pivot   = steered.pivot(index="layer", columns="coef",
                            values="pct_improved_logprob").sort_index()

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns],
                        rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title("% Questions with Improved Target First-Token Log-Prob")
    plt.colorbar(im, ax=ax, label="Fraction improved")
    plt.tight_layout()
    return fig


# ---- Plot 5: Rank-change heatmap  (layers × coefficients) ---------------

def plot_rank_change_heatmap(summary: pd.DataFrame) -> plt.Figure:
    """
    Heatmap: mean rank_change = base_rank − steered_rank across layers
    and coefficients.  Positive (blue) = target token rose in vocab ranking.
    No direct analog in the MCQ version (there rank was 1–4).
    """
    steered = summary[summary["coef"] != 0.0].copy()
    pivot   = steered.pivot(index="layer", columns="coef",
                            values="mean_rank_change").sort_index()

    vmax = np.nanmax(np.abs(pivot.values))

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns],
                        rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title("Mean Vocab Rank Change  (base_rank − steered_rank)\nBlue = rose in ranking, Red = fell")
    plt.colorbar(im, ax=ax, label="Rank change (positive = improved)")
    plt.tight_layout()
    return fig


# ---- Plot 6: Max delta per layer (bar chart) ----------------------------

def plot_max_delta_per_layer(summary: pd.DataFrame) -> plt.Figure:
    """
    For each layer: the best achievable delta and the worst delta across
    all tested coefficients.  Analogous to max accuracy change per layer.
    """
    steered = summary[summary["coef"] != 0.0]
    layers  = sorted(steered["layer"].unique())

    max_delta, min_delta = [], []
    for layer in layers:
        sub = steered[steered["layer"] == layer]["mean_delta_logprob"]
        max_delta.append(sub.max())
        min_delta.append(sub.min())

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(layers))

    ax.bar(x, max_delta, color="#2ecc71", alpha=0.75, label="Max Δ log-prob")
    ax.bar(x, min_delta, color="#e74c3c", alpha=0.75, label="Min Δ log-prob")
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δ Log-Prob")
    ax.set_title("Max / Min Achievable Mean Δ Log-Prob per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---- Plot 6b: Log-prob vs Prob side-by-side heatmap  --------------------

def plot_logprob_vs_prob_heatmap(results: pd.DataFrame) -> plt.Figure:
    """
    Side-by-side heatmaps comparing mean Δ log-prob and mean Δ prob
    (linear probability) across layers and steering coefficients.

    Δ prob = exp(logprob_steered) − exp(logprob_baseline)
           = exp(logprob) − exp(logprob − delta_logprob)

    Useful because a fixed Δ log-prob of e.g. 4 means very different things
    depending on the baseline probability level.  The prob-space panel shows
    the actual magnitude of change in probability.

    Blue = improved, Red = hurt (in both panels).
    """
    steered = results[results["coef"] != 0.0].copy()

    # Δ prob per question: exp(logprob_steered) − exp(logprob_baseline)
    steered["delta_prob"] = (
        np.exp(steered["logprob"])
        - np.exp(steered["logprob"] - steered["delta_logprob"])
    )

    agg = steered.groupby(["layer", "coef"]).agg(
        mean_delta_logprob=("delta_logprob", "mean"),
        mean_delta_prob=("delta_prob",       "mean"),
    ).reset_index()

    pivot_lp = agg.pivot(index="layer", columns="coef",
                         values="mean_delta_logprob").sort_index()
    pivot_p  = agg.pivot(index="layer", columns="coef",
                         values="mean_delta_prob").sort_index()

    vmax_lp = np.nanmax(np.abs(pivot_lp.values))
    vmax_p  = np.nanmax(np.abs(pivot_p.values))

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    for ax, pivot, vmax, title, cbar_label in [
        (axes[0], pivot_lp, vmax_lp,
         "Mean Δ Log-Prob  (log space)",
         "Δ log-prob"),
        (axes[1], pivot_p, vmax_p,
         "Mean Δ Prob  (linear probability space)",
         "Δ probability"),
    ]:
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Steering Coefficient")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=cbar_label)

    fig.suptitle(
        "Steering Effect: Log-Prob Space vs Linear Probability Space\n"
        "Blue = improved, Red = hurt",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


# =========================================================================
# SINGLE-LAYER PLOTS
# =========================================================================

# ---- Plot 7: Log-prob vs coefficient  (single layer) --------------------

def plot_logprob_vs_coef(summary: pd.DataFrame, layer: int) -> plt.Figure:
    """
    Mean log-prob and mean rank vs coefficient for one layer.
    Analogous to accuracy vs coefficient in the MCQ version.
    Shows both metrics on a dual y-axis for easy comparison.
    """
    sub = summary[summary["layer"] == layer].sort_values("coef")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(sub["coef"], sub["mean_logprob"],
             color="steelblue", marker="o", label="Mean log-prob")
    ax2.plot(sub["coef"], sub["mean_rank"],
             color="darkorange", marker="s", linestyle="--", label="Mean rank")

    baseline = sub[sub["coef"] == 0.0]
    if not baseline.empty:
        ax1.axhline(baseline["mean_logprob"].iloc[0],
                    linestyle=":", color="steelblue", alpha=0.5)
        ax2.axhline(baseline["mean_rank"].iloc[0],
                    linestyle=":", color="darkorange", alpha=0.5)

    ax1.set_xlabel("Steering Coefficient")
    ax1.set_ylabel("Mean Log-Prob", color="steelblue")
    ax2.set_ylabel("Mean Vocab Rank (lower = better)", color="darkorange")
    ax1.set_title(f"Layer {layer}: Target First-Token Log-Prob and Rank vs Coefficient")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---- Plot 8: Delta log-prob distribution  (single layer) ----------------

def plot_delta_distribution(results: pd.DataFrame, layer: int) -> plt.Figure:
    """
    Box plot: per-question delta_logprob at each non-zero coefficient,
    for one layer.  Analogous to the delta distribution boxplot in the MCQ
    version but uses the full-vocab log-prob shift instead of MCQ sum/first.
    """
    sub = results[(results["layer"] == layer) & (results["coef"] != 0.0)]
    if sub.empty:
        print(f"No steered data for layer {layer}.")
        return None

    coefs = sorted(sub["coef"].unique())
    data  = [sub[sub["coef"] == c]["delta_logprob"].dropna().values for c in coefs]

    fig, ax = plt.subplots(figsize=(max(8, len(coefs) * 0.6), 5))
    bp = ax.boxplot(data, labels=[f"{c:.2f}" for c in coefs],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.axhline(0.0, linestyle="--", color="red", alpha=0.7, label="No change")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Δ Log-Prob (steered − baseline)")
    ax.set_title(f"Layer {layer}: Per-Question Target First-Token Log-Prob Shift")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---- Plot 9: Rank-change distribution  (single layer) -------------------

def plot_rank_change_distribution(results: pd.DataFrame, layer: int) -> plt.Figure:
    """
    Box plot: per-question rank_change (base_rank − steered_rank) at each
    non-zero coefficient for one layer.  Positive = token rose in vocab
    ranking.  No direct MCQ analog.
    """
    sub = results[(results["layer"] == layer) & (results["coef"] != 0.0)]
    if sub.empty:
        print(f"No steered data for layer {layer}.")
        return None

    coefs = sorted(sub["coef"].unique())
    data  = [sub[sub["coef"] == c]["rank_change"].dropna().values for c in coefs]

    fig, ax = plt.subplots(figsize=(max(8, len(coefs) * 0.6), 5))
    bp = ax.boxplot(data, labels=[f"{c:.2f}" for c in coefs],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightyellow")
        patch.set_alpha(0.8)

    ax.axhline(0.0, linestyle="--", color="red", alpha=0.7, label="No change")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Rank Change  (base − steered,  positive = improved)")
    ax.set_title(f"Layer {layer}: Per-Question Vocab Rank Change")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---- Plot 10: Vocab rank bucket distribution  (single layer) ------------

_RANK_BINS   = [1, 2, 6, 21, 101, np.inf]
_RANK_LABELS = ["1", "2–5", "6–20", "21–100", "101+"]
_RANK_COLORS = ["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]


def _rank_bucket(rank_series: pd.Series) -> pd.Series:
    """Map integer ranks to bucket labels."""
    return pd.cut(rank_series, bins=_RANK_BINS, right=False,
                  labels=_RANK_LABELS)


def plot_vocab_rank_buckets(results: pd.DataFrame, layer: int) -> plt.Figure:
    """
    Stacked bar: fraction of questions falling in each vocab-rank bucket
    per coefficient, for one layer.

    Buckets: rank 1 | 2–5 | 6–20 | 21–100 | 101+

    Analogous to the rank distribution stacked bar in the MCQ version but
    adapted for the much larger vocab (32k+) rather than 4 candidates.
    """
    sub   = results[results["layer"] == layer]
    coefs = sorted(sub["coef"].unique())

    fractions: Dict[str, List[float]] = {b: [] for b in _RANK_LABELS}
    for coef in coefs:
        group   = sub[sub["coef"] == coef]
        buckets = _rank_bucket(group["rank"])
        total   = max(len(group), 1)
        counts  = buckets.value_counts()
        for label in _RANK_LABELS:
            fractions[label].append(counts.get(label, 0) / total)

    fig, ax = plt.subplots(figsize=(max(8, len(coefs) * 0.6), 5))
    x      = np.arange(len(coefs))
    labels = [f"{c:.2f}" for c in coefs]
    bottom = np.zeros(len(coefs))

    for label, color in zip(_RANK_LABELS, _RANK_COLORS):
        vals = np.array(fractions[label])
        ax.bar(x, vals, bottom=bottom, label=f"Rank {label}",
               color=color, alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title(f"Layer {layer}: Target Token Vocab Rank Distribution per Coefficient")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---- Plot 11: Per-question heatmap  (single layer) ----------------------

def plot_per_question_heatmap(results: pd.DataFrame, layer: int,
                               max_questions: int = 50) -> plt.Figure:
    """
    Heatmap: questions × coefficients, colored by delta_logprob.
    Baseline column (coef=0) is shown as all-zero reference.
    Analogous to per-question heatmap in the MCQ version but uses
    delta_logprob instead of target probability.
    """
    sub   = results[results["layer"] == layer]
    q_ids = sorted(sub["question_id"].unique())[:max_questions]
    sub   = sub[sub["question_id"].isin(q_ids)]

    pivot = sub.pivot(index="question_id", columns="coef",
                      values="delta_logprob").fillna(0.0)

    vmax = np.nanmax(np.abs(pivot.values))

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.5),
                                     max(6, len(q_ids) * 0.25)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns],
                        rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Question ID")
    ax.set_title(f"Layer {layer}: Per-Question Δ Log-Prob  (blue = improved, red = hurt)")
    plt.colorbar(im, ax=ax, label="Δ log-prob")
    plt.tight_layout()
    return fig


# ---- Plot 12: Improved / Unchanged / Hurt  (single layer) ---------------

def plot_improved_hurt(summary: pd.DataFrame, layer: int) -> plt.Figure:
    """
    Stacked bar: fraction of questions improved / unchanged / hurt at
    each non-zero coefficient for one layer.  Shown for both log-prob
    and vocab rank.  Analogous to improved/hurt bar in the MCQ version.
    """
    sub = summary[(summary["layer"] == layer) &
                  (summary["coef"] != 0.0)].sort_values("coef")

    if sub.empty or "pct_improved_logprob" not in sub.columns:
        print(f"No steered summary data for layer {layer}.")
        return None

    labels = [f"{c:.2f}" for c in sub["coef"]]
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(labels) * 0.8), 5),
                              sharey=True)

    for ax, imp_col, hurt_col, title_suffix in [
        (axes[0], "pct_improved_logprob", "pct_hurt_logprob", "Log-Prob"),
        (axes[1], "pct_improved_rank",    "pct_hurt_rank",    "Vocab Rank"),
    ]:
        improved  = sub[imp_col].values
        hurt      = sub[hurt_col].values
        unchanged = np.clip(1.0 - improved - hurt, 0, 1)

        ax.bar(x, improved,  label="Improved",   color="#2ecc71", alpha=0.8)
        ax.bar(x, unchanged, bottom=improved,
               label="Unchanged", color="#bdc3c7", alpha=0.6)
        ax.bar(x, hurt, bottom=improved + unchanged,
               label="Hurt", color="#e74c3c", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Steering Coefficient")
        ax.set_ylabel("Fraction of Questions")
        ax.set_title(f"Layer {layer}: Steering Impact on {title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# =========================================================================
# VALIDATION / TEST SPLIT  &  GENERALIZATION CHECK
# =========================================================================

def split_val_test(results: pd.DataFrame,
                   test_frac: float = 0.15,
                   seed: int = 42) -> Tuple[Set[int], Set[int]]:
    """
    Randomly split question IDs into validation and test sets.

    Returns:
        val_ids, test_ids  (sets of question IDs)
    """
    question_ids = np.array(sorted(results["question_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(question_ids)

    n_test   = max(1, int(len(question_ids) * test_frac))
    test_ids = set(question_ids[:n_test])
    val_ids  = set(question_ids[n_test:])

    print(f"Split: {len(val_ids)} val  /  {len(test_ids)} test  "
          f"({len(test_ids)/len(question_ids)*100:.1f}% test)")
    return val_ids, test_ids


def find_best_coef_per_layer(results: pd.DataFrame,
                              val_ids: Set[int]) -> Dict[int, Dict]:
    """
    For each layer, find the steering coefficient that maximises mean
    delta_logprob on the validation set.
    Analogous to find_best_coef_per_layer in the MCQ version.

    Returns:
        dict  {layer: {"best_coef": float, "val_delta": float,
                        "val_baseline_logprob": float}}
    """
    val_df  = results[results["question_id"].isin(val_ids)]
    steered = val_df[val_df["coef"] != 0.0]

    best = {}
    for layer in sorted(steered["layer"].unique()):
        layer_df = steered[steered["layer"] == layer]
        delta_per_coef = layer_df.groupby("coef")["delta_logprob"].mean()
        best_coef      = delta_per_coef.idxmax()

        base_lp = val_df[(val_df["layer"] == layer) &
                         (val_df["coef"] == 0.0)]["logprob"].mean()

        best[layer] = {
            "best_coef":            best_coef,
            "val_delta":            float(delta_per_coef[best_coef]),
            "val_pct_improved":     float((layer_df[layer_df["coef"] == best_coef]
                                           ["delta_logprob"] > 0).mean()),
            "val_baseline_logprob": float(base_lp),
        }
    return best


def evaluate_on_test(results: pd.DataFrame,
                     test_ids: Set[int],
                     best_per_layer: Dict[int, Dict]) -> pd.DataFrame:
    """
    Evaluate the validation-selected best coefficient on the held-out test
    set.  Returns one row per layer.
    Analogous to evaluate_on_test in the MCQ version.
    """
    test_df = results[results["question_id"].isin(test_ids)]

    rows = []
    for layer, info in sorted(best_per_layer.items()):
        best_coef  = info["best_coef"]
        layer_test = test_df[test_df["layer"] == layer]

        at_best       = layer_test[layer_test["coef"] == best_coef]["delta_logprob"]
        test_base_lp  = layer_test[layer_test["coef"] == 0.0]["logprob"].mean()

        rows.append({
            "layer":                  layer,
            "best_coef":              best_coef,
            "val_delta":              info["val_delta"],
            "val_pct_improved":       info["val_pct_improved"],
            "val_baseline_logprob":   info["val_baseline_logprob"],
            "test_delta":             at_best.mean()           if not at_best.empty else np.nan,
            "test_pct_improved":      (at_best > 0).mean()     if not at_best.empty else np.nan,
            "test_baseline_logprob":  float(test_base_lp),
        })
    return pd.DataFrame(rows)


# ---- Plot 13: Val vs test delta (grouped bars per layer) ----------------

def plot_val_test_analysis(results_df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart comparing val-selected performance with test
    performance per layer.  Analogous to plot_val_test_accuracy in the
    MCQ version but uses delta_logprob and pct_improved instead of accuracy.
    """
    layers = results_df["layer"].values
    x      = np.arange(len(layers))
    w      = 0.3

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # --- Top: mean delta logprob ---
    axes[0].bar(x - w / 2, results_df["val_delta"],  w,
                label="Val Δ log-prob",  color="steelblue",  alpha=0.8)
    axes[0].bar(x + w / 2, results_df["test_delta"], w,
                label="Test Δ log-prob", color="darkorange", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Mean Δ Log-Prob")
    axes[0].set_title("Val-Selected Best Coefficient: Generalization to Test Set")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Middle: % improved ---
    axes[1].bar(x - w / 2, results_df["val_pct_improved"],  w,
                label="Val % improved",  color="steelblue",  alpha=0.8)
    axes[1].bar(x + w / 2, results_df["test_pct_improved"], w,
                label="Test % improved", color="darkorange", alpha=0.8)
    axes[1].axhline(0.5, linestyle=":", color="gray", alpha=0.5,
                    label="50% line")
    axes[1].set_ylabel("Fraction of Questions Improved")
    axes[1].set_title("% Questions Improved per Layer")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    # --- Bottom: which coefficient was selected ---
    axes[2].bar(x, results_df["best_coef"], color="teal", alpha=0.7)
    axes[2].axhline(0, linestyle=":", color="gray", alpha=0.4)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Best Coefficient")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(layers)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ---- Plot 14: Val vs test scatter per layer  ----------------------------

def plot_val_test_scatter(results_df: pd.DataFrame) -> plt.Figure:
    """
    Scatter: val_delta vs test_delta per layer.  Points above the y=x line
    generalise better than expected on validation.
    Analogous to plot_val_test_scatter in the MCQ version.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(results_df["val_delta"], results_df["test_delta"],
                    c=results_df["layer"], cmap="viridis",
                    s=70, edgecolors="black", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Layer")

    for _, row in results_df.iterrows():
        ax.annotate(f"L{int(row['layer'])}",
                    (row["val_delta"], row["test_delta"]),
                    fontsize=7, alpha=0.8,
                    textcoords="offset points", xytext=(4, 4))

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
    ax.axhline(0, ls=":", color="gray", alpha=0.4)
    ax.axvline(0, ls=":", color="gray", alpha=0.4)

    ax.set_xlabel("Val Mean Δ Log-Prob")
    ax.set_ylabel("Test Mean Δ Log-Prob")
    ax.set_title("Generalization: Val vs Test Δ Log-Prob per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    return fig


def run_val_test_analysis(results: pd.DataFrame,
                           test_frac: float = 0.15,
                           seed: int = 42):
    """
    End-to-end: split → find best coefs on val → evaluate on test → plot.

    Returns:
        eval_df, fig_bars, fig_scatter
    """
    val_ids, test_ids  = split_val_test(results, test_frac=test_frac, seed=seed)
    best_per_layer     = find_best_coef_per_layer(results, val_ids)
    eval_df            = evaluate_on_test(results, test_ids, best_per_layer)

    print("\n" + "=" * 70)
    print("VALIDATION / TEST RESULTS")
    print("=" * 70)
    print(eval_df.to_string(index=False))
    n_gen = (eval_df["test_delta"] > 0).sum()
    print(f"\nLayers with positive test delta: {n_gen}/{len(eval_df)}")

    fig_bars    = plot_val_test_analysis(eval_df)
    fig_scatter = plot_val_test_scatter(eval_df)

    return eval_df, fig_bars, fig_scatter


# =========================================================================
# K-FOLD CROSS-VALIDATION
# =========================================================================

def kfold_val_test(results: pd.DataFrame,
                   k: int = 5,
                   seed: int = 42) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    K-fold cross-validated coefficient selection.

    For each fold the held-out questions are the test set; the rest are used
    to pick the best coefficient per layer.  Results averaged across folds.
    Analogous to kfold_val_test in the MCQ version.

    Returns:
        fold_results: list of per-fold DataFrames
        avg_results:  DataFrame averaged over folds, one row per layer
    """
    question_ids = np.array(sorted(results["question_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(question_ids)

    folds        = np.array_split(question_ids, k)
    fold_results = []

    for fold_idx in range(k):
        test_ids = set(folds[fold_idx])
        val_ids  = set(qid for i, fold in enumerate(folds)
                       if i != fold_idx for qid in fold)

        print(f"\n--- Fold {fold_idx + 1}/{k}  "
              f"(val={len(val_ids)}, test={len(test_ids)}) ---")

        best_per_layer = find_best_coef_per_layer(results, val_ids)
        fold_df        = evaluate_on_test(results, test_ids, best_per_layer)
        fold_df["fold"] = fold_idx
        fold_results.append(fold_df)

    all_folds = pd.concat(fold_results, ignore_index=True)

    avg_results = (
        all_folds
        .groupby("layer")
        .agg(
            best_coef_mean     =("best_coef",       "mean"),
            best_coef_std      =("best_coef",       "std"),
            val_delta          =("val_delta",        "mean"),
            val_pct_improved   =("val_pct_improved", "mean"),
            test_delta         =("test_delta",       "mean"),
            test_delta_std     =("test_delta",       "std"),
            test_pct_improved  =("test_pct_improved","mean"),
            test_pct_imp_std   =("test_pct_improved","std"),
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print(f"{k}-FOLD CROSS-VALIDATED RESULTS")
    print("=" * 70)
    print(avg_results.to_string(index=False))
    n_gen = (avg_results["test_delta"] > 0).sum()
    print(f"\nLayers with positive mean test delta: {n_gen}/{len(avg_results)}")

    return fold_results, avg_results


# ---- Plot 15: K-Fold results bars per layer  ----------------------------

def plot_kfold_results(avg_results: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart of mean val / test delta and % improved per layer,
    with error bars from cross-validation std.
    Analogous to plot_kfold_accuracy in the MCQ version.
    """
    layers = avg_results["layer"].values
    x      = np.arange(len(layers))
    w      = 0.3

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # --- Top: mean delta logprob ---
    axes[0].bar(x - w / 2, avg_results["val_delta"],  w,
                label="Val Δ log-prob", color="steelblue", alpha=0.8)
    axes[0].bar(x + w / 2, avg_results["test_delta"], w,
                yerr=avg_results["test_delta_std"], capsize=3,
                label="Test Δ log-prob", color="darkorange", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Mean Δ Log-Prob")
    axes[0].set_title(f"K-Fold CV: Val-Selected Coefficient Generalization")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Middle: % improved ---
    axes[1].bar(x - w / 2, avg_results["val_pct_improved"],  w,
                label="Val % improved",  color="steelblue",  alpha=0.8)
    axes[1].bar(x + w / 2, avg_results["test_pct_improved"], w,
                yerr=avg_results["test_pct_imp_std"], capsize=3,
                label="Test % improved", color="darkorange", alpha=0.8)
    axes[1].axhline(0.5, linestyle=":", color="gray", alpha=0.5,
                    label="50% line")
    axes[1].set_ylabel("Fraction of Questions Improved")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    # --- Bottom: mean best coefficient ---
    axes[2].bar(x, avg_results["best_coef_mean"], color="teal", alpha=0.7,
                yerr=avg_results["best_coef_std"], capsize=3)
    axes[2].axhline(0, linestyle=":", color="gray", alpha=0.4)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Best Coefficient (mean ± std)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(layers)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ---- Plot 16: K-Fold val vs test scatter  ------------------------------

def plot_kfold_scatter(avg_results: pd.DataFrame) -> plt.Figure:
    """
    Scatter: mean val delta vs mean test delta per layer, with test std
    error bars.  Analogous to plot_kfold_scatter in the MCQ version.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.errorbar(avg_results["val_delta"], avg_results["test_delta"],
                yerr=avg_results["test_delta_std"],
                fmt="none", ecolor="gray", alpha=0.4, zorder=1)

    sc = ax.scatter(avg_results["val_delta"], avg_results["test_delta"],
                    c=avg_results["layer"], cmap="viridis",
                    s=70, edgecolors="black", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Layer")

    for _, row in avg_results.iterrows():
        ax.annotate(f"L{int(row['layer'])}",
                    (row["val_delta"], row["test_delta"]),
                    fontsize=7, alpha=0.8,
                    textcoords="offset points", xytext=(4, 4))

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
    ax.axhline(0, ls=":", color="gray", alpha=0.4)
    ax.axvline(0, ls=":", color="gray", alpha=0.4)

    ax.set_xlabel("Mean Val Δ Log-Prob")
    ax.set_ylabel("Mean Test Δ Log-Prob")
    ax.set_title("K-Fold Generalization: Val vs Test Δ Log-Prob per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    return fig


def run_kfold_analysis(results: pd.DataFrame, k: int = 5, seed: int = 42):
    """
    End-to-end K-Fold: split → select coefs on val → evaluate on test
    → average across folds → plot.

    Returns:
        fold_results, avg_results, fig_bars, fig_scatter
    """
    fold_results, avg_results = kfold_val_test(results, k=k, seed=seed)
    fig_bars    = plot_kfold_results(avg_results)
    fig_scatter = plot_kfold_scatter(avg_results)
    return fold_results, avg_results, fig_bars, fig_scatter
