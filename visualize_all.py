#!/usr/bin/env python3
"""
visualize_all.py — Comprehensive visualization for one experiment output directory.

Usage:
    python visualize_all.py results/exp_20260413_anatomy_llama8b_pilot/
    python visualize_all.py results/exp_20260413_anatomy_llama8b_pilot/ --top-layers 5
    python visualize_all.py results/exp_20260413_anatomy_llama8b_pilot/ --best-layer 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


# =============================================================================
# Data loading
# =============================================================================

def _find_exp_dir(base_dir: Path) -> Path:
    if (base_dir / "mcf").exists() or (base_dir / "cf").exists():
        return base_dir
    for c in base_dir.iterdir():
        if c.is_dir() and ((c / "mcf").exists() or (c / "cf").exists()):
            return c
    return base_dir


def _load_cf_wide(cf_dir: Path) -> Optional[pd.DataFrame]:
    frames = []
    for layer_dir in sorted(cf_dir.glob("layer_*")):
        try:
            layer_idx = int(layer_dir.name.split("_")[-1])
        except ValueError:
            continue
        wide_path = layer_dir / "detailed_wide.csv"
        if wide_path.exists():
            df = pd.read_csv(wide_path)
            df["layer"] = layer_idx
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded CF wide:      {combined.shape}  ({combined['layer'].nunique()} layers)")
    return combined


def load_data(base_dir: Path) -> dict:
    exp_dir = _find_exp_dir(base_dir)
    data: dict = {
        "exp_dir": exp_dir,
        "mcf_results": None,
        "mcf_summary": None,
        "cf_summary": None,
        "cf_wide": None,
        "config": None,
    }

    mcf_dir = exp_dir / "mcf"
    if mcf_dir.exists():
        p = mcf_dir / "combined_results.csv"
        if p.exists():
            data["mcf_results"] = pd.read_csv(p)
            print(f"Loaded MCF results:  {data['mcf_results'].shape}")
        p = mcf_dir / "combined_summary.csv"
        if p.exists():
            data["mcf_summary"] = pd.read_csv(p)
            print(f"Loaded MCF summary:  {data['mcf_summary'].shape}")

    cf_dir = exp_dir / "cf"
    if cf_dir.exists():
        p = cf_dir / "combined_summary.csv"
        if p.exists():
            data["cf_summary"] = pd.read_csv(p)
            print(f"Loaded CF summary:   {data['cf_summary'].shape}")
        data["cf_wide"] = _load_cf_wide(cf_dir)

    p = exp_dir / "config.yaml"
    if p.exists():
        import yaml
        with open(p) as f:
            data["config"] = yaml.safe_load(f)

    return data


def _exp_label(data: dict) -> str:
    cfg = data.get("config")
    if cfg and "experiment_id" in cfg:
        return cfg["experiment_id"]
    return data["exp_dir"].name


# =============================================================================
# Numerical summaries
# =============================================================================

def print_mcf_numbers(summary: pd.DataFrame, label: str, results: pd.DataFrame = None):
    cols_exist = [c for c in [
        "coef", "accuracy",
        "mean_correct_label_logprob", "mean_delta_correct_logprob",
        "pct_improved_logprob", "pct_hurt_logprob",
        "mean_correct_label_rank", "mean_rank_change",
    ] if c in summary.columns]

    W = 120
    print(f"\n{'='*W}")
    print(f"  MCF NUMBERS — {label}")
    print(f"{'='*W}")

    for layer in sorted(summary["layer"].unique()):
        sub = summary[summary["layer"] == layer][cols_exist].sort_values("coef")
        print(f"\n  Layer {layer}")
        print(sub.to_string(index=False))

    # One-line best-per-layer summary
    print(f"\n{'─'*W}")
    print("  BEST STEERED RESULT PER LAYER (MCF)")
    print(f"{'─'*W}")
    steered = summary[summary["coef"] != 0.0]
    best = steered.loc[steered.groupby("layer")["accuracy"].idxmax()][
        ["layer"] + [c for c in cols_exist if c != "layer"]
    ].sort_values("layer")
    base_acc = summary[summary["coef"] == 0.0].set_index("layer")["accuracy"]
    best = best.copy()
    best["baseline_acc"] = best["layer"].map(base_acc)
    best["delta_acc"] = best["accuracy"] - best["baseline_acc"]
    print(best.to_string(index=False))

    # Asymmetry table: compare +c vs -c at the same magnitude
    if results is not None and "delta_correct_logprob" in results.columns:
        print(f"\n{'─'*W}")
        print("  ASYMMETRY TABLE  (|Δ(+c) + Δ(-c)| near 0 → symmetric hurt)")
        print(f"{'─'*W}")
        print(f"  {'layer':>5} {'|coef|':>7} {'Δ(+c)':>10} {'Δ(-c)':>10} "
              f"{'Δ++Δ-':>10} {'pct+(+c)':>10} {'pct+(-c)':>10}")
        print(f"  {'-'*W}")
        nz = results[results["coef"] != 0.0].copy()
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
        for _, r in wide.iterrows():
            layer = int(r[("layer", "")])
            ac = float(r[("abs_coef", "")])
            d_pos = r[("mean_delta", "pos")]
            d_neg = r[("mean_delta", "neg")]
            p_pos = r[("pct_improved", "pos")]
            p_neg = r[("pct_improved", "neg")]
            if any(pd.isna(v) for v in [d_pos, d_neg, p_pos, p_neg]):
                continue
            print(f"  {layer:>5d} {ac:>7.2f} {d_pos:>10.4f} {d_neg:>10.4f} "
                  f"{(d_pos + d_neg):>10.4f} {p_pos:>10.3f} {p_neg:>10.3f}")
    print()


def print_cf_numbers(summary: pd.DataFrame, label: str):
    cols_exist = [c for c in [
        "coef",
        "accuracy_sum", "accuracy_char", "accuracy_mean", "accuracy_pmi",
        "mean_target_sum_lp", "mean_delta_target_sum_lp",
        "pct_improved_sum", "pct_hurt_sum",
        "mean_margin_sum", "mean_target_rank_sum",
    ] if c in summary.columns]

    W = 120
    print(f"\n{'='*W}")
    print(f"  CF NUMBERS — {label}")
    print(f"{'='*W}")

    for layer in sorted(summary["layer"].unique()):
        sub = summary[summary["layer"] == layer][cols_exist].sort_values("coef")
        print(f"\n  Layer {layer}")
        print(sub.to_string(index=False))

    print(f"\n{'─'*W}")
    print("  BEST STEERED RESULT PER LAYER (CF — sum scoring)")
    print(f"{'─'*W}")
    steered = summary[summary["coef"] != 0.0]
    if "accuracy_sum" in steered.columns:
        best = steered.loc[steered.groupby("layer")["accuracy_sum"].idxmax()][
            ["layer"] + [c for c in cols_exist if c != "layer"]
        ].sort_values("layer")
        base_acc = summary[summary["coef"] == 0.0].set_index("layer")["accuracy_sum"]
        best = best.copy()
        best["baseline_acc"] = best["layer"].map(base_acc)
        best["delta_acc"] = best["accuracy_sum"] - best["baseline_acc"]
        print(best.to_string(index=False))
    print()


# =============================================================================
# Helpers
# =============================================================================

def _show(fig: plt.Figure):
    plt.figure(fig.number)
    plt.show()


def _heatmap(ax, pivot: pd.DataFrame, cmap: str,
             vmin=None, vmax=None, cbar_label: str = "",
             symmetric: bool = False):
    if symmetric:
        v = np.nanmax(np.abs(pivot.values))
        vmin, vmax = -v, v
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    plt.colorbar(im, ax=ax, label=cbar_label)
    return im


def _pivot(df: pd.DataFrame, value_col: str,
           steered_only: bool = False) -> Optional[pd.DataFrame]:
    if value_col not in df.columns:
        return None
    src = df[df["coef"] != 0.0] if steered_only else df
    try:
        return src.pivot(index="layer", columns="coef", values=value_col).sort_index()
    except Exception:
        return None


def _best_layers(summary: pd.DataFrame, acc_col: str, n: int) -> List[int]:
    if acc_col not in summary.columns:
        return sorted(summary["layer"].unique())[:n]
    steered = summary[summary["coef"] != 0.0]
    best = steered.groupby("layer")[acc_col].max()
    return best.nlargest(n).index.tolist()


def _baseline_series(summary: pd.DataFrame, col: str) -> pd.Series:
    return summary[summary["coef"] == 0.0].set_index("layer")[col]


# =============================================================================
# MCF — Cross-layer plots
# =============================================================================

def mcf_accuracy_heatmap(summary, label):
    pv = _pivot(summary, "accuracy")
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "RdYlGn", vmin=0, vmax=1, cbar_label="Accuracy")
    ax.set_title(f"{label} | MCF: Accuracy by Layer × Coefficient")
    plt.tight_layout()
    return fig


def mcf_delta_logprob_heatmap(summary, label):
    pv = _pivot(summary, "mean_delta_correct_logprob", steered_only=True)
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "RdBu", symmetric=True, cbar_label="Mean Δ correct log-prob")
    ax.set_title(f"{label} | MCF: Mean Δ Correct-Label Log-Prob  (blue=improved)")
    plt.tight_layout()
    return fig


def mcf_pct_improved_heatmap(summary, label):
    pv = _pivot(summary, "pct_improved_logprob", steered_only=True)
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "YlGn", vmin=0, vmax=1, cbar_label="Fraction improved")
    ax.set_title(f"{label} | MCF: % Questions with Improved Correct-Label Log-Prob")
    plt.tight_layout()
    return fig


def mcf_pct_hurt_heatmap(summary, label):
    pv = _pivot(summary, "pct_hurt_logprob", steered_only=True)
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "YlOrRd", vmin=0, vmax=1, cbar_label="Fraction hurt")
    ax.set_title(f"{label} | MCF: % Questions with Hurt Correct-Label Log-Prob")
    plt.tight_layout()
    return fig


def mcf_rank_heatmap(summary, label):
    pv = _pivot(summary, "mean_correct_label_rank")
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "RdYlGn_r", cbar_label="Mean correct-label rank (lower=better)")
    ax.set_title(f"{label} | MCF: Mean Correct-Label Vocab Rank (lower=better)")
    plt.tight_layout()
    return fig


def mcf_best_coef_per_layer(summary, label):
    if "accuracy" not in summary.columns:
        return None
    baseline = _baseline_series(summary, "accuracy")
    steered = summary[summary["coef"] != 0.0]
    best = steered.loc[steered.groupby("layer")["accuracy"].idxmax()].sort_values("layer")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors = ["#2ecc71" if best.iloc[i]["accuracy"] > baseline.get(best.iloc[i]["layer"], 0)
              else "#e74c3c" for i in range(len(best))]
    axes[0].bar(best["layer"], best["accuracy"], color=colors, alpha=0.85)
    axes[0].plot(best["layer"], [baseline.get(l, np.nan) for l in best["layer"]],
                 "k--", alpha=0.6, label="Baseline")
    axes[0].axhline(0.25, ls=":", color="gray", alpha=0.4, label="Random (25%)")
    axes[0].set_ylabel("Best Accuracy")
    axes[0].set_title(f"{label} | MCF: Best Steering Result per Layer")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(best["layer"], best["coef"], color="steelblue", alpha=0.75)
    axes[1].axhline(0, ls=":", color="gray", alpha=0.4)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Best Coefficient")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def mcf_max_accuracy_change(summary, label):
    if "accuracy" not in summary.columns:
        return None
    baseline = _baseline_series(summary, "accuracy")
    steered = summary[summary["coef"] != 0.0]
    layers = sorted(steered["layer"].unique())
    gains = [steered[steered["layer"] == l]["accuracy"].max() - baseline.get(l, 0) for l in layers]
    losses = [steered[steered["layer"] == l]["accuracy"].min() - baseline.get(l, 0) for l in layers]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(layers))
    ax.bar(x, gains, color="#2ecc71", alpha=0.75, label="Max gain")
    ax.bar(x, losses, color="#e74c3c", alpha=0.75, label="Max loss")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy Δ from Baseline")
    ax.set_title(f"{label} | MCF: Max Accuracy Gain / Loss per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def mcf_pos_neg_asymmetry(summary, label):
    if "accuracy" not in summary.columns:
        return None
    pos = summary[summary["coef"] > 0].groupby("layer")["accuracy"].max()
    neg = summary[summary["coef"] < 0].groupby("layer")["accuracy"].max()
    base = _baseline_series(summary, "accuracy")
    layers = sorted(set(pos.index) | set(neg.index) | set(base.index))
    x = np.arange(len(layers))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, [neg.get(l, np.nan) for l in layers], w,
           label="Best negative coef", color="darkorange", alpha=0.8)
    ax.bar(x, [base.get(l, np.nan) for l in layers], w,
           label="Baseline (coef=0)", color="gray", alpha=0.6)
    ax.bar(x + w, [pos.get(l, np.nan) for l in layers], w,
           label="Best positive coef", color="steelblue", alpha=0.8)
    ax.axhline(0.25, ls=":", color="gray", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{label} | MCF: Positive vs Negative Coefficient Asymmetry")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def mcf_accuracy_lines(summary, label, highlight_layers=None):
    if "accuracy" not in summary.columns:
        return None
    layers = sorted(summary["layer"].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for layer in layers:
        sub = summary[summary["layer"] == layer].sort_values("coef")
        if highlight_layers is not None:
            if layer in highlight_layers:
                ax.plot(sub["coef"], sub["accuracy"], marker="o",
                        linewidth=2, label=f"Layer {layer}")
            else:
                ax.plot(sub["coef"], sub["accuracy"], color="lightgray",
                        linewidth=0.5, alpha=0.4)
        else:
            ax.plot(sub["coef"], sub["accuracy"], marker=".", markersize=3,
                    linewidth=1, alpha=0.5)
    ax.axhline(0.25, ls=":", color="gray", alpha=0.5, label="Random (25%)")
    base = summary[summary["coef"] == 0.0]["accuracy"]
    if not base.empty:
        ax.axhline(base.iloc[0], ls="--", color="black", alpha=0.6, label="Baseline")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Accuracy")
    suffix = " (top layers highlighted)" if highlight_layers else " (all layers)"
    ax.set_title(f"{label} | MCF: Accuracy vs Coefficient per Layer{suffix}")
    if highlight_layers:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# MCF — Per-layer plots
# =============================================================================

def mcf_delta_distribution(results, label, layer):
    col = "delta_correct_logprob"
    if col not in results.columns:
        return None
    sub = results[(results["layer"] == layer) & (results["coef"] != 0.0)]
    if sub.empty:
        return None
    coefs = sorted(sub["coef"].unique())
    data = [sub[sub["coef"] == c][col].dropna().values for c in coefs]

    fig, ax = plt.subplots(figsize=(max(10, len(coefs) * 0.7), 5))
    bp = ax.boxplot(data, tick_labels=[f"{c:.1f}" for c in coefs],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax.axhline(0, ls="--", color="red", alpha=0.7, label="No change")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Δ Correct-Label Log-Prob")
    ax.set_title(f"{label} | MCF Layer {layer}: Per-Question Δ Log-Prob Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def mcf_rank_change_distribution(results, label, layer):
    col = "rank_change"
    if col not in results.columns:
        return None
    sub = results[(results["layer"] == layer) & (results["coef"] != 0.0)]
    if sub.empty:
        return None
    coefs = sorted(sub["coef"].unique())
    data = [sub[sub["coef"] == c][col].dropna().values for c in coefs]

    fig, ax = plt.subplots(figsize=(max(10, len(coefs) * 0.7), 5))
    bp = ax.boxplot(data, tick_labels=[f"{c:.1f}" for c in coefs],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightyellow")
        patch.set_alpha(0.8)
    ax.axhline(0, ls="--", color="red", alpha=0.7, label="No change")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Rank Change (positive = rose)")
    ax.set_title(f"{label} | MCF Layer {layer}: Per-Question Rank Change Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def mcf_per_question_heatmap(results, label, layer, max_questions=60):
    col = "delta_correct_logprob"
    if col not in results.columns:
        return None
    sub = results[results["layer"] == layer]
    if sub.empty:
        return None
    q_ids = sorted(sub["question_id"].unique())[:max_questions]
    sub = sub[sub["question_id"].isin(q_ids)]
    try:
        pivot = sub.pivot(index="question_id", columns="coef", values=col).fillna(0.0)
    except Exception:
        return None
    vmax = np.nanmax(np.abs(pivot.values)) or 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.6),
                                    max(6, len(q_ids) * 0.22)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Question ID")
    ax.set_title(f"{label} | MCF Layer {layer}: Per-Question Δ Log-Prob  (blue=improved)")
    plt.colorbar(im, ax=ax, label="Δ correct-label log-prob")
    plt.tight_layout()
    return fig


def mcf_improved_hurt_bar(summary, label, layer):
    if "pct_improved_logprob" not in summary.columns:
        return None
    sub = summary[(summary["layer"] == layer) & (summary["coef"] != 0.0)].sort_values("coef")
    if sub.empty:
        return None
    labels = [f"{c:.1f}" for c in sub["coef"]]
    x = np.arange(len(labels))
    improved = sub["pct_improved_logprob"].values
    hurt = sub["pct_hurt_logprob"].values
    unchanged = np.clip(1.0 - improved - hurt, 0, 1)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.bar(x, improved, label="Improved", color="#2ecc71", alpha=0.85)
    ax.bar(x, unchanged, bottom=improved, label="Unchanged", color="#bdc3c7", alpha=0.6)
    ax.bar(x, hurt, bottom=improved + unchanged, label="Hurt", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title(f"{label} | MCF Layer {layer}: Steering Impact")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# =============================================================================
# CF — Cross-layer plots
# =============================================================================

def cf_accuracy_heatmap(summary, label, scoring="sum"):
    pv = _pivot(summary, f"accuracy_{scoring}")
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "RdYlGn", vmin=0, vmax=1, cbar_label="Accuracy")
    ax.set_title(f"{label} | CF ({scoring}): Accuracy by Layer × Coefficient")
    plt.tight_layout()
    return fig


def cf_delta_heatmap(summary, label):
    pv = _pivot(summary, "mean_delta_target_sum_lp", steered_only=True)
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "RdBu", symmetric=True, cbar_label="Mean Δ target sum log-prob")
    ax.set_title(f"{label} | CF: Mean Δ Target Sum Log-Prob  (blue=improved)")
    plt.tight_layout()
    return fig


def cf_pct_improved_heatmap(summary, label, scoring="sum"):
    pv = _pivot(summary, f"pct_improved_{scoring}", steered_only=True)
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "YlGn", vmin=0, vmax=1, cbar_label="Fraction improved")
    ax.set_title(f"{label} | CF: % Improved ({scoring})")
    plt.tight_layout()
    return fig


def cf_rank_heatmap(summary, label, scoring="sum"):
    pv = _pivot(summary, f"mean_target_rank_{scoring}")
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "RdYlGn_r", cbar_label=f"Mean target rank ({scoring})")
    ax.set_title(f"{label} | CF: Mean Target Rank ({scoring}, lower=better)")
    plt.tight_layout()
    return fig


def cf_margin_heatmap(summary, label):
    pv = _pivot(summary, "mean_margin_sum")
    if pv is None:
        return None
    fig, ax = plt.subplots(figsize=(13, 7))
    _heatmap(ax, pv, "RdBu", symmetric=True, cbar_label="Mean margin")
    ax.set_title(f"{label} | CF: Mean Margin (sum)  (blue=target leads)")
    plt.tight_layout()
    return fig


def cf_best_coef_per_layer(summary, label, scoring="sum"):
    col = f"accuracy_{scoring}"
    if col not in summary.columns:
        return None
    baseline = _baseline_series(summary, col)
    steered = summary[summary["coef"] != 0.0]
    best = steered.loc[steered.groupby("layer")[col].idxmax()].sort_values("layer")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors = ["#2ecc71" if best.iloc[i][col] > baseline.get(best.iloc[i]["layer"], 0)
              else "#e74c3c" for i in range(len(best))]
    axes[0].bar(best["layer"], best[col], color=colors, alpha=0.85)
    axes[0].plot(best["layer"], [baseline.get(l, np.nan) for l in best["layer"]],
                 "k--", alpha=0.6, label="Baseline")
    axes[0].axhline(0.25, ls=":", color="gray", alpha=0.4, label="Random (25%)")
    axes[0].set_ylabel(f"Best Accuracy ({scoring})")
    axes[0].set_title(f"{label} | CF: Best Steering Result per Layer ({scoring})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(best["layer"], best["coef"], color="steelblue", alpha=0.75)
    axes[1].axhline(0, ls=":", color="gray", alpha=0.4)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Best Coefficient")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def cf_multi_scoring_comparison(summary, label):
    scorings = [s for s in ["sum", "mean", "char", "pmi"]
                if f"accuracy_{s}" in summary.columns]
    if not scorings:
        return None
    steered = summary[summary["coef"] != 0.0]
    layers = sorted(summary["layer"].unique())
    best_per = {sc: steered.groupby("layer")[f"accuracy_{sc}"].max() for sc in scorings}

    x = np.arange(len(layers))
    w = 0.8 / len(scorings)
    colors = ["steelblue", "darkorange", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, sc in enumerate(scorings):
        vals = [best_per[sc].get(l, np.nan) for l in layers]
        ax.bar(x + i * w - (len(scorings) - 1) * w / 2, vals, w,
               label=sc, color=colors[i % len(colors)], alpha=0.8)
    ax.axhline(0.25, ls=":", color="gray", alpha=0.4, label="Random (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Best Accuracy")
    ax.set_title(f"{label} | CF: Best Accuracy per Layer by Scoring Method")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def cf_accuracy_lines(summary, label, scoring="sum", highlight_layers=None):
    col = f"accuracy_{scoring}"
    if col not in summary.columns:
        return None
    layers = sorted(summary["layer"].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for layer in layers:
        sub = summary[summary["layer"] == layer].sort_values("coef")
        if highlight_layers is not None:
            if layer in highlight_layers:
                ax.plot(sub["coef"], sub[col], marker="o", linewidth=2,
                        label=f"Layer {layer}")
            else:
                ax.plot(sub["coef"], sub[col], color="lightgray",
                        linewidth=0.5, alpha=0.4)
        else:
            ax.plot(sub["coef"], sub[col], marker=".", markersize=3,
                    linewidth=1, alpha=0.5)
    ax.axhline(0.25, ls=":", color="gray", alpha=0.5, label="Random (25%)")
    base = summary[summary["coef"] == 0.0][col]
    if not base.empty:
        ax.axhline(base.iloc[0], ls="--", color="black", alpha=0.6, label="Baseline")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel(f"Accuracy ({scoring})")
    suffix = " (top layers highlighted)" if highlight_layers else " (all layers)"
    ax.set_title(f"{label} | CF ({scoring}): Accuracy vs Coefficient{suffix}")
    if highlight_layers:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def cf_delta_max_per_layer(summary, label):
    col = "mean_delta_target_sum_lp"
    if col not in summary.columns:
        return None
    steered = summary[summary["coef"] != 0.0]
    layers = sorted(steered["layer"].unique())
    max_d = [steered[steered["layer"] == l][col].max() for l in layers]
    min_d = [steered[steered["layer"] == l][col].min() for l in layers]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(layers))
    ax.bar(x, max_d, color="#2ecc71", alpha=0.75, label="Max Δ log-prob")
    ax.bar(x, min_d, color="#e74c3c", alpha=0.75, label="Min Δ log-prob")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δ Target Sum Log-Prob")
    ax.set_title(f"{label} | CF: Max / Min Achievable Mean Δ Log-Prob per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# =============================================================================
# CF — Per-layer plots
# =============================================================================

def cf_delta_distribution(cf_wide, label, layer):
    col = "delta_target_sum_lp"
    if col not in cf_wide.columns:
        return None
    sub = cf_wide[(cf_wide["layer"] == layer) & (cf_wide["coef"] != 0.0)]
    if sub.empty:
        return None
    coefs = sorted(sub["coef"].unique())
    data = [sub[sub["coef"] == c][col].dropna().values for c in coefs]

    fig, ax = plt.subplots(figsize=(max(10, len(coefs) * 0.7), 5))
    bp = ax.boxplot(data, tick_labels=[f"{c:.1f}" for c in coefs],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax.axhline(0, ls="--", color="red", alpha=0.7, label="No change")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Δ Target Sum Log-Prob")
    ax.set_title(f"{label} | CF Layer {layer}: Per-Question Δ Target Log-Prob")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def cf_per_question_heatmap(cf_wide, label, layer, max_questions=60):
    col = "delta_target_sum_lp"
    if col not in cf_wide.columns:
        return None
    sub = cf_wide[cf_wide["layer"] == layer]
    if sub.empty:
        return None
    q_ids = sorted(sub["question_id"].unique())[:max_questions]
    sub = sub[sub["question_id"].isin(q_ids)]
    try:
        pivot = sub.pivot(index="question_id", columns="coef", values=col).fillna(0.0)
    except Exception:
        return None
    vmax = np.nanmax(np.abs(pivot.values)) or 1.0

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.6),
                                    max(6, len(q_ids) * 0.22)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Question ID")
    ax.set_title(f"{label} | CF Layer {layer}: Per-Question Δ Target Sum Log-Prob")
    plt.colorbar(im, ax=ax, label="Δ log-prob")
    plt.tight_layout()
    return fig


def cf_improved_hurt_bar(cf_wide, label, layer):
    col = "delta_target_sum_lp"
    if col not in cf_wide.columns:
        return None
    sub = cf_wide[(cf_wide["layer"] == layer) & (cf_wide["coef"] != 0.0)]
    if sub.empty:
        return None
    coefs = sorted(sub["coef"].unique())
    improved  = [(sub[sub["coef"] == c][col] > 0).mean() for c in coefs]
    hurt      = [(sub[sub["coef"] == c][col] < 0).mean() for c in coefs]
    unchanged = [max(0.0, 1.0 - i - h) for i, h in zip(improved, hurt)]

    labels = [f"{c:.1f}" for c in coefs]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.bar(x, improved, label="Improved", color="#2ecc71", alpha=0.85)
    ax.bar(x, unchanged, bottom=improved, label="Unchanged", color="#bdc3c7", alpha=0.6)
    ax.bar(x, hurt, bottom=[i + u for i, u in zip(improved, unchanged)],
           label="Hurt", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title(f"{label} | CF Layer {layer}: Steering Impact (target sum log-prob)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def cf_target_rank_stacked_bar(cf_wide, label, layer):
    col = "target_rank_sum"
    if col not in cf_wide.columns:
        return None
    sub = cf_wide[cf_wide["layer"] == layer]
    if sub.empty:
        return None
    coefs = sorted(sub["coef"].unique())
    rank_colors = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(max(10, len(coefs) * 0.8), 5))
    x = np.arange(len(coefs))
    bottom = np.zeros(len(coefs))
    for rank, color in zip([1, 2, 3, 4], rank_colors):
        vals = [(sub[sub["coef"] == c][col] == rank).mean() for c in coefs]
        ax.bar(x, vals, bottom=bottom, label=f"Rank {rank}", color=color, alpha=0.85)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.1f}" for c in coefs], rotation=45, ha="right")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title(f"{label} | CF Layer {layer}: Target Rank Distribution (sum scoring)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def cf_margin_distribution(cf_wide, label, layer):
    col = "delta_margin_sum"
    if col not in cf_wide.columns:
        return None
    sub = cf_wide[(cf_wide["layer"] == layer) & (cf_wide["coef"] != 0.0)]
    if sub.empty:
        return None
    coefs = sorted(sub["coef"].unique())
    data = [sub[sub["coef"] == c][col].dropna().values for c in coefs]

    fig, ax = plt.subplots(figsize=(max(10, len(coefs) * 0.7), 5))
    bp = ax.boxplot(data, tick_labels=[f"{c:.1f}" for c in coefs],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgreen")
        patch.set_alpha(0.7)
    ax.axhline(0, ls="--", color="red", alpha=0.7, label="No change")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Δ Margin (target − best wrong)")
    ax.set_title(f"{label} | CF Layer {layer}: Per-Question Δ Margin Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# =============================================================================
# Joint MCF + CF plots
# =============================================================================

def joint_best_layer_comparison(mcf_summary, cf_summary, label, cf_scoring="sum"):
    cf_col = f"accuracy_{cf_scoring}"
    if cf_col not in cf_summary.columns:
        return None
    mcf_best = mcf_summary[mcf_summary["coef"] != 0.0].groupby("layer")["accuracy"].max()
    cf_best  = cf_summary[cf_summary["coef"] != 0.0].groupby("layer")[cf_col].max()
    mcf_base = _baseline_series(mcf_summary, "accuracy")
    cf_base  = _baseline_series(cf_summary, cf_col)

    layers = sorted(set(mcf_best.index) | set(cf_best.index))
    x = np.arange(len(layers))
    w = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5 * w, [mcf_best.get(l, np.nan) for l in layers], w,
           label="MCF steered", color="steelblue", alpha=0.9)
    ax.bar(x - 0.5 * w, [mcf_base.get(l, np.nan) for l in layers], w,
           label="MCF baseline", color="steelblue", alpha=0.35)
    ax.bar(x + 0.5 * w, [cf_best.get(l, np.nan) for l in layers], w,
           label=f"CF ({cf_scoring}) steered", color="darkorange", alpha=0.9)
    ax.bar(x + 1.5 * w, [cf_base.get(l, np.nan) for l in layers], w,
           label=f"CF ({cf_scoring}) baseline", color="darkorange", alpha=0.35)
    ax.axhline(0.25, ls=":", color="gray", alpha=0.4, label="Random (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{label} | MCF vs CF: Best Steered Accuracy per Layer")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def joint_improvement_scatter(mcf_summary, cf_summary, label, cf_scoring="sum"):
    cf_col = f"accuracy_{cf_scoring}"
    if cf_col not in cf_summary.columns:
        return None
    mcf_base = _baseline_series(mcf_summary, "accuracy")
    cf_base  = _baseline_series(cf_summary, cf_col)
    mcf_best = mcf_summary[mcf_summary["coef"] != 0.0].groupby("layer")["accuracy"].max()
    cf_best  = cf_summary[cf_summary["coef"] != 0.0].groupby("layer")[cf_col].max()

    layers = sorted(set(mcf_best.index) & set(cf_best.index))
    mcf_imp = [mcf_best.get(l, np.nan) - mcf_base.get(l, 0.0) for l in layers]
    cf_imp  = [cf_best.get(l, np.nan) - cf_base.get(l, 0.0) for l in layers]

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(mcf_imp, cf_imp, c=layers, cmap="viridis", s=70,
                    edgecolors="black", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Layer")
    for i, layer in enumerate(layers):
        ax.annotate(f"L{layer}", (mcf_imp[i], cf_imp[i]), fontsize=7, alpha=0.8,
                    textcoords="offset points", xytext=(4, 4))
    finite_vals = [v for v in mcf_imp + cf_imp if np.isfinite(v)]
    if finite_vals:
        lo, hi = min(finite_vals) - 0.01, max(finite_vals) + 0.01
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, label="y = x")
    ax.axhline(0, ls=":", color="gray", alpha=0.4)
    ax.axvline(0, ls=":", color="gray", alpha=0.4)
    ax.set_xlabel("MCF Accuracy Improvement")
    ax.set_ylabel(f"CF ({cf_scoring}) Accuracy Improvement")
    ax.set_title(f"{label} | Joint: Accuracy Improvement per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    return fig


def joint_best_coef_alignment(mcf_summary, cf_summary, label, cf_scoring="sum"):
    cf_col = f"accuracy_{cf_scoring}"
    if cf_col not in cf_summary.columns:
        return None
    mcf_best_coef = mcf_summary[mcf_summary["coef"] != 0.0].loc[
        mcf_summary[mcf_summary["coef"] != 0.0].groupby("layer")["accuracy"].idxmax()
    ].set_index("layer")["coef"]
    cf_best_coef = cf_summary[cf_summary["coef"] != 0.0].loc[
        cf_summary[cf_summary["coef"] != 0.0].groupby("layer")[cf_col].idxmax()
    ].set_index("layer")["coef"]

    layers = sorted(set(mcf_best_coef.index) & set(cf_best_coef.index))
    if not layers:
        return None
    mcf_vals = [mcf_best_coef.get(l, np.nan) for l in layers]
    cf_vals  = [cf_best_coef.get(l, np.nan) for l in layers]

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(mcf_vals, cf_vals, c=layers, cmap="viridis", s=70,
                    edgecolors="black", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Layer")
    for i, layer in enumerate(layers):
        ax.annotate(f"L{layer}", (mcf_vals[i], cf_vals[i]), fontsize=7, alpha=0.8,
                    textcoords="offset points", xytext=(4, 4))
    finite_vals = [v for v in mcf_vals + cf_vals if np.isfinite(v)]
    if finite_vals:
        lo, hi = min(finite_vals) - 0.5, max(finite_vals) + 0.5
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, label="y = x")
    ax.set_xlabel("MCF Best Coefficient")
    ax.set_ylabel(f"CF ({cf_scoring}) Best Coefficient")
    ax.set_title(f"{label} | Joint: Best Coefficient Agreement (MCF vs CF)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Show all visualizations for one experiment result directory."
    )
    parser.add_argument("base_dir", help="Path to experiment output directory")
    parser.add_argument("--out-dir", default=None,
                        help="Where to save plots (default: <base_dir>/plots)")
    parser.add_argument("--top-layers", type=int, default=5,
                        help="Number of top layers for highlighted line plots (default: 5)")
    parser.add_argument("--best-layer", type=int, default=None,
                        help="Override which layer to use for per-layer plots")
    parser.add_argument("--cf-scoring", default="sum",
                        choices=["sum", "mean", "char", "pmi"],
                        help="CF scoring method for joint plots (default: sum)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Loading data from: {base_dir}")
    print(f"{'='*60}")
    data = load_data(base_dir)
    label = _exp_label(data)

    mcf_results = data["mcf_results"]
    mcf_summary = data["mcf_summary"]
    cf_summary  = data["cf_summary"]
    cf_wide     = data["cf_wide"]

    if mcf_summary is None and cf_summary is None:
        print("No usable data found.", file=sys.stderr)
        sys.exit(1)

    exp_dir = data["exp_dir"]
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    PRIORITY_PLOTS = {
        "MCF_accuracy_heatmap",
        "MCF_best_coef_per_layer",
        "MCF_pos_neg_asymmetry",
        "CF_sum_accuracy_heatmap",
        "JOINT_best_layer_MCF_vs_CF",
    }

    counter = [0]

    def show(fig, name: str):
        if fig is None:
            return
        counter[0] += 1
        fname = f"{counter[0]:02d}_{name}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        tag = "★ " if name in PRIORITY_PLOTS else "  "
        print(f"  {tag}[{counter[0]:02d}] {fname}")
        if name in PRIORITY_PLOTS:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Numerical tables
    # ------------------------------------------------------------------
    if mcf_summary is not None:
        print_mcf_numbers(mcf_summary, label, results=mcf_results)

    if cf_summary is not None:
        print_cf_numbers(cf_summary, label)

    # ------------------------------------------------------------------
    # MCF cross-layer
    # ------------------------------------------------------------------
    if mcf_summary is not None:
        print(f"\n--- MCF cross-layer ---")
        mcf_top = _best_layers(mcf_summary, "accuracy", args.top_layers)
        best_mcf_layer = args.best_layer if args.best_layer is not None \
            else (mcf_top[0] if mcf_top else 0)

        show(mcf_accuracy_heatmap(mcf_summary, label),          "MCF_accuracy_heatmap")
        show(mcf_delta_logprob_heatmap(mcf_summary, label),     "MCF_delta_logprob_heatmap")
        show(mcf_pct_improved_heatmap(mcf_summary, label),      "MCF_pct_improved_heatmap")
        show(mcf_pct_hurt_heatmap(mcf_summary, label),          "MCF_pct_hurt_heatmap")
        show(mcf_rank_heatmap(mcf_summary, label),               "MCF_rank_heatmap")
        show(mcf_best_coef_per_layer(mcf_summary, label),        "MCF_best_coef_per_layer")
        show(mcf_max_accuracy_change(mcf_summary, label),        "MCF_max_accuracy_change")
        show(mcf_pos_neg_asymmetry(mcf_summary, label),          "MCF_pos_neg_asymmetry")
        show(mcf_accuracy_lines(mcf_summary, label, highlight_layers=mcf_top),
             "MCF_accuracy_lines_top_layers")
        show(mcf_accuracy_lines(mcf_summary, label, highlight_layers=None),
             "MCF_accuracy_lines_all")

        print(f"\n--- MCF per-layer (layer {best_mcf_layer}) ---")
        if mcf_results is not None:
            show(mcf_delta_distribution(mcf_results, label, best_mcf_layer),
                 f"MCF_L{best_mcf_layer}_delta_distribution")
            show(mcf_rank_change_distribution(mcf_results, label, best_mcf_layer),
                 f"MCF_L{best_mcf_layer}_rank_change_distribution")
            show(mcf_per_question_heatmap(mcf_results, label, best_mcf_layer),
                 f"MCF_L{best_mcf_layer}_per_question_heatmap")
        show(mcf_improved_hurt_bar(mcf_summary, label, best_mcf_layer),
             f"MCF_L{best_mcf_layer}_improved_hurt")

        for layer in mcf_top[1:3]:
            show(mcf_improved_hurt_bar(mcf_summary, label, layer),
                 f"MCF_L{layer}_improved_hurt")
            if mcf_results is not None:
                show(mcf_delta_distribution(mcf_results, label, layer),
                     f"MCF_L{layer}_delta_distribution")

    # ------------------------------------------------------------------
    # CF cross-layer
    # ------------------------------------------------------------------
    if cf_summary is not None:
        print(f"\n--- CF cross-layer ---")
        cf_top = _best_layers(cf_summary, "accuracy_sum", args.top_layers)
        best_cf_layer = args.best_layer if args.best_layer is not None \
            else (cf_top[0] if cf_top else 0)

        for scoring in ["sum", "char", "mean"]:
            show(cf_accuracy_heatmap(cf_summary, label, scoring=scoring),
                 f"CF_{scoring}_accuracy_heatmap")

        show(cf_delta_heatmap(cf_summary, label),                       "CF_delta_sum_logprob_heatmap")
        show(cf_pct_improved_heatmap(cf_summary, label, scoring="sum"), "CF_pct_improved_heatmap_sum")
        show(cf_pct_improved_heatmap(cf_summary, label, scoring="char"),"CF_pct_improved_heatmap_char")
        show(cf_rank_heatmap(cf_summary, label, scoring="sum"),         "CF_rank_heatmap_sum")
        show(cf_margin_heatmap(cf_summary, label),                      "CF_margin_heatmap")
        show(cf_best_coef_per_layer(cf_summary, label, scoring="sum"),  "CF_best_coef_per_layer_sum")
        show(cf_best_coef_per_layer(cf_summary, label, scoring="char"), "CF_best_coef_per_layer_char")
        show(cf_multi_scoring_comparison(cf_summary, label),            "CF_multi_scoring_comparison")
        show(cf_delta_max_per_layer(cf_summary, label),                 "CF_delta_max_per_layer")
        show(cf_accuracy_lines(cf_summary, label, scoring="sum", highlight_layers=cf_top),
             "CF_accuracy_lines_sum_top_layers")
        show(cf_accuracy_lines(cf_summary, label, scoring="sum", highlight_layers=None),
             "CF_accuracy_lines_sum_all")

        if cf_wide is not None:
            print(f"\n--- CF per-layer (layer {best_cf_layer}) ---")
            show(cf_delta_distribution(cf_wide, label, best_cf_layer),
                 f"CF_L{best_cf_layer}_delta_distribution")
            show(cf_per_question_heatmap(cf_wide, label, best_cf_layer),
                 f"CF_L{best_cf_layer}_per_question_heatmap")
            show(cf_improved_hurt_bar(cf_wide, label, best_cf_layer),
                 f"CF_L{best_cf_layer}_improved_hurt")
            show(cf_target_rank_stacked_bar(cf_wide, label, best_cf_layer),
                 f"CF_L{best_cf_layer}_target_rank_stacked")
            show(cf_margin_distribution(cf_wide, label, best_cf_layer),
                 f"CF_L{best_cf_layer}_margin_distribution")

            for layer in cf_top[1:3]:
                show(cf_improved_hurt_bar(cf_wide, label, layer),
                     f"CF_L{layer}_improved_hurt")
                show(cf_delta_distribution(cf_wide, label, layer),
                     f"CF_L{layer}_delta_distribution")

    # ------------------------------------------------------------------
    # Joint MCF + CF
    # ------------------------------------------------------------------
    if mcf_summary is not None and cf_summary is not None:
        print(f"\n--- Joint MCF + CF ---")
        show(joint_best_layer_comparison(mcf_summary, cf_summary, label,
                                          cf_scoring=args.cf_scoring),
             "JOINT_best_layer_MCF_vs_CF")
        show(joint_improvement_scatter(mcf_summary, cf_summary, label,
                                        cf_scoring=args.cf_scoring),
             "JOINT_improvement_scatter")
        show(joint_best_coef_alignment(mcf_summary, cf_summary, label,
                                        cf_scoring=args.cf_scoring),
             "JOINT_best_coef_alignment")

    print(f"\n{'='*60}")
    print(f"Saved {counter[0]} plots to: {out_dir}")
    print(f"(★ = shown interactively, rest saved only)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
