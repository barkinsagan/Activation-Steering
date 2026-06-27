#!/usr/bin/env python3
"""
analyze.py — Unified steering-experiment analysis.

One script, all the plots. Merges:
  • analysis_dashboard.py — effect-magnitude / asymmetry / rank / per-question
  • compare.py            — cross-experiment comparison
  • phase0_margin.py      — margin (logit-gap) analysis
  • diagnose_negative_coefs.py — negative-coef diagnostics
  • NEW continuous-metric analyses for target-aware effect detection:
        - Δ-correct vs Δ-best-wrong overlay
        - Selectivity index = (Δ-correct − Δ-best-wrong) / (|Δ-correct| + |Δ-best-wrong|)
        - KL(steered ‖ baseline) over the 4-choice distribution
        - Paired Wilcoxon test on per-question (Δ-correct − Δ-best-wrong)

Sections (controlled by --section):
  continuous   continuous-metric plots (PRIMARY for current research)
  effect       layer × coef effect-magnitude (heatmaps, layer/coef sweeps)
  asymmetry    direction & asymmetry (pos vs neg coef, coef=0 drift)
  rank         ranking behavior (rank violins, % rank-1)
  question     per-question structure (steerable/resistant, scatter)
  cross-method MCF vs CF agreement
  margin       margin / logit-gap analysis (phase0_margin.py)
  negcoef      negative-coef diagnostics (diagnose_negative_coefs.py)
  compare      multi-experiment comparison (compare.py); pass ≥2 experiments
  all          everything (default)

Usage
-----
  # single experiment, all plots, show only
  python analysis/analyze.py results/exp_20260626_phys_vs_bio_continuous_full

  # single experiment, save plots
  python analysis/analyze.py results/exp_20260626_phys_vs_bio_continuous_full --save

  # only continuous-metric section
  python analysis/analyze.py results/exp_… --section continuous --save

  # multi-experiment comparison
  python analysis/analyze.py results/exp_a results/exp_b --section compare --save

Outputs
-------
  Plots go to {exp_root}/analysis/ by default, or --out-dir if specified.
  Per-experiment numerical summaries written as CSVs alongside plots.
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="tab10")
except ImportError:
    sns = None

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ─────────────────────────────────────────────────────────────────────────────
# Defaults / constants
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_LAYERS = 5
TOP_K_QUESTIONS = 10
FOCUS_COEFS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
DRIFT_THRESHOLD = 0.05

# Module-level save state — set by run() based on CLI flags
_OUT_DIR: Optional[Path] = None
_SHOW: bool = True
_SUMMARY_ROWS: Dict[str, pd.DataFrame] = {}   # name → df, saved at end


def _save(fig, name: str):
    """Save (and/or show) a figure according to current run config."""
    if _OUT_DIR is not None:
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(_OUT_DIR / f"{name}.png")
        print(f"  saved {name}.png")
    if _SHOW:
        plt.show()
    plt.close(fig)


def _save_summary(name: str, df: pd.DataFrame):
    """Stash a numerical summary; written at end of run()."""
    _SUMMARY_ROWS[name] = df


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_mcf(exp_root: Path) -> pd.DataFrame:
    """Load all MCF per-question, per-(layer, coef) rows."""
    dfs = []
    for f in sorted(glob.glob(str(exp_root / "mcf" / "layer_*_results.csv"))):
        layer = int(Path(f).stem.split("layer_")[1].split("_")[0])
        df = pd.read_csv(f)
        df["layer"] = layer
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_cf_wide(exp_root: Path) -> pd.DataFrame:
    """Load all CF per-question, per-(layer, coef) rows from detailed_wide.csv."""
    dfs = []
    cf_dir = exp_root / "cf"
    if not cf_dir.exists():
        return pd.DataFrame()
    layer_dirs = sorted(
        [d for d in cf_dir.iterdir() if d.is_dir() and d.name.startswith("layer_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for d in layer_dirs:
        wide = d / "detailed_wide.csv"
        if not wide.exists():
            warnings.warn(f"Missing detailed_wide.csv for {d.name} — skipping")
            continue
        df = pd.read_csv(wide)
        df["layer"] = int(d.name.split("_")[1])
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def attach_split(df: pd.DataFrame, exp_root: Path) -> pd.DataFrame:
    """If a split_manifest.csv exists, attach the 'split' column to per-question data."""
    manifest_path = exp_root / "split_manifest.csv"
    if not manifest_path.exists() or df.empty or "question_id" not in df.columns:
        return df
    manifest = pd.read_csv(manifest_path)
    # eval_question_id is 0-based; question_id should match
    return df.merge(
        manifest[["eval_question_id", "split"]].rename(columns={"eval_question_id": "question_id"}),
        on="question_id", how="left",
    )


# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def agg_mcf(mcf: pd.DataFrame) -> pd.DataFrame:
    """Per-(layer, coef) MCF summary."""
    g = mcf.groupby(["layer", "coef"])
    return g.agg(
        accuracy=("correct", "mean"),
        mean_delta_correct=("delta_correct_logprob", "mean"),
        std_delta_correct=("delta_correct_logprob", "std"),
        pct_improved=("delta_correct_logprob", lambda x: (x > 0).mean()),
        pct_hurt=("delta_correct_logprob", lambda x: (x < 0).mean()),
        mean_rank_change=("rank_change", "mean"),
        pct_rank1=("correct_label_rank", lambda x: (x == 1).mean()),
        n=("correct", "count"),
    ).reset_index()


def agg_cf_continuous(cf: pd.DataFrame) -> pd.DataFrame:
    """Per-(layer, coef) CF aggregates including the new continuous metrics.

    Key new columns:
      mean_delta_correct      mean Δlogprob of correct (target) answer
      mean_delta_best_wrong   mean Δlogprob of the best (highest) wrong answer
      mean_delta_margin       mean Δ(margin) — equivalent to logit-gap shift
      selectivity_index       (Δcorrect − Δbest_wrong) / (|Δcorrect| + |Δbest_wrong|)
      mean_kl_baseline        mean KL(steered ‖ baseline) over the 4-choice softmax
      wilcoxon_p              paired Wilcoxon p-value testing
                              H0: (Δcorrect − Δbest_wrong) = 0
    """
    # Sanity: required columns
    needed = ["delta_target_sum_lp", "delta_max_wrong_sum_lp",
              "delta_margin_sum",
              "target_sum_lp", "false1_sum_lp", "false2_sum_lp", "false3_sum_lp"]
    missing = [c for c in needed if c not in cf.columns]
    if missing:
        warnings.warn(f"CF data missing columns {missing}; continuous metrics will be incomplete")

    # ── Per-row continuous quantities ────────────────────────────────────────
    cf = cf.copy()
    if "delta_target_sum_lp" in cf and "delta_max_wrong_sum_lp" in cf:
        cf["selectivity"] = cf["delta_target_sum_lp"] - cf["delta_max_wrong_sum_lp"]

    # ── KL divergence per question per (layer, coef) ──────────────────────────
    # Compute steered distribution (softmax over the 4 sum_lps) and
    # baseline distribution (sum_lps at coef=0 for the same question/layer).
    if all(c in cf.columns for c in ["target_sum_lp", "false1_sum_lp",
                                      "false2_sum_lp", "false3_sum_lp"]):
        steered_lps = cf[["target_sum_lp", "false1_sum_lp",
                          "false2_sum_lp", "false3_sum_lp"]].values
        # softmax over rows
        m = steered_lps.max(axis=1, keepdims=True)
        exp = np.exp(steered_lps - m)
        steered_p = exp / exp.sum(axis=1, keepdims=True)

        # Build a baseline lookup: (layer, question_id) → 4-vector at coef=0
        if "question_id" in cf.columns:
            baseline = cf[cf.coef == 0.0].copy()
            baseline = baseline.drop_duplicates(["layer", "question_id"])
            baseline_lp = baseline.set_index(["layer", "question_id"])[
                ["target_sum_lp", "false1_sum_lp", "false2_sum_lp", "false3_sum_lp"]
            ].values
            baseline_idx = baseline.set_index(["layer", "question_id"]).index

            # softmax of baseline
            mb = baseline_lp.max(axis=1, keepdims=True)
            exp_b = np.exp(baseline_lp - mb)
            base_p = exp_b / exp_b.sum(axis=1, keepdims=True)

            # Map each cf row to its baseline distribution
            row_key = list(zip(cf["layer"].values, cf["question_id"].values))
            base_lookup = {k: base_p[i] for i, k in enumerate(baseline_idx.tolist())}
            base_for_row = np.array([base_lookup.get(k, np.full(4, 0.25)) for k in row_key])

            # KL(steered || baseline) per row, in nats
            kl = (steered_p * (np.log(steered_p + 1e-30) - np.log(base_for_row + 1e-30))).sum(axis=1)
            cf["kl_to_baseline"] = kl
        else:
            cf["kl_to_baseline"] = np.nan
    else:
        cf["kl_to_baseline"] = np.nan

    # ── Aggregate ────────────────────────────────────────────────────────────
    g = cf.groupby(["layer", "coef"])
    agg = g.agg(
        accuracy_sum=("correct_sum", "mean") if "correct_sum" in cf else ("layer", "size"),
        accuracy_char=("correct_char", "mean") if "correct_char" in cf else ("layer", "size"),
        mean_delta_correct=("delta_target_sum_lp", "mean"),
        std_delta_correct=("delta_target_sum_lp", "std"),
        mean_delta_best_wrong=("delta_max_wrong_sum_lp", "mean"),
        std_delta_best_wrong=("delta_max_wrong_sum_lp", "std"),
        mean_delta_margin=("delta_margin_sum", "mean"),
        std_delta_margin=("delta_margin_sum", "std"),
        mean_kl_baseline=("kl_to_baseline", "mean"),
        pct_improved=("delta_target_sum_lp", lambda x: (x > 0).mean()),
        pct_hurt=("delta_target_sum_lp", lambda x: (x < 0).mean()),
        mean_rank_change=("rank_change_sum", "mean") if "rank_change_sum" in cf else ("layer", "size"),
        pct_rank1=("target_rank_sum", lambda x: (x == 1).mean()) if "target_rank_sum" in cf else ("layer", "size"),
        n=("delta_target_sum_lp", "count"),
    ).reset_index()

    # Selectivity index: derived from aggregated mean deltas (signed, magnitude-normalized)
    denom = agg["mean_delta_correct"].abs() + agg["mean_delta_best_wrong"].abs()
    agg["selectivity_index"] = np.where(
        denom > 1e-9,
        (agg["mean_delta_correct"] - agg["mean_delta_best_wrong"]) / denom,
        0.0,
    )

    # ── Paired Wilcoxon on selectivity per (layer, coef) ──────────────────────
    try:
        from scipy.stats import wilcoxon
        pvals = []
        for (layer, coef), grp in g:
            if coef == 0.0 or "selectivity" not in grp.columns or len(grp) < 5:
                pvals.append({"layer": layer, "coef": coef, "wilcoxon_p": np.nan})
                continue
            vals = grp["selectivity"].dropna().values
            if len(vals) < 5 or np.allclose(vals, 0):
                pvals.append({"layer": layer, "coef": coef, "wilcoxon_p": np.nan})
                continue
            try:
                _, p = wilcoxon(vals, alternative="greater")
                pvals.append({"layer": layer, "coef": coef, "wilcoxon_p": p})
            except Exception:
                pvals.append({"layer": layer, "coef": coef, "wilcoxon_p": np.nan})
        agg = agg.merge(pd.DataFrame(pvals), on=["layer", "coef"], how="left")
    except ImportError:
        warnings.warn("scipy not available — skipping Wilcoxon tests")
        agg["wilcoxon_p"] = np.nan

    return agg


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: CONTINUOUS METRICS (NEW — primary for current research)
# ═════════════════════════════════════════════════════════════════════════════

def plot_continuous_section(cf_agg: pd.DataFrame):
    """All the new continuous-metric plots in one go."""
    if cf_agg.empty:
        print("  [continuous] no CF data; skipping")
        return

    _plot_correct_vs_wrong_overlay(cf_agg)
    _plot_selectivity_heatmap(cf_agg)
    _plot_margin_heatmap(cf_agg)
    _plot_kl_heatmap(cf_agg)
    _plot_selectivity_layer_sweep(cf_agg)
    _plot_significance_grid(cf_agg)
    _save_summary("continuous_agg", cf_agg)


def _plot_correct_vs_wrong_overlay(agg: pd.DataFrame):
    """For each layer-band, plot Δ-correct and Δ-best-wrong on the same axes vs coef.

    The decisive target-aware-effect plot. If the two curves diverge with correct
    above wrong, steering is selectively helping the correct answer. If they
    track each other, steering is just shifting the answer distribution overall.
    """
    layers = sorted(agg["layer"].unique())
    if not layers:
        return

    # 4 representative layers spread across depth
    pick = [layers[i] for i in np.linspace(0, len(layers) - 1, min(4, len(layers))).astype(int)]
    fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 4), sharey=True)
    if len(pick) == 1:
        axes = [axes]
    for ax, layer in zip(axes, pick):
        sub = agg[agg.layer == layer].sort_values("coef")
        ax.plot(sub.coef, sub.mean_delta_correct, marker="o", label="Δ correct",
                color="tab:green", linewidth=2)
        ax.plot(sub.coef, sub.mean_delta_best_wrong, marker="s", label="Δ best wrong",
                color="tab:red", linewidth=2)
        ax.fill_between(sub.coef, sub.mean_delta_correct, sub.mean_delta_best_wrong,
                        alpha=0.15, color="tab:blue")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.axvline(0, color="black", lw=0.6, ls="--")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Coefficient")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean Δ logprob")
    fig.suptitle("Target-aware effect: Δ-correct vs Δ-best-wrong by coefficient (selected layers)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "20_continuous_correct_vs_wrong_overlay")


def _plot_selectivity_heatmap(agg: pd.DataFrame):
    """layer × coef heatmap of (Δ-correct − Δ-best-wrong) / |sum|.

    Positive = steering helps correct more than wrong (target-aware).
    Near zero = effect is symmetric (probability redistribution).
    Negative = wrong is helped more (anti-target).
    """
    pivot = agg.pivot_table(index="coef", columns="layer", values="selectivity_index")
    fig, ax = plt.subplots(figsize=(16, 6))
    if sns is not None:
        sns.heatmap(pivot, ax=ax, center=0, cmap="RdBu_r", linewidths=0.3,
                    cbar_kws={"label": "Selectivity index"})
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                       vmin=-pivot.abs().values.max(), vmax=pivot.abs().values.max())
        fig.colorbar(im, ax=ax, label="Selectivity index")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    ax.set_title("Selectivity (Δcorrect − Δbest-wrong) / |sum|  —  positive = target-aware steering")
    ax.set_xlabel("Layer"); ax.set_ylabel("Coefficient")
    _save(fig, "21_continuous_selectivity_heatmap")


def _plot_margin_heatmap(agg: pd.DataFrame):
    """layer × coef heatmap of Δ-margin (logit gap between correct and best-wrong)."""
    pivot = agg.pivot_table(index="coef", columns="layer", values="mean_delta_margin")
    fig, ax = plt.subplots(figsize=(16, 6))
    if sns is not None:
        sns.heatmap(pivot, ax=ax, center=0, cmap="RdBu_r", linewidths=0.3,
                    cbar_kws={"label": "Δ margin (correct − best-wrong) logprob"})
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", center=0)
        fig.colorbar(im, ax=ax)
    ax.set_title("Δ margin (logit gap between correct and best-wrong) by layer × coef")
    ax.set_xlabel("Layer"); ax.set_ylabel("Coefficient")
    _save(fig, "22_continuous_margin_heatmap")


def _plot_kl_heatmap(agg: pd.DataFrame):
    """layer × coef heatmap of mean KL(steered ‖ baseline) — intervention magnitude."""
    if agg["mean_kl_baseline"].isna().all():
        return
    pivot = agg.pivot_table(index="coef", columns="layer", values="mean_kl_baseline")
    fig, ax = plt.subplots(figsize=(16, 6))
    if sns is not None:
        sns.heatmap(pivot, ax=ax, cmap="viridis", linewidths=0.3,
                    cbar_kws={"label": "Mean KL(steered ‖ baseline), nats"})
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax)
    ax.set_title("KL divergence from baseline — how much steering moves the answer distribution")
    ax.set_xlabel("Layer"); ax.set_ylabel("Coefficient")
    _save(fig, "23_continuous_kl_heatmap")


def _plot_selectivity_layer_sweep(agg: pd.DataFrame):
    """x=layer, y=selectivity, lines for FOCUS_COEFS — picks out best layer band."""
    sub = agg[agg.coef.isin(FOCUS_COEFS)].sort_values("layer")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    for coef, grp in sub.groupby("coef"):
        ax.plot(grp.layer, grp.selectivity_index, marker="o", markersize=4, label=f"{coef:+.2f}")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("Selectivity index")
    ax.set_title("Selectivity by layer — which layers steer target-awarely?")
    ax.legend(title="Coef", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    _save(fig, "24_continuous_selectivity_layer_sweep")


def _plot_significance_grid(agg: pd.DataFrame):
    """layer × coef grid showing Wilcoxon p-value bins.

    Cell shows 'p<0.01', 'p<0.05', or '' (n.s.) if available.
    """
    if "wilcoxon_p" not in agg.columns or agg["wilcoxon_p"].isna().all():
        return
    pivot = agg.pivot_table(index="coef", columns="layer", values="wilcoxon_p")
    fig, ax = plt.subplots(figsize=(16, 6))
    # Use −log10(p) as the color, mask non-significant
    nlog = -np.log10(pivot.fillna(1.0))
    if sns is not None:
        sns.heatmap(nlog, ax=ax, cmap="YlOrRd", linewidths=0.3,
                    cbar_kws={"label": "−log10(p)  —  paired Wilcoxon on (Δcorrect − Δbest-wrong)"})
    else:
        im = ax.imshow(nlog.values, aspect="auto", cmap="YlOrRd")
        fig.colorbar(im, ax=ax)
    ax.set_title("Statistical significance of target-aware steering (Wilcoxon, alt=greater)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Coefficient")
    _save(fig, "25_continuous_significance_grid")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: EFFECT MAGNITUDE  (carried from analysis_dashboard.py)
# ═════════════════════════════════════════════════════════════════════════════

def plot_effect_section(mcf_agg: pd.DataFrame, cf_agg: pd.DataFrame):
    for agg, mode in [(mcf_agg, "MCF"), (cf_agg, "CF")]:
        if agg.empty:
            continue
        delta_col = "mean_delta_correct" if mode == "MCF" else "mean_delta_correct"
        _plot_heatmap(agg, mode, delta_col)
        _plot_layer_sweep(agg, mode, delta_col)
        _plot_coef_sweep(agg, mode, delta_col)
        _plot_best_layer_bar(agg, mode, delta_col)


def _plot_heatmap(agg: pd.DataFrame, mode: str, val_col: str):
    pivot = agg.pivot_table(index="coef", columns="layer", values=val_col)
    fig, ax = plt.subplots(figsize=(16, 8))
    if sns is not None:
        sns.heatmap(pivot, ax=ax, center=0, cmap="RdBu_r", linewidths=0.3,
                    cbar_kws={"label": "Mean Δ correct logprob"})
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", center=0)
        fig.colorbar(im, ax=ax)
    ax.set_title(f"[{mode}] Layer × Coef — Mean Δ correct logprob")
    ax.set_xlabel("Layer"); ax.set_ylabel("Coefficient")
    _save(fig, f"01_heatmap_{mode}")


def _plot_layer_sweep(agg: pd.DataFrame, mode: str, val_col: str):
    sub = agg[agg.coef.isin(FOCUS_COEFS)].sort_values("layer")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    for coef, grp in sub.groupby("coef"):
        ax.plot(grp.layer, grp[val_col], marker="o", markersize=4, label=f"{coef:+.2f}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Δ logprob")
    ax.set_title(f"[{mode}] Layer Sweep — Mean Δ correct logprob by Coefficient")
    ax.legend(title="Coef", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    _save(fig, f"02_layer_sweep_{mode}")


def _plot_coef_sweep(agg: pd.DataFrame, mode: str, val_col: str):
    # Pick top-K layers by |effect|
    layer_effect = agg.groupby("layer")[val_col].apply(lambda x: x.abs().max())
    top_layers = layer_effect.nlargest(TOP_K_LAYERS).index.tolist()
    sub = agg[agg.layer.isin(top_layers)].sort_values("coef")
    fig, ax = plt.subplots(figsize=(11, 5))
    for layer, grp in sub.groupby("layer"):
        ax.plot(grp.coef, grp[val_col], marker="o", markersize=4, label=f"L{layer}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Coefficient"); ax.set_ylabel("Mean Δ logprob")
    ax.set_title(f"[{mode}] Coef Sweep — Top {TOP_K_LAYERS} layers by |effect|")
    ax.legend(title="Layer", fontsize=8)
    _save(fig, f"03_coef_sweep_{mode}")


def _plot_best_layer_bar(agg: pd.DataFrame, mode: str, val_col: str):
    # Best (layer, coef) per layer, ranked
    best_per_layer = agg.loc[agg.groupby("layer")[val_col].idxmax()]
    top = best_per_layer.nlargest(TOP_K_LAYERS, val_col)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top["layer"].astype(str) + f"\n(c={top['coef'].round(2).astype(str)})",
           top[val_col], yerr=top.get("std_delta_correct", 0))
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_ylabel("Mean Δ correct logprob")
    ax.set_title(f"[{mode}] Top {TOP_K_LAYERS} (layer, coef) by mean effect")
    fig.tight_layout()
    _save(fig, f"04_best_layer_bar_{mode}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: ASYMMETRY  (direction & sign symmetry)
# ═════════════════════════════════════════════════════════════════════════════

def plot_asymmetry_section(mcf_agg: pd.DataFrame, cf_agg: pd.DataFrame):
    for agg, mode in [(mcf_agg, "MCF"), (cf_agg, "CF")]:
        if agg.empty:
            continue
        _plot_improved_hurt(agg, mode)
        _plot_asymmetry(agg, mode)
        _plot_coef0_drift(agg, mode)


def _plot_improved_hurt(agg: pd.DataFrame, mode: str):
    sub = agg[agg.coef != 0.0].copy()
    sub["pct_neutral"] = 1 - sub["pct_improved"] - sub["pct_hurt"]
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot = sub.pivot_table(index="layer", columns="coef",
                            values=["pct_improved", "pct_hurt"]).fillna(0)
    pos = pivot["pct_improved"].mean(axis=1)
    neg = pivot["pct_hurt"].mean(axis=1)
    layers = pos.index.tolist()
    ax.bar(layers, pos, label="% improved", color="tab:green", alpha=0.7)
    ax.bar(layers, -neg, label="% hurt", color="tab:red", alpha=0.7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Layer"); ax.set_ylabel("Fraction of questions (mean across coef)")
    ax.set_title(f"[{mode}] Fraction improved vs hurt per layer (averaged across coef)")
    ax.legend()
    _save(fig, f"05_improved_hurt_{mode}")


def _plot_asymmetry(agg: pd.DataFrame, mode: str):
    """For each layer, compare mean Δ at +coef vs mean Δ at −coef of same |coef|."""
    rows = []
    for layer, grp in agg.groupby("layer"):
        for coef_abs in sorted({abs(c) for c in grp.coef.unique() if c != 0}):
            pos = grp[grp.coef == coef_abs]["mean_delta_correct"].mean()
            neg = grp[grp.coef == -coef_abs]["mean_delta_correct"].mean()
            if not (np.isnan(pos) or np.isnan(neg)):
                rows.append({"layer": layer, "coef_abs": coef_abs, "pos": pos, "neg": neg})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df.pos, -df.neg, c=df.layer, cmap="viridis", s=30, alpha=0.7)
    lo = min(df.pos.min(), (-df.neg).min())
    hi = max(df.pos.max(), (-df.neg).max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="symmetric")
    ax.set_xlabel("Mean Δ at +coef")
    ax.set_ylabel("−Mean Δ at −coef")
    ax.set_title(f"[{mode}] Asymmetry check: would expect points on y=x if symmetric")
    ax.legend()
    _save(fig, f"06_asymmetry_{mode}")


def _plot_coef0_drift(agg: pd.DataFrame, mode: str):
    """coef=0 should be exactly 0 Δ. Any drift = bug or numerical issue."""
    zero = agg[agg.coef == 0.0]
    if zero.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(zero.layer, zero.mean_delta_correct, marker="o", color="tab:purple")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axhline(DRIFT_THRESHOLD, color="red", lw=0.6, ls=":", label=f"|drift| > {DRIFT_THRESHOLD}")
    ax.axhline(-DRIFT_THRESHOLD, color="red", lw=0.6, ls=":")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Δ at coef=0")
    ax.set_title(f"[{mode}] coef=0 drift sanity (should be ≈ 0)")
    ax.legend()
    _save(fig, f"07_coef0_drift_{mode}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: RANK BEHAVIOR
# ═════════════════════════════════════════════════════════════════════════════

def plot_rank_section(mcf_agg: pd.DataFrame, cf_agg: pd.DataFrame):
    for agg, mode in [(mcf_agg, "MCF"), (cf_agg, "CF")]:
        if agg.empty:
            continue
        _plot_pct_rank1(agg, mode)


def _plot_pct_rank1(agg: pd.DataFrame, mode: str):
    pivot = agg.pivot_table(index="coef", columns="layer", values="pct_rank1")
    fig, ax = plt.subplots(figsize=(16, 6))
    if sns is not None:
        sns.heatmap(pivot, ax=ax, cmap="viridis", linewidths=0.3,
                    cbar_kws={"label": "Fraction of questions where correct is top-ranked"})
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax)
    ax.set_title(f"[{mode}] Layer × Coef — % of questions where correct answer ranks #1")
    ax.set_xlabel("Layer"); ax.set_ylabel("Coefficient")
    _save(fig, f"09_pct_rank1_{mode}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: PER-QUESTION STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

def plot_question_section(mcf: pd.DataFrame):
    if mcf.empty or "delta_correct_logprob" not in mcf.columns:
        return
    _plot_steerable_resistant(mcf)


def _plot_steerable_resistant(mcf: pd.DataFrame):
    """Best/worst questions by mean effect across layers."""
    # Pick a representative coef = the positive one with biggest median effect
    pos = mcf[mcf.coef > 0]
    if pos.empty:
        return
    best_c = pos.groupby("coef")["delta_correct_logprob"].median().idxmax()
    sub = mcf[mcf.coef == best_c]

    per_q = sub.groupby("question_id")["delta_correct_logprob"].mean().sort_values()
    bottom = per_q.head(TOP_K_QUESTIONS)
    top = per_q.tail(TOP_K_QUESTIONS)
    fig, ax = plt.subplots(figsize=(11, 5))
    qs = pd.concat([bottom, top])
    ax.barh(qs.index.astype(str), qs.values,
            color=["tab:red"] * len(bottom) + ["tab:green"] * len(top))
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel(f"Mean Δ correct logprob at coef={best_c}")
    ax.set_ylabel("Question id")
    ax.set_title(f"[MCF] Most-steered (green) and most-resistant (red) questions @ coef={best_c}")
    fig.tight_layout()
    _save(fig, "12_steerable_resistant_MCF")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: MARGIN  (phase0_margin.py)
# ═════════════════════════════════════════════════════════════════════════════

def plot_margin_section(mcf: pd.DataFrame, cf: pd.DataFrame):
    """Logit-gap / margin analysis. Both MCF and CF if available."""
    if not cf.empty and "delta_margin_sum" in cf.columns:
        _plot_margin_distribution(cf, mode="CF")
    if not mcf.empty and "correct_label_logprob" in mcf.columns:
        # MCF needs synthetic margin: correct logprob vs best-wrong; not directly stored.
        # Skip unless detailed wrong-label data is present.
        pass


def _plot_margin_distribution(cf: pd.DataFrame, mode: str):
    """Per-question Δ margin distribution at the best (positive coef, best layer) cell."""
    pos = cf[cf.coef > 0]
    if pos.empty:
        return
    # Pick (layer, coef) with biggest mean Δ margin
    g = pos.groupby(["layer", "coef"])["delta_margin_sum"].mean()
    best_layer, best_coef = g.idxmax()
    sub = cf[(cf.layer == best_layer) & (cf.coef == best_coef)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sub["delta_margin_sum"].dropna(), bins=30, alpha=0.7, color="tab:blue")
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Δ margin (correct − best-wrong)")
    ax.set_ylabel("# questions")
    ax.set_title(f"[{mode}] Per-question Δ-margin at best cell (L{best_layer}, c={best_coef})")
    _save(fig, f"30_margin_distribution_{mode}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: NEGATIVE-COEF DIAGNOSTICS  (diagnose_negative_coefs.py)
# ═════════════════════════════════════════════════════════════════════════════

def plot_negcoef_section(mcf_agg: pd.DataFrame, cf_agg: pd.DataFrame):
    """Compares positive- vs negative-coef effects, surfaces 'neg helps too' signals."""
    for agg, mode in [(mcf_agg, "MCF"), (cf_agg, "CF")]:
        if agg.empty:
            continue
        _plot_signed_effect(agg, mode)


def _plot_signed_effect(agg: pd.DataFrame, mode: str):
    """For each layer: bar of mean Δ at strongest +coef vs strongest −coef.

    If both bars point the same way (both green or both red), the steering
    isn't truly directional — it's amplitude-only or universal-direction-loaded.
    """
    # Strongest |coef| in each sign
    max_pos = agg[agg.coef > 0]["coef"].max() if (agg.coef > 0).any() else None
    max_neg = agg[agg.coef < 0]["coef"].min() if (agg.coef < 0).any() else None
    if max_pos is None or max_neg is None:
        return
    pos = agg[agg.coef == max_pos].set_index("layer")["mean_delta_correct"]
    neg = agg[agg.coef == max_neg].set_index("layer")["mean_delta_correct"]
    layers = sorted(set(pos.index) | set(neg.index))
    pos = pos.reindex(layers).fillna(0)
    neg = neg.reindex(layers).fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(layers))
    w = 0.4
    ax.bar(x - w / 2, pos.values, w, label=f"coef={max_pos:+.1f}", color="tab:green", alpha=0.7)
    ax.bar(x + w / 2, neg.values, w, label=f"coef={max_neg:+.1f}", color="tab:red",   alpha=0.7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Δ correct logprob")
    ax.set_title(f"[{mode}] Mean effect at extreme +coef vs extreme −coef per layer\n"
                 f"(if both bars same sign at a layer, steering isn't directional there)")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"40_negcoef_signed_effect_{mode}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: CROSS-METHOD  (MCF vs CF agreement)
# ═════════════════════════════════════════════════════════════════════════════

def plot_cross_method_section(mcf_agg: pd.DataFrame, cf_agg: pd.DataFrame):
    if mcf_agg.empty or cf_agg.empty:
        return
    _plot_best_layer_comparison(mcf_agg, cf_agg)


def _plot_best_layer_comparison(mcf_agg: pd.DataFrame, cf_agg: pd.DataFrame):
    """Do MCF and CF agree on which layers are the most-steered?"""
    mcf_eff = mcf_agg.groupby("layer")["mean_delta_correct"].apply(lambda x: x.abs().max())
    cf_eff = cf_agg.groupby("layer")["mean_delta_correct"].apply(lambda x: x.abs().max())
    layers = sorted(set(mcf_eff.index) | set(cf_eff.index))
    mcf_v = mcf_eff.reindex(layers).fillna(0)
    cf_v = cf_eff.reindex(layers).fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax2 = ax.twinx()
    ax.plot(layers, mcf_v, marker="o", color="tab:blue", label="MCF max |effect|")
    ax2.plot(layers, cf_v, marker="s", color="tab:orange", label="CF max |effect|")
    ax.set_xlabel("Layer")
    ax.set_ylabel("MCF max |Δ correct logprob|", color="tab:blue")
    ax2.set_ylabel("CF max |Δ correct logprob|", color="tab:orange")
    ax.set_title("MCF vs CF — do the two methods agree on best layers?")
    fig.tight_layout()
    _save(fig, "15_best_layer_MCF_vs_CF")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: CROSS-EXPERIMENT COMPARISON  (compare.py)
# ═════════════════════════════════════════════════════════════════════════════

def plot_compare_section(experiments: List[Tuple[str, pd.DataFrame, pd.DataFrame]]):
    """experiments = [(exp_id, mcf_agg, cf_agg), ...]"""
    if len(experiments) < 2:
        return

    # Layer sweep: each experiment's mean Δ correct logprob at a focus coef,
    # overlaid for direct comparison.
    target_coef = 1.0
    for mode in ("MCF", "CF"):
        fig, ax = plt.subplots(figsize=(13, 5))
        for exp_id, mcf_agg, cf_agg in experiments:
            agg = mcf_agg if mode == "MCF" else cf_agg
            if agg.empty:
                continue
            sub = agg[agg.coef.between(target_coef - 0.05, target_coef + 0.05)]
            if sub.empty:
                # Closest coef
                if not (agg.coef > 0).any():
                    continue
                closest = agg.iloc[(agg.coef - target_coef).abs().argsort()[:1]]["coef"].iloc[0]
                sub = agg[agg.coef == closest]
            sub = sub.sort_values("layer")
            ax.plot(sub.layer, sub.mean_delta_correct, marker="o", markersize=4, label=exp_id)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_xlabel("Layer"); ax.set_ylabel("Mean Δ correct logprob (coef≈+1)")
        ax.set_title(f"[{mode}] Cross-experiment layer sweep @ coef≈+1")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        _save(fig, f"50_compare_layer_sweep_{mode}")

    # Selectivity comparison (CF only)
    fig, ax = plt.subplots(figsize=(13, 5))
    plotted = False
    for exp_id, _, cf_agg in experiments:
        if cf_agg.empty or "selectivity_index" not in cf_agg.columns:
            continue
        # Aggregate selectivity per layer across positive coefs
        sub = cf_agg[cf_agg.coef > 0]
        if sub.empty:
            continue
        per_layer = sub.groupby("layer")["selectivity_index"].mean()
        ax.plot(per_layer.index, per_layer.values, marker="o", markersize=4, label=exp_id)
        plotted = True
    if plotted:
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_xlabel("Layer"); ax.set_ylabel("Mean selectivity (positive coefs)")
        ax.set_title("Cross-experiment selectivity — which vector is more target-aware?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, "51_compare_selectivity")
    else:
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

ALL_SECTIONS = [
    "continuous", "effect", "asymmetry", "rank", "question",
    "cross-method", "margin", "negcoef", "compare",
]


def run_one(exp_root: Path, sections: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n── {exp_root.name} ──")
    mcf = load_mcf(exp_root)
    cf  = load_cf_wide(exp_root)
    mcf = attach_split(mcf, exp_root)
    cf  = attach_split(cf, exp_root)
    print(f"  MCF rows: {len(mcf)}  CF rows: {len(cf)}")

    mcf_agg = agg_mcf(mcf) if not mcf.empty else pd.DataFrame()
    cf_agg  = agg_cf_continuous(cf) if not cf.empty else pd.DataFrame()

    if "continuous" in sections and not cf_agg.empty:
        print("  [continuous]"); plot_continuous_section(cf_agg)
    if "effect" in sections:
        print("  [effect]"); plot_effect_section(mcf_agg, cf_agg)
    if "asymmetry" in sections:
        print("  [asymmetry]"); plot_asymmetry_section(mcf_agg, cf_agg)
    if "rank" in sections:
        print("  [rank]"); plot_rank_section(mcf_agg, cf_agg)
    if "question" in sections and not mcf.empty:
        print("  [question]"); plot_question_section(mcf)
    if "cross-method" in sections:
        print("  [cross-method]"); plot_cross_method_section(mcf_agg, cf_agg)
    if "margin" in sections:
        print("  [margin]"); plot_margin_section(mcf, cf)
    if "negcoef" in sections:
        print("  [negcoef]"); plot_negcoef_section(mcf_agg, cf_agg)

    # Save aggregates
    if _OUT_DIR is not None:
        if not mcf_agg.empty:
            mcf_agg.to_csv(_OUT_DIR / "mcf_agg.csv", index=False)
            print(f"  saved mcf_agg.csv ({len(mcf_agg)} rows)")
        if not cf_agg.empty:
            cf_agg.to_csv(_OUT_DIR / "cf_agg.csv", index=False)
            print(f"  saved cf_agg.csv ({len(cf_agg)} rows)")

    return mcf_agg, cf_agg


def main():
    global _OUT_DIR, _SHOW

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_roots", nargs="+", type=Path,
                   help="One or more experiment result directories")
    p.add_argument("--section", default="all",
                   help=f"Comma-separated; one of: {','.join(ALL_SECTIONS)}, or 'all'")
    p.add_argument("--save", action="store_true",
                   help="Save plots to {exp_root}/analysis/ (or --out-dir)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override save directory")
    p.add_argument("--no-show", action="store_true", help="Don't display plots interactively")
    args = p.parse_args()

    sections = ALL_SECTIONS if args.section == "all" else args.section.split(",")
    for s in sections:
        if s not in ALL_SECTIONS:
            sys.exit(f"Unknown section: {s!r}. Pick from {ALL_SECTIONS} or 'all'.")

    _SHOW = not args.no_show and not args.save  # if both --save and --no-show absent, show

    # Multi-experiment vs single-experiment branching
    if len(args.exp_roots) == 1:
        exp = args.exp_roots[0]
        if not exp.exists():
            sys.exit(f"Not found: {exp}")
        if args.save:
            _OUT_DIR = args.out_dir or (exp / "analysis")
            print(f"Saving to: {_OUT_DIR}")
        run_one(exp, sections)
    else:
        # Multi-experiment: per-exp plots into each exp's analysis dir if --save,
        # plus a top-level compare directory (or --out-dir if given).
        per_exp_aggs = []
        for exp in args.exp_roots:
            if not exp.exists():
                print(f"  WARNING: {exp} not found, skipping"); continue
            if args.save:
                _OUT_DIR = args.out_dir / exp.name if args.out_dir else (exp / "analysis")
                print(f"  → per-exp save: {_OUT_DIR}")
            mcf_agg, cf_agg = run_one(exp, sections)
            per_exp_aggs.append((exp.name, mcf_agg, cf_agg))

        if "compare" in sections and len(per_exp_aggs) >= 2:
            if args.save:
                _OUT_DIR = (args.out_dir or args.exp_roots[0].parent) / "_compare"
                print(f"\nCross-experiment plots → {_OUT_DIR}")
            print("\n── compare ──")
            plot_compare_section(per_exp_aggs)


if __name__ == "__main__":
    main()
