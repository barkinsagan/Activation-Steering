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
_REPORT: bool = False
_REPORT_LINES: List[str] = []


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
# TEXT REPORT — terminal-friendly output for HPC / no-display environments
# ═════════════════════════════════════════════════════════════════════════════

_W = 76  # report line width


def _rp(line: str = ""):
    print(line)
    _REPORT_LINES.append(line)


def _hr(char: str = "─"):
    _rp(char * _W)


def _section_hdr(title: str, num: str = ""):
    _rp()
    _hr()
    prefix = f"  {num}. " if num else "  "
    _rp(f"{prefix}{title}")
    _hr()


def _fv(v, fmt: str = "+.3f", na: str = "    n/a") -> str:
    """Format float, return na string if NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return format(v, fmt)


def _sig(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "    "
    if p < 0.001: return " ***"
    if p < 0.01:  return "  **"
    if p < 0.05:  return "   *"
    return " n.s"


def _best_per_layer(agg: pd.DataFrame, val_col: str = "mean_delta_correct",
                    pos_only: bool = False) -> pd.DataFrame:
    sub = agg[agg.coef > 0] if pos_only else agg
    if sub.empty:
        return pd.DataFrame()
    idx = sub.groupby("layer")[val_col].idxmax()
    return sub.loc[idx].sort_values("layer").reset_index(drop=True)


# ── Report sections ───────────────────────────────────────────────────────────

def _rpt_overview(exp_root: Path, mcf: pd.DataFrame, cf: pd.DataFrame,
                  mcf_agg: pd.DataFrame, cf_agg: pd.DataFrame):
    _rp("=" * _W)
    _rp("  STEERING VECTOR ANALYSIS REPORT")
    _rp(f"  Experiment : {exp_root.name}")
    _rp("=" * _W)
    _section_hdr("DATA OVERVIEW", "0")
    _rp(f"  MCF rows   : {len(mcf):,}   (per-question rows, all layers × coefs)")
    _rp(f"  CF  rows   : {len(cf):,}")
    ref = mcf_agg if not mcf_agg.empty else cf_agg
    if not ref.empty:
        layers = sorted(ref.layer.unique())
        coefs  = sorted(ref.coef.unique())
        mid_c  = coefs[len(coefs) // 2]
        n_q_s  = ref[ref.coef == mid_c]["n"]
        n_q    = int(n_q_s.mean()) if not n_q_s.empty else "?"
        _rp(f"  Layers     : L{layers[0]:02d}–L{layers[-1]:02d}  ({len(layers)} total)")
        _rp(f"  Coefs      : {coefs[0]:+.2f} → {coefs[-1]:+.2f}  ({len(coefs)} values)")
        half = len(coefs) // 2
        _rp("  Coef list  : " + "  ".join(f"{c:+.2f}" for c in coefs[:half]))
        _rp("               " + "  ".join(f"{c:+.2f}" for c in coefs[half:]))
        _rp(f"  Questions  : ~{n_q} per (layer, coef) cell")
    _rp()


def _rpt_effect(cf_agg: pd.DataFrame, mcf_agg: pd.DataFrame):
    _section_hdr("EFFECT MAGNITUDE — Does steering move logprobs at all?", "1")

    for agg, mode in [(cf_agg, "CF"), (mcf_agg, "MCF")]:
        if agg.empty:
            continue
        _rp(f"\n  [{mode}]")

        # per-layer best (positive coefs only)
        best = _best_per_layer(agg, "mean_delta_correct", pos_only=True)
        if best.empty:
            continue
        has_kl = mode == "CF" and "mean_kl_baseline" in agg.columns
        _rp(f"\n  Per-layer best Δ-correct (best +coef per layer):")
        hdr = f"  {'Layer':>5}  {'Best_C':>7}  {'Δ-correct':>10}  {'±std':>6}  {'%Impr':>6}  {'%Hurt':>6}"
        if has_kl:
            hdr += f"  {'KL':>6}"
        _rp(hdr)
        _hr("·")
        for _, row in best.iterrows():
            line = (f"  L{int(row.layer):02d}    "
                    f"{row.coef:+7.2f}  "
                    f"{_fv(row.mean_delta_correct):>10}  "
                    f"{row.get('std_delta_correct', float('nan')):>6.3f}  "
                    f"{row.get('pct_improved', float('nan')):>5.0%}  "
                    f"{row.get('pct_hurt', float('nan')):>5.0%}")
            if has_kl:
                line += f"  {_fv(row.get('mean_kl_baseline', float('nan')), '.4f', '   n/a'):>6}"
            _rp(line)

        # top-5 cells overall
        top5 = agg.nlargest(5, "mean_delta_correct")
        _rp(f"\n  Top 5 cells by Δ-correct:")
        _rp(f"  {'Rank':>4}  {'Layer':>5}  {'Coef':>6}  {'Δ-correct':>10}  {'%Improved':>10}  {'%Hurt':>6}")
        _hr("·")
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            _rp(f"  {rank:>4}  L{int(row.layer):02d}    "
                f"{row.coef:+6.2f}  "
                f"{_fv(row.mean_delta_correct):>10}  "
                f"{row.get('pct_improved', float('nan')):>9.0%}  "
                f"{row.get('pct_hurt', float('nan')):>5.0%}")

        best_cell  = agg.loc[agg.mean_delta_correct.idxmax()]
        worst_cell = agg.loc[agg.mean_delta_correct.idxmin()]
        _rp()
        _rp(f"  BEST  cell: L{int(best_cell.layer):02d}, coef={best_cell.coef:+.2f}, "
            f"Δ={_fv(best_cell.mean_delta_correct)}")
        _rp(f"  WORST cell: L{int(worst_cell.layer):02d}, coef={worst_cell.coef:+.2f}, "
            f"Δ={_fv(worst_cell.mean_delta_correct)}")
        _rp()


def _rpt_coef_sweeps(cf_agg: pd.DataFrame):
    """Full coef × metric table at the top-4 layers plus first and last."""
    _section_hdr("COEF SWEEPS AT KEY LAYERS — full coef profile", "2")

    if cf_agg.empty:
        _rp("  [no CF data]")
        return

    layer_eff = cf_agg.groupby("layer")["mean_delta_correct"].apply(lambda x: x.abs().max())
    top4      = layer_eff.nlargest(4).index.tolist()
    all_layers = sorted(cf_agg.layer.unique())
    key_layers = sorted(set(top4 + [all_layers[0], all_layers[-1]]))

    has_kl  = "mean_kl_baseline" in cf_agg.columns and not cf_agg.mean_kl_baseline.isna().all()
    has_sel = "selectivity_index" in cf_agg.columns
    has_wp  = "wilcoxon_p" in cf_agg.columns
    has_dw  = "mean_delta_best_wrong" in cf_agg.columns

    for layer in key_layers:
        sub = cf_agg[cf_agg.layer == layer].sort_values("coef")
        _rp(f"\n  ── Layer L{layer:02d} ──")
        hdr = f"  {'Coef':>6}  {'Δ-correct':>10}  {'±std':>6}"
        if has_dw:  hdr += f"  {'Δ-wrong':>8}"
        if has_sel: hdr += f"  {'Selectiv':>8}"
        if has_kl:  hdr += f"  {'KL':>6}"
        if has_wp:  hdr += f"  {'p-val':>7}  sig"
        _rp(hdr)
        _hr("·")
        for _, row in sub.iterrows():
            is_base = abs(row.coef) < 1e-6
            line = (f"  {row.coef:+6.2f}  "
                    f"{_fv(row.mean_delta_correct):>10}  "
                    f"{row.get('std_delta_correct', float('nan')):>6.3f}")
            if has_dw:  line += f"  {_fv(row.get('mean_delta_best_wrong', float('nan'))):>8}"
            if has_sel: line += f"  {_fv(row.get('selectivity_index', float('nan')), '+.3f', '    n/a'):>8}"
            if has_kl:  line += f"  {_fv(row.get('mean_kl_baseline', float('nan')), '.4f', '   n/a'):>6}"
            if has_wp:
                p    = row.get("wilcoxon_p", float("nan"))
                line += f"  {_fv(p, '.4f', '   n/a'):>7}  {_sig(p)}"
            if is_base:
                line += "  ← baseline (should be ≈0)"
            _rp(line)


def _rpt_selectivity(cf_agg: pd.DataFrame):
    _section_hdr("TARGET-AWARE SELECTIVITY — Does steering help correct more than wrong?", "3")
    _rp("  Selectivity = (Δcorrect − Δbest-wrong) / (|Δcorrect| + |Δbest-wrong|)")
    _rp("    +1.0 = perfectly target-aware   0.0 = uniform shift   −1.0 = anti-target")
    _rp()

    if cf_agg.empty or "selectivity_index" not in cf_agg.columns:
        _rp("  [no selectivity data]")
        return

    # Per-layer summary (positive coefs only)
    pos = cf_agg[cf_agg.coef > 0]
    if not pos.empty:
        per_layer_sel = pos.groupby("layer").agg(
            mean_sel=("selectivity_index", "mean"),
            max_sel=("selectivity_index", "max"),
        ).reset_index()
        best_coef_map = (pos.loc[pos.groupby("layer")["selectivity_index"].idxmax()]
                         .set_index("layer")["coef"])
        per_layer_sel["best_coef"] = per_layer_sel["layer"].map(best_coef_map)

        _rp("  Per-layer selectivity (averaged over positive coefs):")
        _rp(f"  {'Layer':>5}  {'Mean Sel':>9}  {'Max Sel':>8}  {'BestCoef':>9}  Interpretation")
        _hr("·")
        for _, row in per_layer_sel.sort_values("layer").iterrows():
            sel = row.mean_sel
            if sel > 0.15:   interp = "target-aware ✓"
            elif sel > 0.05: interp = "weakly positive"
            elif sel > -0.05: interp = "near-zero (uniform)"
            else:             interp = "anti-target ✗"
            _rp(f"  L{int(row.layer):02d}    "
                f"{_fv(sel, '+.3f'):>9}  "
                f"{_fv(row.max_sel, '+.3f'):>8}  "
                f"{row.best_coef:>+9.2f}  "
                f"{interp}")

    # Top-10 cells by selectivity
    _rp()
    has_wp = "wilcoxon_p" in cf_agg.columns
    top10 = cf_agg[cf_agg.coef != 0.0].nlargest(10, "selectivity_index")
    _rp("  Top 10 cells by selectivity index:")
    hdr = f"  {'Rank':>4}  {'Layer':>5}  {'Coef':>6}  {'Selectiv':>8}  {'Δ-correct':>10}  {'Δ-wrong':>8}"
    if has_wp: hdr += f"  {'p-val':>7}  sig"
    _rp(hdr)
    _hr("·")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        line = (f"  {rank:>4}  L{int(row.layer):02d}    "
                f"{row.coef:+6.2f}  "
                f"{_fv(row.selectivity_index, '+.3f'):>8}  "
                f"{_fv(row.mean_delta_correct):>10}  "
                f"{_fv(row.get('mean_delta_best_wrong', float('nan'))):>8}")
        if has_wp:
            p = row.get("wilcoxon_p", float("nan"))
            line += f"  {_fv(p, '.4f', '   n/a'):>7}  {_sig(p)}"
        _rp(line)

    # Wilcoxon significant cells
    _rp()
    if has_wp:
        sig = cf_agg[(cf_agg.coef != 0.0) & (cf_agg.wilcoxon_p < 0.05)].sort_values("wilcoxon_p")
        _rp(f"  STATISTICALLY SIGNIFICANT CELLS (Wilcoxon p < 0.05)  — n={len(sig)}:")
        if sig.empty:
            _rp("    None — no cell shows target-aware effect at p<0.05")
        else:
            _rp(f"  {'Layer':>5}  {'Coef':>6}  {'p-value':>8}  sig  {'Selectiv':>8}  "
                f"{'Δ-correct':>10}  {'Δ-wrong':>8}")
            _hr("·")
            for _, row in sig.iterrows():
                p = row.wilcoxon_p
                _rp(f"  L{int(row.layer):02d}    "
                    f"{row.coef:+6.2f}  "
                    f"{p:>8.4f}  {_sig(p)}  "
                    f"{_fv(row.selectivity_index, '+.3f'):>8}  "
                    f"{_fv(row.mean_delta_correct):>10}  "
                    f"{_fv(row.get('mean_delta_best_wrong', float('nan'))):>8}")
        _rp()


def _rpt_directionality(cf_agg: pd.DataFrame, mcf_agg: pd.DataFrame):
    _section_hdr("DIRECTIONALITY — Does the sign of the coefficient matter?", "4")
    _rp("  +coef and −coef produce OPPOSITE Δ → vector is directional (encodes domain)")
    _rp("  +coef and −coef produce SAME-SIGN Δ → vector captures universal direction")
    _rp()

    for agg, mode in [(cf_agg, "CF"), (mcf_agg, "MCF")]:
        if agg.empty:
            continue
        max_pos_c = agg[agg.coef > 0]["coef"].max() if (agg.coef > 0).any() else None
        min_neg_c = agg[agg.coef < 0]["coef"].min() if (agg.coef < 0).any() else None
        if max_pos_c is None or min_neg_c is None:
            continue

        _rp(f"  [{mode}]  coef={max_pos_c:+.2f} vs coef={min_neg_c:+.2f}")
        _rp(f"  {'Layer':>5}  {'Δ@+coef':>9}  {'Δ@-coef':>9}  {'Opp.signs':>9}  Note")
        _hr("·")

        pos_vals = agg[agg.coef == max_pos_c].set_index("layer")["mean_delta_correct"]
        neg_vals = agg[agg.coef == min_neg_c].set_index("layer")["mean_delta_correct"]
        layers   = sorted(set(pos_vals.index) | set(neg_vals.index))

        n_dir = 0
        for layer in layers:
            p = pos_vals.get(layer, float("nan"))
            n = neg_vals.get(layer, float("nan"))
            if not (np.isnan(p) or np.isnan(n)) and p * n < 0:
                opp, note, n_dir = "YES ✓", "", n_dir + 1
            elif np.isnan(p) or np.isnan(n):
                opp, note = "n/a", ""
            elif p > 0 and n > 0:
                opp, note = "no", "both positive → universal+ direction"
            else:
                opp, note = "no", "both negative → universal− direction"
            _rp(f"  L{layer:02d}    "
                f"{_fv(p):>9}  {_fv(n):>9}  {opp:>9}  {note}")

        _rp(f"\n  Directional at {n_dir}/{len(layers)} layers "
            f"({100 * n_dir / max(len(layers), 1):.0f}%)")
        _rp()


def _rpt_kl(cf_agg: pd.DataFrame):
    _section_hdr("INTERVENTION MAGNITUDE — KL(steered ‖ baseline)", "5")
    _rp("  Measures how much the answer distribution shifts, regardless of direction.")
    _rp("  KL ≈ 0 → negligible intervention  |  KL > 0.1 → substantial shift")
    _rp()

    if cf_agg.empty or "mean_kl_baseline" not in cf_agg.columns \
            or cf_agg.mean_kl_baseline.isna().all():
        _rp("  [no KL data]")
        return

    max_pos_c = cf_agg[cf_agg.coef > 0]["coef"].max() if (cf_agg.coef > 0).any() else None
    if max_pos_c is None:
        return

    sub = cf_agg[cf_agg.coef == max_pos_c].sort_values("layer")
    _rp(f"  KL at coef={max_pos_c:+.2f} (strongest +coef) per layer:")
    _rp(f"  {'Layer':>5}  {'Mean KL':>8}  Magnitude")
    _hr("·")
    for _, row in sub.iterrows():
        kl = row.mean_kl_baseline
        if np.isnan(kl):           mag = "n/a"
        elif kl < 0.005:           mag = "negligible"
        elif kl < 0.02:            mag = "small"
        elif kl < 0.08:            mag = "moderate"
        elif kl < 0.25:            mag = "large"
        else:                       mag = "VERY LARGE"
        _rp(f"  L{int(row.layer):02d}    {_fv(kl, '.4f', '    n/a'):>8}  {mag}")

    _rp()
    pos_kl = cf_agg[cf_agg.coef > 0].groupby("layer")["mean_kl_baseline"].max()
    vals   = "  ".join(f"L{int(l):02d}:{v:.3f}" for l, v in pos_kl.items())
    # wrap at width
    for i in range(0, len(vals), _W - 4):
        _rp("  " + vals[i:i + _W - 4])
    _rp()


def _rpt_accuracy(mcf_agg: pd.DataFrame):
    _section_hdr("ACCURACY (MCF) — Baseline and best-coef accuracy per layer", "6")

    if mcf_agg.empty or "accuracy" not in mcf_agg.columns:
        _rp("  [no MCF accuracy data]")
        return

    coefs   = sorted(mcf_agg.coef.unique())
    base_c  = min(coefs, key=lambda c: abs(c))   # closest to 0
    base    = mcf_agg[mcf_agg.coef == base_c][["layer", "accuracy"]].sort_values("layer")
    best_p  = _best_per_layer(mcf_agg, "accuracy", pos_only=True)

    _rp("  Layer  Baseline Acc  Best(+)Acc  Best(+)C  Acc Δ")
    _hr("·")
    for _, brow in base.iterrows():
        layer  = brow.layer
        brow2  = best_p[best_p.layer == layer]
        if brow2.empty:
            continue
        brow2  = brow2.iloc[0]
        delta  = brow2.accuracy - brow.accuracy
        _rp(f"  L{int(layer):02d}    "
            f"{brow.accuracy:>12.1%}  "
            f"{brow2.accuracy:>10.1%}  "
            f"{brow2.coef:>+8.2f}  "
            f"{delta:>+6.1%}")

    n_q = int(mcf_agg[mcf_agg.coef == base_c]["n"].mean())
    se  = 1 / (2 * n_q ** 0.5)
    _rp()
    _rp(f"  Mean baseline accuracy : {base['accuracy'].mean():.1%}")
    _rp(f"  Mean best-coef accuracy: {best_p['accuracy'].mean():.1%}  "
        f"(Δ = {best_p['accuracy'].mean() - base['accuracy'].mean():+.1%})")
    _rp(f"  ⚠  n≈{n_q} per cell → SE ≈ {se:.1%}; differences < {2*se:.0%} are within 2σ noise")
    _rp()


def _rpt_sanity(cf_agg: pd.DataFrame, mcf_agg: pd.DataFrame):
    _section_hdr("SANITY CHECKS", "7")

    for agg, mode in [(cf_agg, "CF"), (mcf_agg, "MCF")]:
        if agg.empty:
            continue
        zero = agg[agg.coef.abs() < 1e-6]
        if zero.empty:
            continue
        max_drift = zero["mean_delta_correct"].abs().max()
        layer_md  = int(zero.loc[zero["mean_delta_correct"].abs().idxmax(), "layer"])
        status    = "PASS ✓" if max_drift < 0.05 else "WARNING ✗ (possible bug)"
        _rp(f"  [{mode}] coef=0 drift: max |Δ| = {max_drift:.4f} at L{layer_md:02d}  →  {status}")
        if max_drift >= 0.05:
            _rp("    Drift > 0.05 at coef=0 suggests a bug or numerical issue. "
                "Investigate before trusting results.")
    _rp()


def _rpt_verdict(cf_agg: pd.DataFrame, mcf_agg: pd.DataFrame):
    _section_hdr("SUMMARY VERDICT", "8")

    agg = cf_agg if not cf_agg.empty else mcf_agg
    if agg.empty:
        _rp("  No data — cannot produce verdict.")
        return

    findings = []

    # 1. Effect size
    max_eff  = agg["mean_delta_correct"].abs().max()
    best_c   = agg.loc[agg["mean_delta_correct"].abs().idxmax()]
    eff_desc = ("negligible (<0.05)" if max_eff < 0.05 else
                "small (0.05–0.2)"  if max_eff < 0.2  else
                "moderate (0.2–0.5)" if max_eff < 0.5 else "large (>0.5)")
    findings.append(
        f"EFFECT SIZE: {eff_desc} — max |Δ-correct| = {max_eff:.3f} "
        f"at L{int(best_c.layer):02d}, coef={best_c.coef:+.2f}."
    )

    # 2. Selectivity
    if not cf_agg.empty and "selectivity_index" in cf_agg.columns:
        pos_sel  = cf_agg[cf_agg.coef > 0]["selectivity_index"]
        mean_sel = pos_sel.mean()
        max_sel  = pos_sel.max()
        best_sel = cf_agg.loc[cf_agg.selectivity_index.idxmax()]
        if mean_sel > 0.1:
            desc = f"positive (mean={mean_sel:+.3f}) — steering is target-aware"
        elif mean_sel > 0:
            desc = f"weakly positive (mean={mean_sel:+.3f}) — marginal target-awareness"
        elif mean_sel > -0.05:
            desc = f"near-zero (mean={mean_sel:+.3f}) — uniform probability redistribution"
        else:
            desc = f"negative (mean={mean_sel:+.3f}) — anti-target effect"
        findings.append(
            f"SELECTIVITY: {desc}. "
            f"Max={max_sel:+.3f} at L{int(best_sel.layer):02d}, coef={best_sel.coef:+.2f}."
        )

    # 3. Statistical significance
    if not cf_agg.empty and "wilcoxon_p" in cf_agg.columns:
        sig05 = cf_agg[(cf_agg.coef != 0.0) & (cf_agg.wilcoxon_p < 0.05)]
        sig01 = cf_agg[(cf_agg.coef != 0.0) & (cf_agg.wilcoxon_p < 0.01)]
        if len(sig05) == 0:
            findings.append(
                "SIGNIFICANCE: No cells significant at p<0.05 — "
                "target-aware effect is not statistically confirmed."
            )
        else:
            best_sig = sig05.loc[sig05.wilcoxon_p.idxmin()]
            findings.append(
                f"SIGNIFICANCE: {len(sig05)} cells at p<0.05, {len(sig01)} at p<0.01. "
                f"Most significant: L{int(best_sig.layer):02d}, coef={best_sig.coef:+.2f}, "
                f"p={best_sig.wilcoxon_p:.4f}."
            )

    # 4. Best layer band
    top5_layers = (agg.groupby("layer")["mean_delta_correct"]
                   .apply(lambda x: x.abs().max())
                   .nlargest(5).index.tolist())
    findings.append(f"BEST LAYERS: {sorted(top5_layers)} — top-5 by max |Δ-correct|.")

    # 5. Directionality
    if (agg.coef > 0).any() and (agg.coef < 0).any():
        max_pc = agg[agg.coef > 0]["coef"].max()
        min_nc = agg[agg.coef < 0]["coef"].min()
        pv = agg[agg.coef == max_pc]["mean_delta_correct"].values
        nv = agg[agg.coef == min_nc]["mean_delta_correct"].values
        n_dir = int((pv * nv < 0).sum())
        n_tot = len(pv)
        pct   = 100 * n_dir / max(n_tot, 1)
        desc  = ("strong" if pct > 70 else "mixed" if pct > 40 else "weak")
        findings.append(
            f"DIRECTIONALITY: {desc} — {n_dir}/{n_tot} layers ({pct:.0f}%) show "
            "opposite sign at +coef vs −coef."
        )

    _rp()
    for i, f in enumerate(findings, 1):
        _rp(f"  {i}. {f}")
        _rp()
    _hr("═")


def _rpt_split_comparison(cf: pd.DataFrame, mcf: pd.DataFrame):
    """Section 9 — Validation vs test results shown separately."""
    _section_hdr("VALIDATION vs TEST SET RESULTS", "9")

    has_cf_split  = not cf.empty  and "split" in cf.columns  and cf["split"].notna().any()
    has_mcf_split = not mcf.empty and "split" in mcf.columns and mcf["split"].notna().any()

    if not has_cf_split and not has_mcf_split:
        _rp("  split_manifest.csv not found — all questions were pooled.")
        _rp("  Re-run experiment with a split config to enable this analysis.")
        return

    for raw_df, mode in [(cf, "CF"), (mcf, "MCF")]:
        if raw_df.empty or "split" not in raw_df.columns:
            continue
        _rp(f"\n  [{mode}]")

        # Aggregate each split independently
        results: Dict[str, Tuple[pd.DataFrame, int]] = {}
        for split in ["validation", "test"]:
            sub = raw_df[raw_df["split"] == split]
            if sub.empty:
                continue
            n_q = int(sub["question_id"].nunique()) if "question_id" in sub.columns else len(sub)
            agg = agg_cf_continuous(sub) if mode == "CF" else agg_mcf(sub)
            results[split] = (agg, n_q)

        if not results:
            _rp("  No split labels found."); continue

        val_agg,  n_val  = results.get("validation", (pd.DataFrame(), 0))
        test_agg, n_test = results.get("test",       (pd.DataFrame(), 0))
        _rp(f"  n_validation={n_val}  n_test={n_test}")

        if val_agg.empty:
            _rp("  Validation set empty — cannot select best cells."); continue

        # Top-5 by Δ-correct on validation
        top5 = val_agg.nlargest(5, "mean_delta_correct")
        has_sel = "selectivity_index" in val_agg.columns
        has_wp  = "wilcoxon_p" in val_agg.columns

        _rp(f"\n  VALIDATION — Top 5 cells by Δ-correct:")
        hdr = f"  {'Rank':>4}  {'Layer':>5}  {'Coef':>6}  {'Δ-correct':>10}  {'%Impr':>6}"
        if has_sel: hdr += f"  {'Selectiv':>9}"
        if has_wp:  hdr += f"  {'p-val':>7}  sig"
        _rp(hdr); _hr("·")

        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            line = (f"  {rank:>4}  L{int(row.layer):02d}    "
                    f"{row.coef:+6.2f}  "
                    f"{_fv(row.mean_delta_correct):>10}  "
                    f"{row.get('pct_improved', float('nan')):>5.0%}")
            if has_sel: line += f"  {_fv(row.get('selectivity_index', float('nan')), '+.3f'):>9}"
            if has_wp:
                p = row.get("wilcoxon_p", float("nan"))
                line += f"  {_fv(p, '.4f', '   n/a'):>7}  {_sig(p)}"
            _rp(line)

        if test_agg.empty:
            _rp("\n  Test set empty — no generalisation check possible."); continue

        # For those same validation-best cells, look up test numbers
        test_idx = test_agg.set_index(["layer", "coef"])
        _rp(f"\n  TEST SET — same val-selected cells (do they generalise?):")
        hdr2 = (f"  {'Layer':>5}  {'Coef':>6}  {'Δ-corr(val)':>12}  "
                f"{'Δ-corr(tst)':>12}  {'Δ-diff':>7}")
        if has_sel: hdr2 += f"  {'Sel(val)':>9}  {'Sel(tst)':>9}"
        _rp(hdr2); _hr("·")

        for _, row in top5.iterrows():
            key = (row.layer, row.coef)
            try:
                tr = test_idx.loc[key]
                diff = tr["mean_delta_correct"] - row["mean_delta_correct"]
                line = (f"  L{int(row.layer):02d}    {row.coef:+6.2f}  "
                        f"{_fv(row.mean_delta_correct):>12}  "
                        f"{_fv(tr['mean_delta_correct']):>12}  "
                        f"{diff:>+7.3f}")
                if has_sel:
                    line += (f"  {_fv(row.get('selectivity_index', float('nan')), '+.3f'):>9}  "
                             f"{_fv(tr.get('selectivity_index', float('nan')), '+.3f'):>9}")
            except KeyError:
                line = (f"  L{int(row.layer):02d}    {row.coef:+6.2f}  "
                        f"{_fv(row.mean_delta_correct):>12}  {'n/a':>12}  {'n/a':>7}")
            _rp(line)

        # Layer-sweep correlation: val vs test at the val-best coef
        val_best_coef = top5.iloc[0]["coef"]
        v_sweep = val_agg[val_agg.coef == val_best_coef].set_index("layer")["mean_delta_correct"]
        t_sweep = test_agg[test_agg.coef == val_best_coef].set_index("layer")["mean_delta_correct"]
        common  = sorted(set(v_sweep.index) & set(t_sweep.index))
        if len(common) >= 4:
            try:
                from scipy.stats import pearsonr
                r, p = pearsonr(v_sweep.reindex(common).values, t_sweep.reindex(common).values)
                agree = "agree ✓" if r > 0.7 else ("moderate" if r > 0.4 else "disagree ✗")
                _rp(f"\n  Layer-sweep correlation at coef={val_best_coef:+.2f}: "
                    f"r={r:.3f}, p={p:.3f}  →  {agree}")
            except ImportError:
                pass

        # Best-layer agreement
        vbl = int(val_agg.loc[val_agg.mean_delta_correct.idxmax(), "layer"])
        tbl = int(test_agg.loc[test_agg.mean_delta_correct.idxmax(), "layer"])
        match = "MATCH ✓" if vbl == tbl else "MISMATCH ✗ (overfitting risk)"
        _rp(f"  Best layer: val=L{vbl:02d}  test=L{tbl:02d}  →  {match}")
        _rp()


def _rpt_crossval(cf: pd.DataFrame, mcf: pd.DataFrame, k: int = 5, n_boot: int = 500):
    """Section 10 — K-fold CV + bootstrap CI on per-question logprob deltas."""
    _section_hdr(f"CROSS-VALIDATION ({k}-fold + bootstrap, question-level)", "10")
    _rp("  Randomly splits questions into k folds; each fold is held out in turn.")
    _rp("  Tells you whether the effect is consistent across different question subsets,")
    _rp("  not just a fluke of one particular sample.")
    _rp()

    for raw_df, mode in [(cf, "CF"), (mcf, "MCF")]:
        if raw_df.empty or "question_id" not in raw_df.columns:
            continue

        delta_col = ("delta_target_sum_lp"    if mode == "CF" and "delta_target_sum_lp"    in raw_df.columns
                     else "delta_correct_logprob" if "delta_correct_logprob" in raw_df.columns
                     else None)
        if delta_col is None:
            continue

        _rp(f"  [{mode}]")

        # Use validation split for layer/coef selection if available
        ref_df  = (raw_df[raw_df["split"] == "validation"]
                   if "split" in raw_df.columns and "validation" in raw_df["split"].values
                   else raw_df)
        cv_df   = raw_df   # always CV over all available questions
        cv_label = ("val+test questions" if "split" in raw_df.columns
                    and raw_df["split"].notna().any() else "all questions")

        ref_agg = agg_cf_continuous(ref_df) if mode == "CF" else agg_mcf(ref_df)
        if ref_agg.empty:
            continue

        best_row   = ref_agg.loc[ref_agg["mean_delta_correct"].idxmax()]
        best_layer = best_row["layer"]
        best_coef  = best_row["coef"]

        cell = cv_df[(cv_df["layer"] == best_layer) & (cv_df["coef"] == best_coef)].copy()
        if len(cell) < k:
            _rp(f"  Too few questions ({len(cell)}) at L{int(best_layer):02d}, "
                f"coef={best_coef:+.2f} for {k}-fold CV — skipping"); continue

        _rp(f"  Selected cell (val-best): L{int(best_layer):02d}, coef={best_coef:+.2f}  "
            f"n={len(cell)} ({cv_label})")
        _rp()

        has_wrong = (mode == "CF"
                     and "delta_target_sum_lp" in cell.columns
                     and "delta_max_wrong_sum_lp" in cell.columns)

        # ── k-fold ──────────────────────────────────────────────────────────
        rng = np.random.default_rng(42)
        qids = cell["question_id"].unique()
        rng.shuffle(qids)
        folds = np.array_split(qids, k)

        fold_deltas: List[float] = []
        fold_sels:   List[float] = []

        hdr = f"  {'Fold':>4}  {'n_q':>4}  {'Δ-correct':>10}"
        if has_wrong: hdr += f"  {'Selectivity':>12}"
        _rp(hdr); _hr("·")

        for fi, held_qids in enumerate(folds, 1):
            fsub = cell[cell["question_id"].isin(held_qids)]
            md   = fsub[delta_col].mean()
            fold_deltas.append(md)
            line = f"  {fi:>4}  {len(fsub):>4}  {_fv(md):>10}"
            if has_wrong:
                dc  = fsub["delta_target_sum_lp"].values
                dw  = fsub["delta_max_wrong_sum_lp"].values
                den = np.abs(dc) + np.abs(dw)
                sel = np.where(den > 1e-9, (dc - dw) / den, 0.0).mean()
                fold_sels.append(sel)
                line += f"  {_fv(sel, '+.3f'):>12}"
            _rp(line)

        _hr("·")
        mn_d, sd_d = np.mean(fold_deltas), np.std(fold_deltas, ddof=1)
        line = f"  {'Mean':>4}  {'':>4}  {_fv(mn_d):>10}"
        if has_wrong and fold_sels:
            mn_s, sd_s = np.mean(fold_sels), np.std(fold_sels, ddof=1)
            line += f"  {_fv(mn_s, '+.3f'):>12}"
        _rp(line)
        line = f"  {'±std':>4}  {'':>4}  {sd_d:>10.4f}"
        if has_wrong and fold_sels:
            line += f"  {sd_s:>12.4f}"
        _rp(line)

        _rp()
        n_pos     = sum(d > 0 for d in fold_deltas)
        cv_ratio  = sd_d / abs(mn_d) if abs(mn_d) > 1e-9 else float("inf")
        if n_pos == k:
            stab = f"STABLE ✓  Δ-correct positive in all {k}/{k} folds"
        elif n_pos >= int(k * 0.6):
            stab = f"MOSTLY STABLE  positive in {n_pos}/{k} folds"
        else:
            stab = f"UNSTABLE ✗  positive in only {n_pos}/{k} folds"
        _rp(f"  Stability : {stab}")
        _rp(f"  CV ratio  : std/|mean| = {cv_ratio:.2f}  "
            f"{'(consistent)' if cv_ratio < 0.5 else '(noisy — wide fold-to-fold variation)'}")

        # ── Bootstrap 95% CI ────────────────────────────────────────────────
        _rp()
        _rp(f"  Bootstrap 95% CI  (n={n_boot} resamples with replacement):")
        all_d = cell[delta_col].dropna().values
        boot_means = np.array([
            rng.choice(all_d, size=len(all_d), replace=True).mean()
            for _ in range(n_boot)
        ])
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        concl = ("fully positive → effect is reliable" if lo > 0
                 else "crosses zero  → effect is uncertain")
        _rp(f"    Δ-correct      : [{lo:+.3f}, {hi:+.3f}]  {concl}")

        if has_wrong:
            all_dc = cell["delta_target_sum_lp"].dropna().values
            all_dw = cell["delta_max_wrong_sum_lp"].dropna().values
            if len(all_dc) == len(all_dw) and len(all_dc) > 0:
                boot_sels = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(all_dc), size=len(all_dc))
                    dc2, dw2 = all_dc[idx], all_dw[idx]
                    den2 = np.abs(dc2) + np.abs(dw2)
                    boot_sels.append(np.where(den2 > 1e-9, (dc2 - dw2) / den2, 0.0).mean())
                s_lo, s_hi = np.percentile(boot_sels, [2.5, 97.5])
                s_concl = ("fully positive → target-aware effect confirmed"
                           if s_lo > 0 else "crosses zero  → selectivity uncertain")
                _rp(f"    Selectivity    : [{s_lo:+.3f}, {s_hi:+.3f}]  {s_concl}")
        _rp()


def print_report(exp_root: Path, mcf: pd.DataFrame, cf: pd.DataFrame,
                 mcf_agg: pd.DataFrame, cf_agg: pd.DataFrame):
    """Print the full structured text report and optionally save to report.txt."""
    global _REPORT_LINES
    _REPORT_LINES = []

    _rpt_overview(exp_root, mcf, cf, mcf_agg, cf_agg)
    _rpt_effect(cf_agg, mcf_agg)
    _rpt_coef_sweeps(cf_agg)
    _rpt_selectivity(cf_agg)
    _rpt_directionality(cf_agg, mcf_agg)
    _rpt_kl(cf_agg)
    _rpt_accuracy(mcf_agg)
    _rpt_sanity(cf_agg, mcf_agg)
    _rpt_split_comparison(cf, mcf)
    _rpt_crossval(cf, mcf, k=5, n_boot=500)
    _rpt_verdict(cf_agg, mcf_agg)

    if _OUT_DIR is not None:
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = _OUT_DIR / "report.txt"
        report_path.write_text("\n".join(_REPORT_LINES) + "\n")
        print(f"\n  Report saved → {report_path}")


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

    if _REPORT:
        print_report(exp_root, mcf, cf, mcf_agg, cf_agg)

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
    global _OUT_DIR, _SHOW, _REPORT

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
    p.add_argument("--report", action="store_true",
                   help="Print structured text report (no images needed — good for HPC)")
    args = p.parse_args()

    sections = ALL_SECTIONS if args.section == "all" else args.section.split(",")
    for s in sections:
        if s not in ALL_SECTIONS:
            sys.exit(f"Unknown section: {s!r}. Pick from {ALL_SECTIONS} or 'all'.")

    _SHOW   = not args.no_show and not args.save
    _REPORT = args.report

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
