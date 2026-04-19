#!/usr/bin/env python3
"""
analysis_dashboard.py — All steering-vector analysis plots in one script.

Plots generated
---------------
Effect magnitude
  01_heatmap_{MCF,CF}            layer × coef heatmap of mean Δlogprob
  02_layer_sweep_{MCF,CF}        x=layer, y=mean Δlogprob, line per coef
  03_coef_sweep_{MCF,CF}         x=coef, y=mean Δlogprob, line per top layer
  04_best_layer_bar_{MCF,CF}     top-k layers by |effect|, error bars

Direction & asymmetry
  05_improved_hurt_{MCF,CF}      stacked bar: % improved vs % hurt per layer
  06_asymmetry_{MCF,CF}          +coef vs −coef symmetry check
  07_coef0_drift_{MCF,CF}        coef=0 drift (bug detector)

Ranking behavior
  08_rank_violin_{MCF,CF}        rank-change violin per layer
  09_pct_rank1_{MCF,CF}          % questions reaching rank 1

Per-question structure
  10_baseline_vs_steered_MCF     scatter: baseline vs steered logprob
  11_question_layer_heatmap_MCF  question × layer Δlogprob heatmap
  12_steerable_resistant_MCF     top/bottom questions by mean effect

Cross-method
  15_best_layer_MCF_vs_CF        do methods agree on best layer?
  16_question_correlation        per-question effect correlation MCF vs CF

Usage
-----
    python analysis/analysis_dashboard.py                          # show only (default)
    python analysis/analysis_dashboard.py --save                   # save only
    python analysis/analysis_dashboard.py --save --out-dir /path   # save to custom dir
    python analysis/analysis_dashboard.py --save --show            # save + show
"""

import argparse
import glob
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_EXP_ROOT = (
    "results/exp_20260413_anatomy_llama8b_pilot-20260419T074949Z-3-001"
    "/exp_20260413_anatomy_llama8b_pilot"
)
DEFAULT_OUT_DIR = (
    "results/exp_20260413_anatomy_llama8b_pilot-20260419T074949Z-3-001"
    "/exp_20260413_anatomy_llama8b_pilot/analysis"
)

TOP_K_LAYERS = 5
TOP_K_QUESTIONS = 10
FOCUS_COEFS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
DRIFT_THRESHOLD = 0.05

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_mcf(exp_root: Path) -> pd.DataFrame:
    """Load all MCF per-question data from per-layer CSVs."""
    dfs = []
    for f in sorted(glob.glob(str(exp_root / "mcf" / "layer_*_results.csv"))):
        layer = int(Path(f).stem.split("layer_")[1].split("_")[0])
        df = pd.read_csv(f)
        df["layer"] = layer
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No MCF layer files in {exp_root / 'mcf'}")
    return pd.concat(dfs, ignore_index=True)


def load_cf_wide(exp_root: Path) -> pd.DataFrame:
    """Load all CF detailed_wide per-question data from per-layer dirs."""
    dfs = []
    layer_dirs = sorted(
        [d for d in (exp_root / "cf").iterdir() if d.is_dir() and d.name.startswith("layer_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for d in layer_dirs:
        layer = int(d.name.split("_")[1])
        wide = d / "detailed_wide.csv"
        if not wide.exists():
            warnings.warn(f"Missing detailed_wide.csv for CF {d.name} — skipping")
            continue
        df = pd.read_csv(wide)
        df["layer"] = layer
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CF layer dirs in {exp_root / 'cf'}")
    return pd.concat(dfs, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def agg_mcf(mcf: pd.DataFrame) -> pd.DataFrame:
    g = mcf.groupby(["layer", "coef"])
    return g.agg(
        accuracy=("correct", "mean"),
        mean_delta=("delta_correct_logprob", "mean"),
        std_delta=("delta_correct_logprob", "std"),
        pct_improved=("delta_correct_logprob", lambda x: (x > 0).mean()),
        pct_hurt=("delta_correct_logprob", lambda x: (x < 0).mean()),
        mean_rank_change=("rank_change", "mean"),
        pct_rank1=("correct_label_rank", lambda x: (x == 1).mean()),
        n=("correct", "count"),
    ).reset_index()


def agg_cf(cf: pd.DataFrame) -> pd.DataFrame:
    g = cf.groupby(["layer", "coef"])
    return g.agg(
        accuracy_sum=("correct_sum", "mean"),
        accuracy_char=("correct_char", "mean"),
        mean_delta=("delta_target_sum_lp", "mean"),
        std_delta=("delta_target_sum_lp", "std"),
        mean_delta_char=("delta_target_char_norm_lp", "mean"),
        pct_improved=("delta_target_sum_lp", lambda x: (x > 0).mean()),
        pct_hurt=("delta_target_sum_lp", lambda x: (x < 0).mean()),
        mean_rank_change=("rank_change_sum", "mean"),
        pct_rank1=("target_rank_sum", lambda x: (x == 1).mean()),
        n=("correct_sum", "count"),
    ).reset_index()


def find_best_pos_coef(agg: pd.DataFrame) -> float:
    """Coef > 0 with highest mean Δlogprob across all layers."""
    pos = agg[agg.coef > 0]
    return float(pos.groupby("coef")["mean_delta"].mean().idxmax())


# ══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ══════════════════════════════════════════════════════════════════════════════

# Set by main() based on CLI flags
_out_dir: Path = None
_show: bool = True


def save(fig, name: str):
    if _out_dir is not None:
        _out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(_out_dir / f"{name}.png")
        print(f"  saved {name}.png")
    if _show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 01 — Heatmap: layer × coef → mean Δlogprob
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(agg: pd.DataFrame, mode: str):
    pivot = agg.pivot_table(index="coef", columns="layer", values="mean_delta")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        pivot, ax=ax, center=0, cmap="RdBu_r", linewidths=0.3,
        cbar_kws={"label": "Mean Δlogprob"},
    )
    ax.set_title(f"[{mode}] Layer × Coef — Mean Δlogprob")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Coefficient")
    save(fig, f"01_heatmap_{mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 02 — Layer sweep
# ══════════════════════════════════════════════════════════════════════════════

def plot_layer_sweep(agg: pd.DataFrame, mode: str):
    sub = agg[agg.coef.isin(FOCUS_COEFS)].sort_values("layer")
    fig, ax = plt.subplots(figsize=(13, 5))
    for coef, grp in sub.groupby("coef"):
        ax.plot(grp.layer, grp.mean_delta, marker="o", markersize=4, label=f"{coef:+.2f}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δlogprob")
    ax.set_title(f"[{mode}] Layer Sweep — Mean Δlogprob by Coefficient")
    ax.legend(title="Coef", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    save(fig, f"02_layer_sweep_{mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 03 — Coef sweep (best layers)
# ══════════════════════════════════════════════════════════════════════════════

def plot_coef_sweep(agg: pd.DataFrame, mode: str):
    layer_effect = agg.groupby("layer")["mean_delta"].apply(lambda x: x.abs().max())
    top_layers = layer_effect.nlargest(TOP_K_LAYERS).index.tolist()
    sub = agg[agg.layer.isin(top_layers)].sort_values("coef")
    fig, ax = plt.subplots(figsize=(13, 5))
    for layer, grp in sub.groupby("layer"):
        ax.plot(grp.coef, grp.mean_delta, marker="o", markersize=4, label=f"L{layer}")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Mean Δlogprob")
    ax.set_title(f"[{mode}] Coef Sweep — Top-{TOP_K_LAYERS} Layers")
    ax.legend(title="Layer", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    save(fig, f"03_coef_sweep_{mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 04 — Best-layer bar with error bars
# ══════════════════════════════════════════════════════════════════════════════

def plot_best_layer_bar(agg: pd.DataFrame, mode: str):
    bc = find_best_pos_coef(agg)
    sub = agg[agg.coef == bc].copy()
    sub["abs_delta"] = sub["mean_delta"].abs()
    top = sub.nlargest(TOP_K_LAYERS, "abs_delta").sort_values("mean_delta", ascending=False)
    se = top.std_delta / np.sqrt(top.n)
    colors = ["#d73027" if v > 0 else "#4575b4" for v in top.mean_delta]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(top.layer.astype(str), top.mean_delta, color=colors,
           yerr=se, capsize=5, error_kw={"lw": 1.5})
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δlogprob")
    ax.set_title(f"[{mode}] Top-{TOP_K_LAYERS} Layers by |Effect|  (coef={bc:+g})")
    save(fig, f"04_best_layer_bar_{mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 05 — % improved vs % hurt stacked bar
# ══════════════════════════════════════════════════════════════════════════════

def plot_improved_hurt(agg: pd.DataFrame, mode: str):
    bc = find_best_pos_coef(agg)
    sub = agg[agg.coef == bc].sort_values("layer")
    layers = sub.layer.astype(str)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(layers, sub.pct_improved * 100, label="% Improved", color="#2ca02c")
    ax.bar(layers, sub.pct_hurt * 100, bottom=sub.pct_improved * 100,
           label="% Hurt", color="#d62728")
    ax.axhline(50, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Layer")
    ax.set_ylabel("% of Questions")
    ax.set_title(f"[{mode}] % Improved vs % Hurt per Layer  (coef={bc:+g})")
    ax.legend()
    save(fig, f"05_improved_hurt_{mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 06 — +/− coef asymmetry
# ══════════════════════════════════════════════════════════════════════════════

def plot_asymmetry(agg: pd.DataFrame, mode: str):
    pos = agg[agg.coef > 0].copy().rename(columns={"mean_delta": "delta_pos", "coef": "abs_coef"})
    neg = agg[agg.coef < 0].copy()
    neg["abs_coef"] = neg.coef.abs()
    neg = neg.rename(columns={"mean_delta": "delta_neg"})
    merged = pos[["layer", "abs_coef", "delta_pos"]].merge(
        neg[["layer", "abs_coef", "delta_neg"]], on=["layer", "abs_coef"]
    )
    merged["asymmetry"] = merged.delta_pos + merged.delta_neg  # 0 = perfectly symmetric
    rep = [1.0, 2.0, 3.0]
    sub = merged[merged.abs_coef.isin(rep)].sort_values("layer")
    fig, ax = plt.subplots(figsize=(13, 5))
    for ac, grp in sub.groupby("abs_coef"):
        ax.plot(grp.layer, grp.asymmetry, marker="o", markersize=4, label=f"|coef|={ac:g}")
    ax.axhline(0, color="black", lw=0.8, ls="--", label="perfect symmetry")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ(+coef) + Δ(−coef)")
    ax.set_title(f"[{mode}] +/− Coef Asymmetry  (0 = symmetric)")
    ax.legend()
    save(fig, f"06_asymmetry_{mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 07 — Coef=0 drift check
# ══════════════════════════════════════════════════════════════════════════════

def plot_coef0_drift(agg: pd.DataFrame, mode: str):
    sub = agg[agg.coef == 0].sort_values("layer")
    colors = ["#d62728" if abs(v) > DRIFT_THRESHOLD else "#1f77b4" for v in sub.mean_delta]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(sub.layer.astype(str), sub.mean_delta, color=colors)
    ax.axhline(DRIFT_THRESHOLD, color="red", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-DRIFT_THRESHOLD, color="red", lw=0.8, ls="--", alpha=0.7,
               label=f"±threshold ({DRIFT_THRESHOLD})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δlogprob at coef=0")
    ax.set_title(f"[{mode}] Coef=0 Drift Check  (red = |drift| > {DRIFT_THRESHOLD})")
    ax.legend()
    save(fig, f"07_coef0_drift_{mode}")
    flagged = sub[sub.mean_delta.abs() > DRIFT_THRESHOLD].layer.tolist()
    if flagged:
        print(f"  !! [{mode}] coef=0 drift flagged at layers: {flagged}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 08 — Rank-change violin per layer
# ══════════════════════════════════════════════════════════════════════════════

def plot_rank_violin(df: pd.DataFrame, mode: str, rank_col: str, bc: float):
    sub = df[df.coef == bc][["layer", rank_col]].dropna()
    layers = sorted(sub.layer.unique())
    data = [sub[sub.layer == l][rank_col].values for l in layers]
    fig, ax = plt.subplots(figsize=(14, 5))
    parts = ax.violinplot(data, positions=layers, showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_alpha(0.55)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank Change")
    ax.set_title(f"[{mode}] Rank-Change Distribution per Layer  (coef={bc:+g})")
    save(fig, f"08_rank_violin_{mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 09 — % questions reaching rank 1
# ══════════════════════════════════════════════════════════════════════════════

def plot_pct_rank1(agg: pd.DataFrame, mode: str):
    focus = [c for c in FOCUS_COEFS if c != 0]
    sub = agg[agg.coef.isin(focus)].sort_values("layer")
    fig, ax = plt.subplots(figsize=(13, 5))
    for coef, grp in sub.groupby("coef"):
        ax.plot(grp.layer, grp.pct_rank1 * 100, marker="o", markersize=4, label=f"{coef:+.2f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("% Questions at Rank 1")
    ax.set_title(f"[{mode}] % Questions Reaching Rank 1 per Layer")
    ax.legend(title="Coef", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    save(fig, f"09_pct_rank1_{mode}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10 — Baseline vs steered scatter (MCF)
# ══════════════════════════════════════════════════════════════════════════════

def plot_baseline_vs_steered(mcf: pd.DataFrame, bc: float, subsample: int = 3000):
    base = mcf[mcf.coef == 0][["layer", "question_id", "correct_label_logprob"]].rename(
        columns={"correct_label_logprob": "lp_base"}
    )
    steer = mcf[mcf.coef == bc][["layer", "question_id", "correct_label_logprob"]].rename(
        columns={"correct_label_logprob": "lp_steer"}
    )
    merged = base.merge(steer, on=["layer", "question_id"])
    if len(merged) > subsample:
        merged = merged.sample(subsample, random_state=42)
    layers = sorted(merged.layer.unique())
    norm = plt.Normalize(min(layers), max(layers))
    cmap = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(merged.lp_base, merged.lp_steer,
                    c=merged.layer, cmap=cmap, norm=norm, alpha=0.25, s=6)
    lim = [min(merged.lp_base.min(), merged.lp_steer.min()),
           max(merged.lp_base.max(), merged.lp_steer.max())]
    ax.plot(lim, lim, "k--", lw=0.9, label="no change")
    plt.colorbar(sc, ax=ax, label="Layer")
    ax.set_xlabel("Baseline logprob  (coef=0)")
    ax.set_ylabel(f"Steered logprob  (coef={bc:+g})")
    ax.set_title(f"[MCF] Baseline vs Steered Logprob  (n={len(merged):,}, colored by layer)")
    ax.legend()
    save(fig, "10_baseline_vs_steered_MCF")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 11 — Question × layer heatmap (MCF)
# ══════════════════════════════════════════════════════════════════════════════

def plot_question_layer_heatmap(mcf: pd.DataFrame, bc: float, n_q: int = 100):
    sub = mcf[mcf.coef == bc][["question_id", "layer", "delta_correct_logprob"]]
    pivot = sub.pivot_table(index="question_id", columns="layer", values="delta_correct_logprob")
    pivot = pivot.dropna()
    top_q = pivot.var(axis=1).nlargest(n_q).index
    pivot_top = pivot.loc[top_q]
    vmax = pivot_top.abs().quantile(0.95).max()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot_top, ax=ax, center=0, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "Δlogprob"}, xticklabels=True, yticklabels=False)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Question (top {n_q} by variance)")
    ax.set_title(f"[MCF] Question × Layer Δlogprob Heatmap  (coef={bc:+g})")
    save(fig, "11_question_layer_heatmap_MCF")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 12 — Steerable vs resistant questions (MCF)
# ══════════════════════════════════════════════════════════════════════════════

def plot_steerable_resistant(mcf: pd.DataFrame, bc: float):
    per_q = (
        mcf[mcf.coef == bc]
        .groupby("question_id")["delta_correct_logprob"]
        .mean()
        .sort_values()
    )
    most_steerable = per_q.nlargest(TOP_K_QUESTIONS)
    most_resistant = per_q.nsmallest(TOP_K_QUESTIONS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(range(TOP_K_QUESTIONS), most_steerable.values[::-1], color="#2ca02c")
    axes[0].set_yticks(range(TOP_K_QUESTIONS))
    axes[0].set_yticklabels([f"Q{i}" for i in most_steerable.index[::-1]], fontsize=7)
    axes[0].set_xlabel("Mean Δlogprob")
    axes[0].set_title(f"Top {TOP_K_QUESTIONS} Most Steerable  (coef={bc:+g})")
    axes[1].barh(range(TOP_K_QUESTIONS), most_resistant.values, color="#d62728")
    axes[1].set_yticks(range(TOP_K_QUESTIONS))
    axes[1].set_yticklabels([f"Q{i}" for i in most_resistant.index], fontsize=7)
    axes[1].set_xlabel("Mean Δlogprob")
    axes[1].set_title(f"Top {TOP_K_QUESTIONS} Most Resistant  (coef={bc:+g})")
    plt.tight_layout()
    save(fig, "12_steerable_resistant_MCF")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 15 — Best layer: MCF vs CF agreement
# ══════════════════════════════════════════════════════════════════════════════

def plot_best_layer_comparison(mcf_s: pd.DataFrame, cf_s: pd.DataFrame):
    def best_layer_per_coef(agg):
        rows = []
        for coef, grp in agg.groupby("coef"):
            best_row = grp.loc[grp["mean_delta"].abs().idxmax()]
            rows.append({"coef": coef, "best_layer": best_row["layer"]})
        return pd.DataFrame(rows)

    mcf_bl = best_layer_per_coef(mcf_s)
    cf_bl = best_layer_per_coef(cf_s)
    merged = mcf_bl.merge(cf_bl, on="coef", suffixes=("_mcf", "_cf"))

    coefs = merged.coef.values
    x = np.arange(len(coefs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, merged.best_layer_mcf, width, label="MCF", color="#1f77b4")
    ax.bar(x + width / 2, merged.best_layer_cf, width, label="CF (sum)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:+g}" for c in coefs], rotation=45, fontsize=8)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Layer with max |Δlogprob|")
    ax.set_title("Best Layer per Coef: MCF vs CF — do they agree?")
    ax.legend()
    save(fig, "15_best_layer_MCF_vs_CF")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 16 — Per-question effect correlation MCF vs CF
# ══════════════════════════════════════════════════════════════════════════════

def plot_question_correlation(mcf: pd.DataFrame, cf: pd.DataFrame, bc: float,
                              subsample: int = 2000):
    mcf_q = (
        mcf[mcf.coef == bc]
        .groupby(["question_id", "layer"])["delta_correct_logprob"]
        .mean()
        .reset_index()
    )
    cf_q = (
        cf[cf.coef == bc]
        .groupby(["question_id", "layer"])["delta_target_sum_lp"]
        .mean()
        .reset_index()
    )
    merged = mcf_q.merge(cf_q, on=["question_id", "layer"])

    r = merged["delta_correct_logprob"].corr(merged["delta_target_sum_lp"])
    m, b = np.polyfit(merged["delta_correct_logprob"], merged["delta_target_sum_lp"], 1)

    if len(merged) > subsample:
        merged = merged.sample(subsample, random_state=42)

    x = merged["delta_correct_logprob"]
    y = merged["delta_target_sum_lp"]
    xl = np.linspace(x.min(), x.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.25, s=7, color="#1f77b4")
    ax.plot(xl, m * xl + b, "r-", lw=1.5, label=f"y = {m:.2f}x + {b:.2f}")
    ax.set_xlabel("MCF Δlogprob")
    ax.set_ylabel("CF Δlogprob (sum)")
    ax.set_title(f"Per-Question Effect Correlation  (coef={bc:+g},  r = {r:.3f})")
    ax.legend()
    save(fig, "16_question_correlation_MCF_vs_CF")
    print(f"  MCF vs CF correlation at coef={bc:+g}: r = {r:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _out_dir, _show

    parser = argparse.ArgumentParser(description="Steering vector analysis dashboard")
    parser.add_argument("--exp-root", default=DEFAULT_EXP_ROOT,
                        help="Path to folder that contains mcf/ and cf/ subdirs")
    parser.add_argument("--save", action="store_true",
                        help="Save plots as PNGs (default: off)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Directory for saved PNGs (requires --save)")
    parser.add_argument("--show", action="store_true", default=None,
                        help="Show plots interactively (default: on when --save is not set)")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if args.save:
        _out_dir = Path(args.out_dir) if args.out_dir else exp_root / "analysis"
    else:
        _out_dir = None
    # default: show=True unless --save is given without --show
    _show = args.show if args.show is not None else (not args.save)

    print("Loading MCF data...")
    mcf = load_mcf(exp_root)
    print(f"  {len(mcf):,} rows | layers: {sorted(mcf.layer.unique())}")

    print("Loading CF data...")
    cf = load_cf_wide(exp_root)
    print(f"  {len(cf):,} rows  | layers: {sorted(cf.layer.unique())}")

    print("Aggregating...")
    mcf_s = agg_mcf(mcf)
    cf_s = agg_cf(cf)

    bc_mcf = find_best_pos_coef(mcf_s)
    bc_cf = find_best_pos_coef(cf_s)
    bc_cross = min(bc_mcf, bc_cf)  # conservative shared coef for cross-plots
    print(f"  Best pos coef — MCF: {bc_mcf:+g}   CF: {bc_cf:+g}")

    print(f"\nGenerating plots → {_out_dir if _out_dir else '(show only)'}")

    # Effect magnitude
    print("── Effect magnitude")
    plot_heatmap(mcf_s, "MCF")
    plot_heatmap(cf_s, "CF")
    plot_layer_sweep(mcf_s, "MCF")
    plot_layer_sweep(cf_s, "CF")
    plot_coef_sweep(mcf_s, "MCF")
    plot_coef_sweep(cf_s, "CF")
    plot_best_layer_bar(mcf_s, "MCF")
    plot_best_layer_bar(cf_s, "CF")

    # Direction & asymmetry
    print("── Direction & asymmetry")
    plot_improved_hurt(mcf_s, "MCF")
    plot_improved_hurt(cf_s, "CF")
    plot_asymmetry(mcf_s, "MCF")
    plot_asymmetry(cf_s, "CF")
    plot_coef0_drift(mcf_s, "MCF")
    plot_coef0_drift(cf_s, "CF")

    # Ranking behavior
    print("── Ranking behavior")
    plot_rank_violin(mcf, "MCF", "rank_change", bc_mcf)
    plot_rank_violin(cf, "CF", "rank_change_sum", bc_cf)
    plot_pct_rank1(mcf_s, "MCF")
    plot_pct_rank1(cf_s, "CF")

    # Per-question structure
    print("── Per-question structure")
    plot_baseline_vs_steered(mcf, bc_mcf)
    plot_question_layer_heatmap(mcf, bc_mcf)
    plot_steerable_resistant(mcf, bc_mcf)

    # Cross-method
    print("── Cross-method MCF vs CF")
    plot_best_layer_comparison(mcf_s, cf_s)
    plot_question_correlation(mcf, cf, bc_cross)

    print("\nDone.")


if __name__ == "__main__":
    main()
