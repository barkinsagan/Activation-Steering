"""
Weights & Biases integration helpers.

Soft-imports wandb so the project still runs without it installed. All public
functions accept `run=None` to mean "logging disabled" — callers can stay
unconditional.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def init_run(cfg) -> Optional[Any]:
    """Initialise a W&B run from an ExperimentConfig. Returns None if disabled."""
    if not cfg.wandb.enabled:
        return None
    if not _WANDB_AVAILABLE:
        print("[wandb] enabled in config but `wandb` package not installed — skipping.")
        return None

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.experiment_id,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        notes=cfg.wandb.notes or None,
        mode=cfg.wandb.mode,
        config=dataclasses.asdict(cfg),
        reinit=True,
    )
    print(f"[wandb] run initialised: {run.url if hasattr(run, 'url') else run.id}")
    return run


def finish_run(run) -> None:
    if run is None:
        return
    run.finish()


_SKIP_COLS = {"layer", "coef", "n", "n_questions"}

def make_layer_callback(run, prefix: str, summarize_fn) -> Optional[Callable]:
    """Build an `on_layer_complete(layer_idx, layer_df)` callback.

    After each layer, re-logs one matplotlib image per metric:
      x=layer, y=metric, one colored line per coef (diverging colormap: blue=negative, red=positive).
    """
    if run is None:
        return None

    accumulated: list = []

    def _callback(layer_idx: int, layer_df: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        try:
            summary = summarize_fn(layer_df)
        except Exception as e:
            print(f"[wandb] failed to summarise layer {layer_idx}: {e}")
            return

        summary = summary.copy()
        if "layer" not in summary.columns:
            summary["layer"] = layer_idx

        accumulated.append(summary)
        combined = pd.concat(accumulated, ignore_index=True)
        combined["coef"] = combined["coef"].apply(lambda c: f"{float(c):+g}")

        coef_strs = sorted(combined["coef"].unique(), key=lambda c: float(c))
        coef_floats = [float(c) for c in coef_strs]
        abs_max = max(abs(min(coef_floats)), abs(max(coef_floats))) or 1.0
        norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
        cmap = plt.cm.RdBu_r

        metrics = [c for c in combined.columns if c not in _SKIP_COLS]
        charts: Dict[str, Any] = {}
        for metric in metrics:
            col_data = combined[["layer", "coef", metric]].dropna()
            if col_data.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            for coef_str in coef_strs:
                grp = col_data[col_data["coef"] == coef_str].sort_values("layer")
                if grp.empty:
                    continue
                ax.plot(grp["layer"], grp[metric],
                        color=cmap(norm(float(coef_str))),
                        marker="o", markersize=3, lw=1.5, label=coef_str)
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric)
            ax.set_title(f"{prefix.upper()} — {metric} (layers 0–{layer_idx})")
            ax.legend(title="coef", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            charts[f"{prefix}/{metric}"] = wandb.Image(fig)
            plt.close(fig)

        if charts:
            run.log(charts)

    return _callback


def log_combined_summary(run, summary_df: pd.DataFrame, prefix: str) -> None:
    """One-shot log of an entire combined_summary table (used as a fallback if
    streaming callbacks weren't wired in). Logs each layer with step=layer_idx."""
    if run is None or summary_df is None or len(summary_df) == 0:
        return
    for layer_idx, grp in summary_df.groupby("layer"):
        _log_summary_rows(run, grp, prefix=prefix, step=int(layer_idx))


def log_artifact(run, path: Path, name: str, artifact_type: str = "results") -> None:
    if run is None or not path.exists():
        return
    art = wandb.Artifact(name=name, type=artifact_type)
    if path.is_dir():
        art.add_dir(str(path))
    else:
        art.add_file(str(path))
    run.log_artifact(art)


# =============================================================================
# Internals
# =============================================================================

def _log_summary_rows(run, summary_df: pd.DataFrame, prefix: str, step: int) -> None:
    """Log each (layer, coef) row as a single wandb step.
    Packs all coefs for a layer into one log call keyed by `<prefix>/coef_<coef>/<metric>`."""
    payload: Dict[str, Any] = {}

    for _, row in summary_df.iterrows():
        coef = float(row["coef"])
        coef_key = f"coef_{coef:+g}".replace("+", "p").replace("-", "n")
        for col, val in row.items():
            if col in ("layer", "coef"):
                continue
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            payload[f"{prefix}/{coef_key}/{col}"] = val

    if payload:
        payload[f"{prefix}/layer"] = step
        run.log(payload)


# =============================================================================
# End-of-sweep summary artifacts
# =============================================================================

_CF_CORRECT_COL: Dict[str, str] = {
    "none":      "correct_sum",
    "token":     "correct_mean",
    "character": "correct_char",
    "pmi":       "correct_pmi",
}
_CF_DELTA_COL: Dict[str, str] = {
    "none":      "delta_target_sum_lp",
    "token":     "delta_target_sum_lp",
    "character": "delta_target_char_norm_lp",
    "pmi":       "delta_target_sum_lp",
}
_CF_LOGPROB_COL: Dict[str, str] = {
    "none":      "target_sum_lp",
    "token":     "target_mean_lp",
    "character": "target_char_norm_lp",
    "pmi":       "target_sum_lp",
}


def log_final_summary(run, cfg, eval_df: pd.DataFrame, out_dir) -> None:
    """Log end-of-sweep summary artifacts for all active formulations.

    Artifacts (duplicated under mcf/ and cf/ namespaces):
      1. layer_summary   — table: layer, best_coef_val, val_acc, val_delta_mean,
                                  test_acc, test_delta_mean
      2. validation_curves — table: layer, coef, split, acc, delta_mean, n
                             (filterable by layer in the W&B UI)
      3. split_info      — table: split, n_rows, n_examples
      4. layer_overview  — plot: 4 traces (fixed coef + oracle) × (val + test)
    """
    if run is None or not _WANDB_AVAILABLE:
        return

    from pathlib import Path as _Path

    s = cfg.sweep
    out_dir = _Path(out_dir)
    has_split = "split" in eval_df.columns
    cf_results_exist = any((out_dir / "cf").glob("layer_*_results.csv"))

    split_table = _build_split_info_table(cfg, eval_df)

    if s.formulation in ("mcf", "both"):
        mcf_df = _load_mcf_results(out_dir / "mcf", eval_df)
        if mcf_df is not None:
            _log_formulation_artifacts(
                run=run,
                results_df=mcf_df,
                prefix="mcf",
                acc_col="correct",
                logprob_col="correct_label_logprob",
                delta_col="delta_correct_logprob",
                criterion=cfg.wandb.best_coef_criterion,
                split_table=split_table,
                has_split=has_split,
            )

    cf_dir = out_dir / "cf"
    if cf_dir.exists():
        all_cf_files = list(cf_dir.rglob("*"))
        print(f"[wandb] cf dir contents ({len(all_cf_files)} files): {[str(f.relative_to(cf_dir)) for f in all_cf_files[:20]]}")
    else:
        print(f"[wandb] cf dir does not exist: {cf_dir}")
    print(f"[wandb] cf_results_exist={cf_results_exist}, formulation={s.formulation}")
    if s.formulation in ("cf", "both") and cf_results_exist:
        norm = s.cf_normalization
        cf_df = _load_cf_results(out_dir / "cf", eval_df)
        print(f"[wandb] cf_df: {len(cf_df) if cf_df is not None else None} rows, split col={'split' in cf_df.columns if cf_df is not None else 'N/A'}")
        if cf_df is not None:
            _log_formulation_artifacts(
                run=run,
                results_df=cf_df,
                prefix="cf",
                acc_col=_CF_CORRECT_COL.get(norm, "correct_char"),
                logprob_col=_CF_LOGPROB_COL.get(norm, "target_char_norm_lp"),
                delta_col=_CF_DELTA_COL.get(norm, "delta_target_char_norm_lp"),
                criterion=cfg.wandb.best_coef_criterion,
                split_table=split_table,
                has_split=has_split,
            )



def _log_formulation_artifacts(
    run,
    results_df: pd.DataFrame,
    prefix: str,
    acc_col: str,
    logprob_col: str,
    delta_col: str,
    criterion: str,
    split_table,
    has_split: bool,
) -> None:
    try:
        print(f"[wandb] {prefix}: {len(results_df)} rows, columns: {list(results_df.columns)}")
        print(f"[wandb] {prefix}: split values = {results_df['split'].unique().tolist() if 'split' in results_df.columns else 'NO SPLIT COLUMN'}")
        charts = _build_oracle_charts(results_df, acc_col, prefix)
        print(f"[wandb] {prefix}: oracle charts generated = {list(charts.keys())}")
        if charts:
            run.log(charts)
    except Exception as e:
        print(f"[wandb] oracle charts failed ({prefix}): {e}")


def _build_oracle_charts(
    results_df: pd.DataFrame,
    acc_col: str,
    prefix: str,
) -> Dict[str, Any]:
    """Two charts per formulation:
      1. best_coef_per_layer  — x=layer, y=best coef from validation (bar, red=pos/blue=neg)
      2. test_oracle_gain     — x=layer, y=test_acc(oracle coef) − test_baseline
    """
    import matplotlib.pyplot as plt

    out: Dict[str, Any] = {}
    if not _WANDB_AVAILABLE or results_df.empty or "split" not in results_df.columns:
        return out
    if acc_col not in results_df.columns:
        return out

    val_df  = results_df[results_df["split"] == "validation"]
    test_df = results_df[results_df["split"] == "test"]
    if val_df.empty or test_df.empty:
        return out

    val_non_base = val_df[val_df["coef"] != 0.0]
    if val_non_base.empty:
        return out

    val_per = val_non_base.groupby(["layer", "coef"])[acc_col].mean().reset_index()
    best_rows = val_per.loc[val_per.groupby("layer")[acc_col].idxmax()].copy()
    best_rows["layer"] = best_rows["layer"].astype(int)
    best_rows["coef"]  = best_rows["coef"].astype(float)
    best_rows = best_rows.sort_values("layer")

    layers    = best_rows["layer"].tolist()
    best_coef = best_rows["coef"].tolist()

    # ── Chart 1: best coef per layer ──────────────────────────────────────────
    colors = ["tomato" if c > 0 else ("steelblue" if c < 0 else "gray") for c in best_coef]
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(layers, best_coef, color=colors, width=0.8)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Best Coefficient")
    ax1.set_title(f"{prefix.upper()} — Best Coefficient per Layer  (selected on Validation)")
    ax1.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out[f"analysis/{prefix}/best_coef_per_layer"] = wandb.Image(fig1)
    plt.close(fig1)

    # ── Chart 2: oracle test gain per layer ───────────────────────────────────
    test_per = test_df.groupby(["layer", "coef"])[acc_col].mean().reset_index()
    oracle_map = dict(zip(best_rows["layer"], best_rows["coef"]))

    def _get(layer, coef):
        row = test_per[(test_per["layer"] == layer) & (test_per["coef"] == coef)]
        return float(row[acc_col].values[0]) if len(row) else float("nan")

    deltas = []
    for layer in layers:
        steered  = _get(layer, oracle_map[layer])
        baseline = _get(layer, 0.0)
        delta = steered - baseline
        if math.isnan(steered) or math.isnan(baseline):
            delta = float("nan")
        deltas.append(delta)

    valid = [(l, d) for l, d in zip(layers, deltas) if not math.isnan(d)]
    if not valid:
        return out
    vl, vd = zip(*valid)

    bar_colors = ["tomato" if d >= 0 else "steelblue" for d in vd]
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(vl, vd, color=bar_colors, width=0.8,
            label=None)
    ax2.axhline(0, color="black", lw=1.0)
    # legend patches
    import matplotlib.patches as mpatches
    ax2.legend(handles=[
        mpatches.Patch(color="tomato",    label="positive gain (steered > baseline)"),
        mpatches.Patch(color="steelblue", label="negative gain (steered < baseline)"),
    ], fontsize=9)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Steered − Baseline Accuracy")
    ax2.set_title(f"{prefix.upper()} — Test Gain per Layer  (oracle: best-val coef per layer)")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out[f"analysis/{prefix}/test_oracle_gain_per_layer"] = wandb.Image(fig2)
    plt.close(fig2)

    return out


def _log_layer_sweep_charts(
    run,
    results_df: pd.DataFrame,
    prefix: str,
    acc_col: str,
    logprob_col: str,
    delta_col: str,
) -> None:
    """Log x=layer charts — one matplotlib image per metric, one colored line per coef.
    Reproduces what the streaming callback logs during the sweep, but from full results.
    """
    import matplotlib.pyplot as plt

    if not _WANDB_AVAILABLE or results_df.empty:
        return

    metric_cols = [c for c in [acc_col, logprob_col, delta_col]
                   if c and c in results_df.columns]
    if not metric_cols:
        return

    agg = results_df.groupby(["layer", "coef"])[metric_cols].mean().reset_index()

    coef_vals = sorted(agg["coef"].unique(), key=float)
    abs_max = max(abs(min(coef_vals)), abs(max(coef_vals))) or 1.0
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = plt.cm.RdBu_r

    charts: Dict[str, Any] = {}
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for coef in coef_vals:
            grp = agg[agg["coef"] == coef].sort_values("layer")
            if grp.empty:
                continue
            ax.plot(grp["layer"], grp[metric],
                    color=cmap(norm(float(coef))),
                    marker="o", markersize=3, lw=1.5, label=f"{float(coef):+g}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.set_title(f"{prefix.upper()} — {metric}")
        ax.legend(title="coef", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        charts[f"{prefix}/{metric}"] = wandb.Image(fig)
        plt.close(fig)

    if charts:
        run.log(charts)


def _apply_split(df: pd.DataFrame, eval_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Add a 'split' column to df from eval_df or split_manifest.csv fallback."""
    if "split" in eval_df.columns:
        split_map: Dict[int, str] = eval_df["split"].to_dict()
        df["split"] = df["question_id"].map(split_map).fillna("unknown")
        return df
    manifest = results_dir / "split_manifest.csv"
    print(f"[wandb] manifest path: {manifest}  exists={manifest.exists()}")
    if manifest.exists():
        mdf = pd.read_csv(manifest)
        print(f"[wandb] manifest columns: {list(mdf.columns)}, rows: {len(mdf)}")
        id_col = next((c for c in ("question_id", "eval_question_id") if c in mdf.columns), None)
        if id_col and "split" in mdf.columns:
            split_map = mdf.set_index(id_col)["split"].to_dict()
            df["split"] = df["question_id"].map(split_map).fillna("unknown")
            print(f"[wandb] split loaded via '{id_col}': {df['split'].value_counts().to_dict()}")
        else:
            print(f"[wandb] manifest missing expected columns — skipping")
    return df


def _load_mcf_results(mcf_dir, eval_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    paths = sorted(Path(mcf_dir).glob("layer_*_results.csv"))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return _apply_split(df, eval_df, Path(mcf_dir).parent)


def _load_cf_results(cf_dir, eval_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    paths = sorted(Path(cf_dir).glob("layer_*_results.csv"))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return _apply_split(df, eval_df, Path(cf_dir).parent)


def _compute_split_summary(
    df: pd.DataFrame, acc_col: str, delta_col: str
) -> pd.DataFrame:
    rows = []
    for (layer, coef), grp in df.groupby(["layer", "coef"]):
        row: Dict[str, Any] = {
            "layer": int(layer),
            "coef":  float(coef),
            "n":     int(len(grp)),
            "acc":   float(grp[acc_col].mean()) if acc_col in grp.columns else float("nan"),
            "delta_mean": float(grp[delta_col].mean())
                          if (coef != 0.0 and delta_col in grp.columns)
                          else 0.0,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["layer", "coef"]).reset_index(drop=True)


def _select_best_coef(val_summary: pd.DataFrame, criterion: str) -> pd.DataFrame:
    non_base = val_summary[val_summary["coef"] != 0.0]
    if non_base.empty:
        return pd.DataFrame(columns=["layer", "best_coef", "val_acc", "val_delta_mean"])
    sort_col     = "acc"        if criterion == "val_acc"  else "delta_mean"
    tiebreak_col = "delta_mean" if criterion == "val_acc"  else "acc"
    rows = []
    for layer, grp in non_base.groupby("layer"):
        best = grp.sort_values([sort_col, tiebreak_col], ascending=[False, False]).iloc[0]
        rows.append({
            "layer":          int(layer),
            "best_coef":      float(best["coef"]),
            "val_acc":        float(best["acc"]),
            "val_delta_mean": float(best["delta_mean"]),
        })
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def _build_layer_coef_table(
    results_df: pd.DataFrame,
    acc_col: str,
    logprob_col: str,
    delta_logprob_col: str,
) -> Optional[Any]:
    """Table: one row per (layer, coef[, split]).
    Shows baseline acc/logprob alongside steered acc/logprob and their deltas.
    Baseline values are the coef=0 row for the same (layer, split) group.
    """
    if not _WANDB_AVAILABLE or results_df.empty:
        return None

    has_split = "split" in results_df.columns
    group_keys = ["layer", "split"] if has_split else ["layer"]

    rows = []
    for group_vals, grp in results_df.groupby(group_keys):
        if has_split:
            layer, split = group_vals
        else:
            layer, split = group_vals, None

        base = grp[grp["coef"] == 0.0]
        base_acc = float(base[acc_col].mean()) if (not base.empty and acc_col in base.columns) else float("nan")
        base_lp  = float(base[logprob_col].mean()) if (not base.empty and logprob_col in base.columns) else float("nan")

        for coef, coef_grp in grp.groupby("coef"):
            coef = float(coef)
            acc  = float(coef_grp[acc_col].mean()) if acc_col in coef_grp.columns else float("nan")
            lp   = float(coef_grp[logprob_col].mean()) if logprob_col in coef_grp.columns else float("nan")

            if coef == 0.0:
                acc_delta = 0.0
                delta_lp  = 0.0
            else:
                acc_delta = (acc - base_acc) if not (math.isnan(acc) or math.isnan(base_acc)) else float("nan")
                delta_lp  = float(coef_grp[delta_logprob_col].mean()) if delta_logprob_col in coef_grp.columns else float("nan")

            row: Dict[str, Any] = {
                "layer":             int(layer),
                "coef":              coef,
                "baseline_acc":      round(base_acc, 4),
                "acc":               round(acc, 4),
                "acc_delta":         round(acc_delta, 4) if not math.isnan(acc_delta) else float("nan"),
                "baseline_logprob":  round(base_lp, 4) if not math.isnan(base_lp) else float("nan"),
                "logprob":           round(lp, 4) if not math.isnan(lp) else float("nan"),
                "delta_logprob":     round(delta_lp, 4) if not math.isnan(delta_lp) else float("nan"),
            }
            if has_split:
                row["split"] = split
            rows.append(row)

    if not rows:
        return None

    cols = ["layer", "coef"] + (["split"] if has_split else []) + [
        "baseline_acc", "acc", "acc_delta",
        "baseline_logprob", "logprob", "delta_logprob",
    ]
    return wandb.Table(
        columns=cols,
        data=[[r.get(c, float("nan")) for c in cols] for r in rows],
    )


def _build_layer_summary_table(
    val_sum: pd.DataFrame,
    test_sum: pd.DataFrame,
    best_coef_df: pd.DataFrame,
) -> Optional[Any]:
    if not _WANDB_AVAILABLE or best_coef_df.empty:
        return None

    def _get(df: pd.DataFrame, layer: int, coef: float, col: str) -> float:
        row = df[(df["layer"] == layer) & (df["coef"] == coef)]
        return round(float(row[col].values[0]), 4) if len(row) else float("nan")

    rows = []
    for _, bc in best_coef_df.iterrows():
        layer = int(bc["layer"])
        coef  = float(bc["best_coef"])
        rows.append([
            layer, coef,
            _get(val_sum,  layer, coef, "acc"),
            _get(val_sum,  layer, coef, "delta_mean"),
            _get(test_sum, layer, coef, "acc"),
            _get(test_sum, layer, coef, "delta_mean"),
        ])
    if not rows:
        return None
    return wandb.Table(
        columns=["layer", "best_coef_val", "val_acc", "val_delta_mean",
                 "test_acc", "test_delta_mean"],
        data=rows,
    )


def _build_validation_curves_table(
    val_sum: pd.DataFrame, test_sum: pd.DataFrame
) -> Optional[Any]:
    if not _WANDB_AVAILABLE:
        return None
    val_tagged  = val_sum.copy();  val_tagged["split"]  = "validation"
    test_tagged = test_sum.copy(); test_tagged["split"] = "test"
    combined = pd.concat([val_tagged, test_tagged], ignore_index=True)
    keep = [c for c in ("layer", "coef", "split", "acc", "delta_mean", "n")
            if c in combined.columns]
    return wandb.Table(dataframe=combined[keep].round(4))


def _build_split_info_table(cfg, eval_df: pd.DataFrame) -> Optional[Any]:
    if not _WANDB_AVAILABLE:
        return None
    n_examples = cfg.sweep.n_examples if cfg.sweep.generate_examples else 0
    if cfg.dataset.split is not None and "split" in eval_df.columns:
        sp = cfg.dataset.split
        pct_eval = sp.validation + sp.test
        n_total  = round(len(eval_df) * 100 / pct_eval) if pct_eval > 0 else len(eval_df)
        rows = [
            ["steering",   n_total - len(eval_df),                    0],
            ["validation", int((eval_df["split"] == "validation").sum()), n_examples],
            ["test",       int((eval_df["split"] == "test").sum()),      0],
        ]
    else:
        rows = [["all", len(eval_df), n_examples]]
    return wandb.Table(columns=["split", "n_rows", "n_examples"], data=rows)


def _build_layer_overview_plot(
    val_sum: pd.DataFrame,
    test_sum: pd.DataFrame,
    best_coef_df: pd.DataFrame,
    prefix: str,
) -> Optional[Any]:
    if not _WANDB_AVAILABLE or best_coef_df.empty:
        return None

    import matplotlib.pyplot as plt

    layers = sorted(val_sum["layer"].unique().tolist())

    non_base = val_sum[val_sum["coef"] != 0.0]
    if non_base.empty:
        return None
    global_best_coef = float(non_base.groupby("coef")["acc"].mean().idxmax())

    def _acc(df: pd.DataFrame, layer: int, coef: float) -> float:
        row = df[(df["layer"] == layer) & (df["coef"] == coef)]
        return float(row["acc"].values[0]) if len(row) else float("nan")

    bc_map = dict(zip(
        best_coef_df["layer"].astype(int),
        best_coef_df["best_coef"].astype(float),
    ))

    val_fixed   = [_acc(val_sum,  l, global_best_coef) for l in layers]
    test_fixed  = [_acc(test_sum, l, global_best_coef) for l in layers]
    val_oracle  = [_acc(val_sum,  l, bc_map.get(l, global_best_coef)) for l in layers]
    test_oracle = [_acc(test_sum, l, bc_map.get(l, global_best_coef)) for l in layers]
    val_base    = [_acc(val_sum,  l, 0.0) for l in layers]
    test_base   = [_acc(test_sum, l, 0.0) for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, val_fixed,   color="steelblue", marker="o", lw=2,
            label=f"val  fixed coef={global_best_coef:+g}")
    ax.plot(layers, test_fixed,  color="tomato",    marker="o", lw=2,
            label=f"test fixed coef={global_best_coef:+g}")
    ax.plot(layers, val_oracle,  color="steelblue", marker="^", lw=2, ls="--",
            label="val  oracle (per-layer best)")
    ax.plot(layers, test_oracle, color="tomato",    marker="^", lw=2, ls="--",
            label="test oracle (per-layer best val coef)")
    ax.plot(layers, val_base,    color="steelblue", lw=1,       ls=":", alpha=0.45,
            label="val  baseline")
    ax.plot(layers, test_base,   color="tomato",    lw=1,       ls=":", alpha=0.45,
            label="test baseline")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{prefix.upper()} — Val & Test Accuracy Across Layers")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _build_metric_by_coef_charts(
    results_df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    prefix: str,
    splits: Optional[list] = None,
) -> Dict[str, Any]:
    """x=coef, y=metric_col, one line per layer colored by depth (viridis colorbar).
    `splits`: list of (split_value, split_label) tuples; auto-detects val/test if None.
    """
    import matplotlib.pyplot as plt

    out: Dict[str, Any] = {}
    if not _WANDB_AVAILABLE or results_df.empty or metric_col not in results_df.columns:
        return out

    if splits is None:
        has_split = "split" in results_df.columns and {"validation", "test"}.issubset(results_df["split"].unique())
        splits = [("validation", "Validation"), ("test", "Test")] if has_split else [(None, "All")]

    for split_val, split_label in splits:
        df = results_df[results_df["split"] == split_val] if split_val else results_df
        if df.empty:
            continue
        summary = (
            df.groupby(["layer", "coef"])[metric_col]
            .mean()
            .reset_index()
            .rename(columns={metric_col: y_label})
            .round(4)
        )
        layers = sorted(summary["layer"].astype(int).unique())
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(layers), vmax=max(layers))

        fig, ax = plt.subplots(figsize=(12, 6))
        for layer in layers:
            grp = summary[summary["layer"] == layer].sort_values("coef")
            if grp.empty:
                continue
            ax.plot(grp["coef"], grp[y_label],
                    color=cmap(norm(layer)), marker="o", markersize=3, lw=1.5)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Layer")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("Coefficient")
        ax.set_ylabel(y_label)
        ax.set_title(f"{prefix.upper()} — {y_label} by Coef ({split_label})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        key = f"{prefix}/{y_label}_by_coef_{split_val}" if split_val else f"{prefix}/{y_label}_by_coef"
        out[key] = wandb.Image(fig)
        plt.close(fig)

    return out


def _build_grouped_layer_charts(
    results_df: pd.DataFrame,
    acc_col: str,
    delta_col: str,
    prefix: str,
    group_size: int = 10,
) -> Dict[str, Any]:
    """x=coef, one line per layer-group (e.g. L0-9, L10-19 …), each a distinct color.
    Logged for validation split only (or all data if no split).
    """
    import matplotlib.pyplot as plt

    out: Dict[str, Any] = {}
    if not _WANDB_AVAILABLE or results_df.empty:
        return out

    df = results_df[results_df["split"] == "validation"] if "split" in results_df.columns else results_df
    if df.empty:
        return out

    df = df.copy()
    base = (df["layer"] // group_size) * group_size
    df["layer_group"] = base.apply(lambda x: f"L{int(x)}-{int(x) + group_size - 1}")
    groups = sorted(df["layer_group"].unique(), key=lambda g: int(g[1:].split("-")[0]))
    colors = plt.cm.tab10.colors

    for metric_col, y_label in [(acc_col, "acc"), (delta_col, "delta_logprob")]:
        if metric_col not in df.columns:
            continue
        src = df if y_label == "acc" else df[df["coef"] != 0.0]
        summary = (
            src.groupby(["layer_group", "coef"])[metric_col]
            .mean()
            .reset_index()
            .rename(columns={metric_col: y_label})
            .round(4)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, group in enumerate(groups):
            grp = summary[summary["layer_group"] == group].sort_values("coef")
            if grp.empty:
                continue
            ax.plot(grp["coef"], grp[y_label],
                    color=colors[i % len(colors)], marker="o", markersize=4, lw=1.5, label=group)
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("Coefficient")
        ax.set_ylabel(y_label)
        ax.set_title(f"{prefix.upper()} — {y_label} by Coef, Grouped Layers (Validation)")
        ax.legend(title="Layer Group", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out[f"{prefix}/grouped_{y_label}_by_coef"] = wandb.Image(fig)
        plt.close(fig)

    return out


def _build_best_coef_table(
    results_df: pd.DataFrame,
    acc_col: str,
    delta_col: str,
) -> Optional[Any]:
    """Table: for each layer, which coef gave highest validation acc, and its val_acc + val_delta_logprob."""
    if not _WANDB_AVAILABLE or results_df.empty or acc_col not in results_df.columns:
        return None

    df = results_df[results_df["split"] == "validation"] if "split" in results_df.columns else results_df
    non_base = df[df["coef"] != 0.0]
    if non_base.empty:
        return None

    agg_cols = {acc_col: "mean"}
    if delta_col in non_base.columns:
        agg_cols[delta_col] = "mean"

    per_layer_coef = non_base.groupby(["layer", "coef"]).agg(agg_cols).reset_index()
    best_idx = per_layer_coef.groupby("layer")[acc_col].idxmax()
    best = per_layer_coef.loc[best_idx].sort_values("layer").reset_index(drop=True)

    cols = ["layer", "coef", acc_col] + ([delta_col] if delta_col in best.columns else [])
    rename = {acc_col: "val_acc", delta_col: "val_delta_logprob"}
    best = best[cols].rename(columns=rename).round(4)

    return wandb.Table(dataframe=best)


def log_analysis_dashboard(run, out_dir: Path) -> None:
    """Run the analysis dashboard and log all plots as images under analysis/."""
    if run is None or not _WANDB_AVAILABLE:
        return
    try:
        import sys
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from analysis.analysis_dashboard import run_all_plots
        print("[wandb] generating analysis dashboard plots …")
        figures = run_all_plots(Path(out_dir))
        if figures:
            run.log({f"analysis/{name}": img for name, img in figures.items()})
            print(f"[wandb] logged {len(figures)} analysis dashboard figures")
    except FileNotFoundError as e:
        print(f"[wandb] analysis dashboard skipped — data not found: {e}")
    except Exception as e:
        print(f"[wandb] analysis dashboard failed: {e}")


def _build_test_steering_plot(
    results_df: pd.DataFrame,
    acc_col: str,
    prefix: str,
) -> Optional[Any]:
    """x=layer, y=test_acc, 3 lines (each a distinct color):
      - baseline (coef=0 on test)
      - oracle   (best val coef per layer, evaluated on test)
      - fixed    (single globally-best val coef, evaluated on test)
    """
    if not _WANDB_AVAILABLE or results_df.empty or "split" not in results_df.columns:
        return None
    if acc_col not in results_df.columns:
        return None

    val_df  = results_df[results_df["split"] == "validation"]
    test_df = results_df[results_df["split"] == "test"]
    if val_df.empty or test_df.empty:
        return None

    val_non_base = val_df[val_df["coef"] != 0.0]
    if val_non_base.empty:
        return None
    val_per = val_non_base.groupby(["layer", "coef"])[acc_col].mean().reset_index()
    oracle_map = (
        val_per.loc[val_per.groupby("layer")[acc_col].idxmax()]
        .set_index("layer")["coef"]
        .astype(float)
        .to_dict()
    )
    global_best = float(val_per.groupby("coef")[acc_col].mean().idxmax())

    test_per = test_df.groupby(["layer", "coef"])[acc_col].mean().reset_index()

    def _acc(layer, coef):
        row = test_per[(test_per["layer"] == layer) & (test_per["coef"] == coef)]
        return round(float(row[acc_col].values[0]), 4) if len(row) else float("nan")

    layers = sorted(int(l) for l in test_df["layer"].unique())

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    specs = [
        ("baseline (coef=0)",                lambda l: _acc(l, 0.0),                          "gray",      "o", "--"),
        ("oracle (best val coef/layer)",      lambda l: _acc(l, oracle_map.get(l, global_best)), "steelblue", "^", "-"),
        (f"fixed best (coef={global_best:+g})", lambda l: _acc(l, global_best),               "tomato",    "o", "-"),
    ]
    for label, fn, color, marker, ls in specs:
        ys = [fn(l) for l in layers]
        ax.plot(layers, ys, color=color, marker=marker, ls=ls,
                markersize=4, lw=2, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{prefix.upper()} — Test: Baseline vs Steering")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img
