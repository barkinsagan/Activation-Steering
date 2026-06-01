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


def make_layer_callback(run, prefix: str, summarize_fn) -> Optional[Callable]:
    """Build an `on_layer_complete(layer_idx, layer_df)` callback that logs
    per-(layer, coef) summary metrics under `<prefix>/<metric>` keys.

    `summarize_fn(layer_df) -> pd.DataFrame` aggregates raw per-question records
    into one row per (layer, coef). Pass the existing _compute_summary functions.

    Returns None if `run` is None, so the runner can pass it unconditionally.
    """
    if run is None:
        return None

    def _callback(layer_idx: int, layer_df: pd.DataFrame) -> None:
        try:
            summary = summarize_fn(layer_df)
        except Exception as e:
            print(f"[wandb] failed to summarise layer {layer_idx}: {e}")
            return
        _log_summary_rows(run, summary, prefix=prefix, step=layer_idx)

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
    """Log each (layer, coef) row as a single wandb step. We pack all coefs for
    a layer into one log call keyed by `<prefix>/coef_<coef>/<metric>`, plus a
    flat table for sweep-wide queries."""
    payload: Dict[str, Any] = {}
    table_rows = []

    for _, row in summary_df.iterrows():
        coef = float(row["coef"])
        coef_key = f"coef_{coef:+g}".replace("+", "p").replace("-", "n")
        for col, val in row.items():
            if col in ("layer", "coef"):
                continue
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            payload[f"{prefix}/{coef_key}/{col}"] = val
        table_rows.append({k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in row.items()})

    if table_rows:
        cols = list(table_rows[0].keys())
        table = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in table_rows])
        payload[f"{prefix}/layer_table"] = table

    if payload:
        run.log(payload, step=step)


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
    has_false = any(c in eval_df.columns for c in ("false1", "false2", "false3"))

    split_table = _build_split_info_table(cfg, eval_df)

    if s.formulation in ("mcf", "both"):
        mcf_df = _load_mcf_results(out_dir / "mcf", eval_df)
        if mcf_df is not None:
            _log_formulation_artifacts(
                run=run,
                results_df=mcf_df,
                prefix="mcf",
                acc_col="correct",
                delta_col="delta_correct_logprob",
                criterion=cfg.wandb.best_coef_criterion,
                split_table=split_table,
                has_split=has_split,
            )

    if s.formulation in ("cf", "both") and has_false:
        norm = s.cf_normalization
        cf_df = _load_cf_results(out_dir / "cf", eval_df)
        if cf_df is not None:
            _log_formulation_artifacts(
                run=run,
                results_df=cf_df,
                prefix="cf",
                acc_col=_CF_CORRECT_COL.get(norm, "correct_char"),
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
    delta_col: str,
    criterion: str,
    split_table,
    has_split: bool,
) -> None:
    if split_table is not None:
        try:
            run.log({f"{prefix}/split_info": split_table})
        except Exception as e:
            print(f"[wandb] split_info log failed ({prefix}): {e}")

    if not has_split or "split" not in results_df.columns:
        return

    val_df  = results_df[results_df["split"] == "validation"]
    test_df = results_df[results_df["split"] == "test"]
    if val_df.empty or test_df.empty:
        return

    val_sum  = _compute_split_summary(val_df,  acc_col, delta_col)
    test_sum = _compute_split_summary(test_df, acc_col, delta_col)
    if val_sum.empty or test_sum.empty:
        return

    best_coef_df = _select_best_coef(val_sum, criterion)

    try:
        tbl = _build_layer_summary_table(val_sum, test_sum, best_coef_df)
        if tbl is not None:
            run.log({f"{prefix}/layer_summary": tbl})
    except Exception as e:
        print(f"[wandb] layer_summary log failed ({prefix}): {e}")

    try:
        curves_tbl = _build_validation_curves_table(val_sum, test_sum)
        if curves_tbl is not None:
            run.log({f"{prefix}/validation_curves": curves_tbl})
    except Exception as e:
        print(f"[wandb] validation_curves log failed ({prefix}): {e}")

    try:
        fig = _build_layer_overview_plot(val_sum, test_sum, best_coef_df, prefix)
        if fig is not None:
            run.log({f"{prefix}/layer_overview": wandb.Image(fig)})
    except Exception as e:
        print(f"[wandb] layer_overview plot failed ({prefix}): {e}")
    finally:
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass


def _load_mcf_results(mcf_dir, eval_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    from pathlib import Path as _Path
    paths = sorted(_Path(mcf_dir).glob("layer_*_results.csv"))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "split" in eval_df.columns:
        split_map: Dict[int, str] = eval_df["split"].to_dict()
        df["split"] = df["question_id"].map(split_map).fillna("unknown")
    return df


def _load_cf_results(cf_dir, eval_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    from pathlib import Path as _Path
    frames = []
    for layer_dir in sorted(_Path(cf_dir).glob("layer_*")):
        if not layer_dir.is_dir():
            continue
        try:
            layer_idx = int(layer_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        wide_path = layer_dir / "detailed_wide.csv"
        if not wide_path.exists():
            continue
        frame = pd.read_csv(wide_path)
        frame["layer"] = layer_idx
        frames.append(frame)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "split" in eval_df.columns:
        split_map = eval_df["split"].to_dict()
        combined["split"] = combined["question_id"].map(split_map).fillna("unknown")
    return combined


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
