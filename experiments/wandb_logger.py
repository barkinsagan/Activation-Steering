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
