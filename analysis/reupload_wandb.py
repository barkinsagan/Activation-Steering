#!/usr/bin/env python3
"""
Re-log all end-of-sweep WandB plots for a completed experiment.

Useful after changing wandb_logger.py to update an existing run with improved charts.
Resumes the original WandB run and re-logs everything: layer/coef tables, matplotlib
charts, analysis dashboard plots.

Usage
-----
    # Let the script find the run by experiment name (searches wandb project):
    python analysis/reupload_wandb.py --exp-dir results/exp_20260602_gpqa_physics_qwen32b

    # Or pass the run ID explicitly (find it in the WandB UI URL):
    python analysis/reupload_wandb.py --exp-dir results/exp_20260602_gpqa_physics_qwen32b \
        --run-id abc123xyz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make sure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import load_config
from experiments import wandb_logger


def find_run_id(project: str, experiment_id: str, entity: str | None) -> str | None:
    """Search the WandB project for a run whose name matches experiment_id."""
    try:
        import wandb
        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"display_name": experiment_id})
        for run in runs:
            return run.id
    except Exception as e:
        print(f"[reupload] wandb API search failed: {e}")
    return None


def build_eval_df(exp_dir: Path) -> pd.DataFrame:
    """Reconstruct a minimal eval_df with split info from split_manifest.csv."""
    manifest = exp_dir / "split_manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        if "split" in df.columns and "question_id" in df.columns:
            return df.set_index("question_id")[["split"]]
        print("[reupload] split_manifest.csv missing expected columns — split info unavailable")
    else:
        print("[reupload] no split_manifest.csv found — split info unavailable")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Re-upload WandB plots for a completed experiment")
    parser.add_argument("--exp-dir", required=True,
                        help="Path to the experiment results directory (contains config.yaml)")
    parser.add_argument("--run-id", default=None,
                        help="WandB run ID to resume (skip auto-search if provided)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"[reupload] config.yaml not found in {exp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[reupload] loading config from {config_path}")
    cfg = load_config(str(config_path))

    run_id = args.run_id
    if run_id is None:
        print(f"[reupload] searching WandB project '{cfg.wandb.project}' for run '{cfg.experiment_id}' …")
        run_id = find_run_id(cfg.wandb.project, cfg.experiment_id, cfg.wandb.entity or None)
        if run_id is None:
            print("[reupload] run not found — pass --run-id manually", file=sys.stderr)
            sys.exit(1)
        print(f"[reupload] found run ID: {run_id}")

    eval_df = build_eval_df(exp_dir)
    print(f"[reupload] eval_df: {len(eval_df)} rows, columns: {list(eval_df.columns)}")

    import wandb
    print(f"[reupload] resuming WandB run {run_id} …")
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        id=run_id,
        resume="must",
    )
    print(f"[reupload] run URL: {run.url}")

    print("[reupload] logging end-of-sweep artifacts …")
    wandb_logger.log_final_summary(run, cfg, eval_df, exp_dir)

    run.finish()
    print("[reupload] done.")


if __name__ == "__main__":
    main()
