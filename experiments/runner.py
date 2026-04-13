"""
Experiment runner CLI.

Usage:
    python experiments/runner.py configs/my_experiment.yaml
    python experiments/runner.py configs/*.yaml          # run multiple sequentially
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from experiments.config import ExperimentConfig, load_config
from experiments.registry import load_eval_dataset, load_model, load_steering_prompts
from olmes.formatter import build_formatter


# =============================================================================
# Core runner
# =============================================================================

def run_experiment(cfg: ExperimentConfig):
    """Run a single experiment defined by cfg."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {cfg.experiment_id}")
    print(f"{'='*70}")

    # --- Setup output directory ---
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save a snapshot of the config used
    config_snapshot = out_dir / "config.yaml"
    # Re-read original config path from argv (best effort) or reconstruct
    _save_config_snapshot(cfg, config_snapshot)

    # --- Load model and data ---
    model, tokenizer = load_model(cfg)
    eval_df, false_cols = load_eval_dataset(cfg)
    pos_prompts, neg_prompts = load_steering_prompts(cfg)

    s = cfg.sweep

    # --- Build OLMES formatter (shared by both MCF and CF sweeps) ---
    formatter = build_formatter(
        task_prefix=s.task_prefix,
        num_shots=s.num_shots,
        fewshot_source=s.fewshot_source,
        shuffle_choices=s.shuffle_choices,
        seed=42,
    )

    run_mcf = s.formulation in ("mcf", "both")
    run_cf  = s.formulation in ("cf",  "both")

    has_false_targets = len(false_cols) > 0

    # --- MCF sweep ---
    if run_mcf:
        print(f"\n>>> Running MCF sweep")
        from single_token_completion_test import sweep_layers_mcf

        sweep_layers_mcf(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_df,
            positive_prompts=pos_prompts,
            negative_prompts=neg_prompts,
            coef_list=s.coef_list,
            formatter=formatter,
            out_dir=str(out_dir / "mcf"),
            layers=s.layers,
            token_position=s.token_position,
            normalize_vector=s.normalize_vector,
            norm_type=s.norm_type,
            layer_name_pattern=s.layer_name_pattern,
            verbose_every=s.verbose_every,
            resume=s.resume,
        )

    # --- CF sweep ---
    if run_cf:
        if has_false_targets:
            print(f"\n>>> Running CF sweep  (false cols: {false_cols})")
            from token_completion_test import sweep_layers_cf

            sweep_layers_cf(
                model=model,
                tokenizer=tokenizer,
                ml_test_df=eval_df,
                positive_prompts=pos_prompts,
                negative_prompts=neg_prompts,
                coef_list=s.coef_list,
                formatter=formatter,
                cf_normalization=s.cf_normalization,
                out_dir=str(out_dir / "cf"),
                layers=s.layers,
                token_position=s.token_position,
                normalize_vector=s.normalize_vector,
                norm_type=s.norm_type,
                layer_name_pattern=s.layer_name_pattern,
                verbose_every=s.verbose_every,
                resume=s.resume,
            )
        else:
            print(f"\n>>> Running CF sweep  (target only, no false cols)")
            _run_cf_target_only(
                model=model,
                tokenizer=tokenizer,
                eval_df=eval_df,
                pos_prompts=pos_prompts,
                neg_prompts=neg_prompts,
                formatter=formatter,
                cfg=cfg,
                out_dir=out_dir / "cf",
            )

    print(f"\n{'='*70}")
    print(f"DONE: {cfg.experiment_id}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*70}\n")


# =============================================================================
# Continuation scorer for target-only datasets (no false columns)
# =============================================================================

def _run_cf_target_only(model, tokenizer, eval_df, pos_prompts,
                        neg_prompts, formatter, cfg, out_dir):
    """
    Run continuation scoring when there are no false targets.
    Logs logprob of the full target string per (layer, question, coef).
    """
    import pandas as pd
    import torch
    from pathlib import Path
    from dataclasses import asdict

    from hook import ModelWithHooks
    from dim import DifferenceInMeansSteering
    from token_completion_test import ContinuationProbability

    s = cfg.sweep
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_with_hooks = ModelWithHooks(model)
    scorer = ContinuationProbability(
        forward_fn=model_with_hooks,
        tokenizer=tokenizer,
        max_length=s.max_length,
    )

    layers = s.layers
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    coefs = [0.0] + [c for c in s.coef_list if c != 0.0]

    all_records = []
    baselines = {}  # (layer, question_id) -> logprob at coef=0

    for layer_idx in layers:
        layer_result_path = out_path / f"layer_{layer_idx}_results.csv"

        if s.resume and layer_result_path.exists():
            print(f"Layer {layer_idx}: already done, loading from disk.")
            df = pd.read_csv(layer_result_path)
            all_records.append(df)
            for _, r in df[df["coef"] == 0.0].iterrows():
                baselines[(int(r["layer"]), int(r["question_id"]))] = float(r["sum_logprob"])
            continue

        print(f"\n{'='*60}\nLayer {layer_idx}\n{'='*60}")
        layer_name = s.layer_name_pattern.format(layer_idx=layer_idx)

        dim_steerer = DifferenceInMeansSteering(
            model_with_hooks=model_with_hooks,
            tokenizer=tokenizer,
            target_layer=layer_name,
            token_position=s.token_position,
        )
        dim_steerer.capture_positive_activations(pos_prompts)
        dim_steerer.capture_negative_activations(neg_prompts)
        steering_vector = dim_steerer.compute_steering_vector(
            normalize=s.normalize_vector,
            norm_type=s.norm_type,
        )

        layer_records = []

        for coef in coefs:
            print(f"\n  [Layer {layer_idx}] coef={coef}")
            if coef != 0.0:
                dim_steerer.apply_steering(steering_vector, coefficient=coef)
            try:
                for i, row in eval_df.iterrows():
                    rr = formatter.format_cf(row)
                    result = scorer.continuation_logprob(rr.prompt, rr.target)

                    key = (layer_idx, i)
                    if coef == 0.0:
                        baselines[key] = result.sum_logprob

                    base_lp = baselines.get(key, result.sum_logprob)
                    delta = result.sum_logprob - base_lp if coef != 0.0 else 0.0

                    layer_records.append({
                        "layer": layer_idx,
                        "question_id": i,
                        "coef": coef,
                        "prompt": rr.prompt,
                        "target_text": rr.target,
                        "token_count": result.token_count,
                        "sum_logprob": result.sum_logprob,
                        "mean_logprob": result.mean_logprob,
                        "char_norm_logprob": result.char_norm_logprob,
                        "delta_sum_logprob": delta,
                    })

                    if s.verbose_every and (i + 1) % s.verbose_every == 0:
                        print(f"    Processed {i + 1}/{len(eval_df)}")
            finally:
                if coef != 0.0:
                    dim_steerer.reset_steering()

        layer_df = pd.DataFrame(layer_records)
        layer_df.to_csv(layer_result_path, index=False)
        print(f"\n>>> Layer {layer_idx} COMPLETE — {len(layer_df)} records")

        all_records.append(layer_df)

        combined = pd.concat(all_records, ignore_index=True)
        combined.to_csv(out_path / "combined_results.csv", index=False)
        _save_continuation_summary(combined, out_path / "combined_summary.csv")

        dim_steerer.cleanup()

    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        combined.to_csv(out_path / "combined_results.csv", index=False)
        _save_continuation_summary(combined, out_path / "combined_summary.csv")
        print(f"Saved combined results to {out_path}")


def _save_continuation_summary(df: pd.DataFrame, path):
    rows = []
    for (layer, coef), grp in df.groupby(["layer", "coef"]):
        row = {
            "layer": int(layer),
            "coef": float(coef),
            "n": len(grp),
            "mean_sum_logprob": grp["sum_logprob"].mean(),
            "mean_mean_logprob": grp["mean_logprob"].mean(),
        }
        if coef != 0.0:
            row["mean_delta_sum_logprob"] = grp["delta_sum_logprob"].mean()
            row["pct_improved"] = (grp["delta_sum_logprob"] > 0).mean()
            row["pct_hurt"] = (grp["delta_sum_logprob"] < 0).mean()
        rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).sort_values(["layer", "coef"]).to_csv(path, index=False)


# =============================================================================
# Config snapshot
# =============================================================================

def _save_config_snapshot(cfg: ExperimentConfig, path: Path):
    """Save a YAML snapshot of the config used for this run."""
    import dataclasses
    snapshot = dataclasses.asdict(cfg)
    with open(path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run steering vector experiments from YAML config files."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Path(s) to YAML experiment config file(s)",
    )
    args = parser.parse_args()

    configs = []
    for pattern in args.configs:
        matched = list(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        configs.extend(matched)

    if not configs:
        print("No config files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(configs)} experiment(s): {[c.name for c in configs]}")

    for config_path in configs:
        cfg = load_config(str(config_path))
        run_experiment(cfg)


if __name__ == "__main__":
    main()
